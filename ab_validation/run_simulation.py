"""
AB Validation Simulation
========================
Runs the full simulation (warmup → observational → AB experiment),
fits IPW-CBPS estimators across multiple forward-spending horizons, and
saves a cache file that plot_figure.py script reads.

During the warmup + observational phases the same run also accumulates the
obs-phase diagnostics (lifecycle transitions, per-state delay/value, time-of-day
& day-of-week marketplace patterns, lifecycle composition over time) used for
Figures 1, 2, 3 and 6. 

Phases
------
  1. Warmup        (N_WARMUP days)   — reach market equilibrium; no data collected
  2. Observational (N_OBS_DAYS days) — record confounded order panel for CBPS
  3. AB experiment (N_AB_DAYS days)  — 50/50 randomized proactive-dispatch treatment

After the AB phase the script fits the three estimators (Naive, IPW-MLP,
IPW-CBPS) on the observational data for each forward horizon K, computes the
ANCOVA ground-truth ΔLTV% and generate main result table (Table 1).

Runtime:
  RUN_BOOTSTRAP=False : ~25 min (sim + full-sample point estimates) — figures + Table-1 point estimates
  RUN_BOOTSTRAP=True  : ~9 h    (adds the B=200 user-level bootstrap CIs for K in K_TABLE)

Output:
    ab_validation/cache/ab_validation_cache.npz
"""
from __future__ import annotations
import os
# Pin BLAS to a single thread before numpy/sklearn import. The MLP-IPW estimator
# runs many small minibatch matmuls; multi-threaded BLAS oversubscribes cores and
# thread-sync overhead dominates (~10× slowdown). Must be set before numpy loads.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import sys, time, warnings, multiprocessing as mp
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.neural_network import MLPClassifier

from sim.params import DGPParams
from sim.users import (generate_users, generate_users_warm,
                       update_daily_state, draw_lifecycle_transitions,
                       CHURNED, N_STATES)
from sim.demand import draw_daily_orders
from sim.marketplace import run_marketplace

# ── Simulation parameters ─────────────────────────────────────────────────────
SEED               = 123
LAMBDA_MEM         = 0.06    # delay-memory decay; half-life ≈ 11.5 days
SIGMA_DEMAND       = 0.10    # daily macro shock std
COURIER_SCALE_BASE = 700
P_TREAT            = 0.50    # 50/50 randomization
DISPATCH_ADVANCE   = 5.0     # predictive-dispatch advance (min); ETA unchanged
N_USERS            = 100_000
N_WARMUP           = 100
N_OBS_DAYS         = 150
N_AB_DAYS          = 120
K_ENTRY            = 28      # active-cohort window: first AB order within K_ENTRY days
LAST_K_DAYS        = 28      # last-4-week aggregate for β_AB ground truth
N_PRE_AB           = 28      # pre-AB spending lookback for ANCOVA covariate
DELTA_PI           = 0.01    # policy-gradient normalizer (1 pp delay change)
MAX_N              = 1_000_000
E_MATCH_ALPHA      = 0.05    # EWMA weight for avg_match_time (half-life ≈ 14 days)
N_WEEKS            = N_AB_DAYS // 7
K_SWEEP            = [28, 42, 56, 70]   # forward horizons for the Figure-4 Chronos lines (CBPS)
K_TABLE            = [56, 70]           # horizons reported in Table 1 (all three estimators)
B_BOOT             = 200                # user-level bootstrap reps for the Table-1 CIs
N_WORKERS          = 32                 # processes for the bootstrap
RUN_BOOTSTRAP      = False              # True → full Table 1 with CIs (~9h); False → point estimates only
STEADY_START       = 50      # obs day at which steady-state diagnostic accumulators begin
SLOTS_PER_HR       = 12      # marketplace 5-min slots per hour (288 slots/day)

OBS_START = N_WARMUP
OBS_END   = OBS_START + N_OBS_DAYS
AB_START  = OBS_END

CACHE_DIR  = Path(__file__).parent / "cache"
CACHE_PATH = CACHE_DIR / "ab_validation_cache.npz"

# ── CBPS helpers ──────────────────────────────────────────────────────────────
def _cbps_obj(Z, W, dpi):
    n = len(W)
    def _f(theta):
        z = Z @ theta; enZ = np.exp(np.clip(-z, -50., 50.))
        return (np.mean(dpi * (W * enZ + (1. - W) * z)),
                (Z.T @ (dpi * ((1. - W) - W * enZ))) / n)
    return _f

def _cbps_weights(Z, W, dpi):
    res = minimize(_cbps_obj(Z, W, dpi), np.zeros(Z.shape[1]),
                   method="BFGS", jac=True, options={"maxiter": 500, "gtol": 1e-6})
    return 1. + np.exp(np.clip(-Z @ res.x, -50., 50.))

def _policy_gradient(T, Y, spend, w0, w1):
    return float(((T == 1) * w1 * Y - (T == 0) * w0 * Y).sum() * DELTA_PI / spend.sum())

def _standardize(Z_raw):
    mu = Z_raw.mean(0); mu[0] = 0.
    sd = Z_raw.std(0);  sd[0] = 1.; sd[sd == 0] = 1.
    return (Z_raw - mu) / sd

def _fit_cbps(Z_raw, T, Y, spend):
    """Fit CBPS weights and return the policy-gradient estimate of β."""
    Z   = _standardize(Z_raw.copy())
    dpi = np.full(len(T), DELTA_PI)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        w0 = _cbps_weights(Z, (T == 0).astype(float), dpi)
        w1 = _cbps_weights(Z, (T == 1).astype(float), dpi)
    return _policy_gradient(T, Y, spend, w0, w1)

def _ipw_mlp(Z, T, Y, spend, seed):
    """IPW estimate with an MLP propensity model (the paper's MLP-IPW). Z must be standardized."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mlp = MLPClassifier(hidden_layer_sizes=(64, 32), activation="relu",
                            max_iter=300, random_state=seed, early_stopping=True,
                            validation_fraction=0.1, n_iter_no_change=15)
        mlp.fit(Z[:, 1:], T)
    p_hat = np.clip(mlp.predict_proba(Z[:, 1:])[:, 1], 0.01, 0.99)
    w1 = np.where(T == 1, 1. / p_hat,        0.)
    w0 = np.where(T == 0, 1. / (1. - p_hat), 0.)
    return _policy_gradient(T, Y, spend, w0, w1)

def _estimate(Z_raw, T, Y, spend, seed):
    """Table-1 point estimates on one sample: (β_naive, β_mlp_ipw, β_cbps)."""
    Z   = _standardize(Z_raw.copy())
    dpi = np.full(len(T), DELTA_PI)
    b_naive = _policy_gradient(T, Y, spend, np.ones(len(T)), np.ones(len(T)))
    b_ipw   = _ipw_mlp(Z, T, Y, spend, seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        w0 = _cbps_weights(Z, (T == 0).astype(float), dpi)
        w1 = _cbps_weights(Z, (T == 1).astype(float), dpi)
    b_cbps = _policy_gradient(T, Y, spend, w0, w1)
    return b_naive, b_ipw, b_cbps

# Bootstrap worker: resamples users with replacement (full sample) and re-estimates.
# Reads the per-K arrays from the module-level _BT dict (shared via fork on Linux).
_BT: dict = {}

def _boot_one(seed_b):
    rng_b = np.random.default_rng(seed_b)
    users = _BT["users"]
    bu    = rng_b.choice(users, size=len(users), replace=True)
    idx   = np.concatenate([_BT["u2r"][int(u)] for u in bu])
    return _estimate(_BT["Z"][idx], _BT["T"][idx], _BT["Y"][idx], _BT["spend"][idx], seed_b)

def _append(base, new):
    """Concatenate two latent/state dicts along axis 0."""
    return {k: np.concatenate([base[k], new[k]], 0) for k in base}

def _lc_share(state, n_now):
    """Population share in each lifecycle state (for the composition diagnostic)."""
    return np.array([(state["lifecycle"][:n_now] == s).sum() / max(n_now, 1)
                     for s in range(N_STATES)], dtype=np.float64)

def _update_delay_buffers(n, mkt, sim_day, buf30, sum30, buf7, ord_buf7, sum7, ord_sum7):
    """
    Update the rolling 30-day and 7-day delay-count buffers used to compute
    delay_rate_30d and delay_rate_7d features for CBPS propensity estimation.

    buf30 / sum30 : circular 30-slot buffer of daily delayed counts; sum = F30-delayed
    buf7  / sum7  : circular  7-slot buffer of daily delayed counts
    ord_buf7 / ord_sum7 : circular 7-slot buffer of daily matched counts (denominator)
    """
    delayed = mkt["delayed_per_user"][:n].astype(np.int32)
    matched = mkt["matched_per_user"][:n].astype(np.int32)
    s30 = sim_day % 30
    sum30[:n] -= buf30[:n, s30]; buf30[:n, s30] = delayed; sum30[:n] += delayed
    s7  = sim_day % 7
    sum7[:n]     -= buf7[:n, s7];     buf7[:n, s7]     = delayed; sum7[:n]     += delayed
    ord_sum7[:n] -= ord_buf7[:n, s7]; ord_buf7[:n, s7] = matched; ord_sum7[:n] += matched

# ── Initialise ────────────────────────────────────────────────────────────────
t0 = time.time()
print("AB Validation Simulation")
print(f"  SEED={SEED}  lambda_mem={LAMBDA_MEM}  sigma_demand={SIGMA_DEMAND}")
print(f"  P_treat={P_TREAT}  dispatch_advance={DISPATCH_ADVANCE}min")
print(f"  N_users={N_USERS:,}  K_sweep={K_SWEEP}")

params = DGPParams(n_users_initial=N_USERS, seed=SEED,
                   lambda_mem=LAMBDA_MEM,
                   courier_scale_base=COURIER_SCALE_BASE,
                   sigma_demand=SIGMA_DEMAND)
rng = np.random.default_rng(SEED)
latent, state = generate_users_warm(params, rng)
n = len(latent["beta_y"])
print(f"  Warm-start users: {n:,}")

_buf30   = np.zeros((MAX_N, 30), dtype=np.int32); _sum30   = np.zeros(MAX_N, dtype=np.int32)
_buf7    = np.zeros((MAX_N,  7), dtype=np.int32); _ord_buf7 = np.zeros((MAX_N, 7), dtype=np.int32)
_sum7    = np.zeros(MAX_N, dtype=np.int32);       _ord_sum7 = np.zeros(MAX_N, dtype=np.int32)
e_match  = 10.0   # initial estimate; converges to true market avg during warmup
spend_mat = np.zeros((MAX_N, N_OBS_DAYS), dtype=np.float32)

# ── Obs-phase diagnostic accumulators (Figures 1, 2, 3, 6) ────────────────────
# Collected during the warmup + observational phases of this same run (no extra
# simulation). Plotted by plot_figure.py; they do not feed the CBPS/Chronos path.
trans_ontime   = np.zeros((N_STATES, N_STATES), dtype=np.int64)   # ordered, on-time
trans_delay    = np.zeros((N_STATES, N_STATES), dtype=np.int64)   # ordered, delayed
trans_organic  = np.zeros((N_STATES, N_STATES), dtype=np.int64)   # did not order
ss_matched_cum = np.zeros(MAX_N, dtype=np.int64)                  # per-user, steady-state
ss_delayed_cum = np.zeros(MAX_N, dtype=np.int64)
dow_hr_orders   = np.zeros((7, 24), dtype=np.float64)
dow_hr_couriers = np.zeros((7, 24), dtype=np.float64)
dow_hr_matched  = np.zeros((7, 24), dtype=np.float64)
dow_hr_match_t  = np.zeros((7, 24), dtype=np.float64)
dow_hr_count    = np.zeros(7, dtype=np.int64)
lc_share_daily  = []                                             # (warmup + obs, N_STATES)

# ── Phase 1: Warmup ───────────────────────────────────────────────────────────
print(f"\nPhase 1: Warmup ({N_WARMUP} days)...")
for day in range(N_WARMUP):
    n_active = (state["lifecycle"] != CHURNED).sum()
    n_new = rng.poisson(params.lambda_growth * n_active)
    if n_new > 0:
        nl, ns = generate_users(params, rng, n=n_new)
        latent = _append(latent, nl); state = _append(state, ns)
    n = len(latent["beta_y"]); assert n <= MAX_N
    n_ord, _, orders = draw_daily_orders(latent, state, params, rng, day,
                                         expected_matching_time=e_match)
    dlyd, rspend, mkt = run_marketplace(orders, n, params, rng, day)
    if mkt["n_matched"] > 0:
        e_match = (1 - E_MATCH_ALPHA) * e_match + E_MATCH_ALPHA * mkt["avg_match_time"]
    update_daily_state(state, latent, n_ord, rspend, dlyd, day, params)
    draw_lifecycle_transitions(state, latent, params, rng)
    _update_delay_buffers(n, mkt, day, _buf30, _sum30, _buf7, _ord_buf7, _sum7, _ord_sum7)
    lc_share_daily.append(_lc_share(state, len(latent["beta_y"])))
    if day % 25 == 0 or day == N_WARMUP - 1:
        print(f"  day {day:3d}: n={n:,}  delay={mkt['delay_rate']*100:.1f}%  e_match={e_match:.1f}min")

# ── Phase 2: Observational ────────────────────────────────────────────────────
print(f"\nPhase 2: Observational ({N_OBS_DAYS} days)...")
obs_records = []
for od in range(N_OBS_DAYS):
    sim_day = OBS_START + od
    n_active = (state["lifecycle"] != CHURNED).sum()
    n_new = rng.poisson(params.lambda_growth * n_active)
    if n_new > 0:
        nl, ns = generate_users(params, rng, n=n_new)
        latent = _append(latent, nl); state = _append(state, ns)
    n = len(latent["beta_y"]); assert n <= MAX_N

    # Rolling delay rates used as confounding proxies in the CBPS feature matrix
    dr30 = _sum30[:n] / np.maximum(state["F30"][:n], 1.)
    dr7  = _sum7[:n]  / np.maximum(_ord_sum7[:n], 1.)

    n_ord, _, orders = draw_daily_orders(latent, state, params, rng, sim_day,
                                         expected_matching_time=e_match)
    dlyd, rspend, mkt = run_marketplace(orders, n, params, rng, sim_day)
    if mkt["n_matched"] > 0:
        e_match = (1 - E_MATCH_ALPHA) * e_match + E_MATCH_ALPHA * mkt["avg_match_time"]
    spend_mat[:n, od] = rspend[:n].astype(np.float32)

    if mkt["n_matched"] > 0:
        mm = mkt["matched_per_order"].astype(bool)
        ou = mkt["order_user_idx"][mm]
        obs_records.append(pd.DataFrame({
            "user_id":        ou.astype(np.int32),
            "obs_day":        np.full(mm.sum(), od, dtype=np.int32),
            "delayed":        mkt["delayed_per_order"][mm].astype(np.int32),
            "distance":       mkt["order_distance"][mm].astype(np.float32),
            "prep_time":      mkt["order_prep_time"][mm].astype(np.float32),
            "tip":            mkt["order_tip"][mm].astype(np.float32),
            "spend":          mkt["order_spend"][mm].astype(np.float32),
            "hour":           mkt["order_hour"][mm].astype(np.int32),
            "F30":            state["F30"][ou].astype(np.float32),
            "S30":            state["S30"][ou].astype(np.float32),
            "R":              state["R"][ou].astype(np.float32),
            "n_orders":       state["n_orders"][ou].astype(np.float32),
            "delay_rate_30d": dr30[ou].astype(np.float32),
            "delay_rate_7d":  dr7[ou].astype(np.float32),
            "n_orders_today": mkt["matched_per_user"][ou].astype(np.float32),
        }))

    lc_pre = state["lifecycle"][:n].copy()
    update_daily_state(state, latent, n_ord, rspend, dlyd, sim_day, params)
    draw_lifecycle_transitions(state, latent, params, rng)
    _update_delay_buffers(n, mkt, sim_day, _buf30, _sum30, _buf7, _ord_buf7, _sum7, _ord_sum7)

    # Diagnostics: lifecycle transitions (all obs days), stratified by day-t outcome
    lc_post   = state["lifecycle"][:n]
    ordered_u = mkt["matched_per_user"][:n] > 0
    delayed_u = mkt["delayed_per_user"][:n] > 0
    np.add.at(trans_ontime,  (lc_pre[ordered_u & ~delayed_u], lc_post[ordered_u & ~delayed_u]), 1)
    np.add.at(trans_delay,   (lc_pre[ordered_u &  delayed_u], lc_post[ordered_u &  delayed_u]), 1)
    np.add.at(trans_organic, (lc_pre[~ordered_u],             lc_post[~ordered_u]),             1)
    lc_share_daily.append(_lc_share(state, len(latent["beta_y"])))

    # Diagnostics: steady-state per-user delay/value + time-of-day / day-of-week
    if od >= STEADY_START:
        ss_matched_cum[:n] += mkt["matched_per_user"][:n].astype(np.int64)
        ss_delayed_cum[:n] += mkt["delayed_per_user"][:n].astype(np.int64)
        dow = sim_day % 7
        ord_slots = mkt["orders_in_by_slot"];  cou_slots = mkt["couriers_by_slot"]
        mat_slots = mkt["matched_by_slot"];     mt_slots  = mkt["match_time_sum_by_slot"]
        for h in range(24):
            sl = slice(h * SLOTS_PER_HR, (h + 1) * SLOTS_PER_HR)
            dow_hr_orders[dow, h]   += ord_slots[sl].sum()
            dow_hr_couriers[dow, h] += cou_slots[sl].sum()
            dow_hr_matched[dow, h]  += mat_slots[sl].sum()
            dow_hr_match_t[dow, h]  += mt_slots[sl].sum()
        dow_hr_count[dow] += 1

    if od % 30 == 0 or od == N_OBS_DAYS - 1:
        print(f"  obs {od:3d}: n={n:,}  delay={mkt['delay_rate']*100:.1f}%  e_match={e_match:.1f}min")

# ── Obs-phase diagnostic snapshot (end of obs = sim day 250, pre-AB) ───────────
n_obs       = len(latent["beta_y"])
lc_final    = state["lifecycle"][:n_obs].astype(np.int8)
spend_final = state["spend_total"][:n_obs].astype(np.float32)
matched_cum = ss_matched_cum[:n_obs].copy()
delayed_cum = ss_delayed_cum[:n_obs].copy()
lc_share_arr = np.array(lc_share_daily, dtype=np.float32)
print(f"  Obs snapshot: n_obs={n_obs:,}  "
      f"state counts={ {s: int((lc_final == s).sum()) for s in range(N_STATES)} }")

# ── Phase 3: AB experiment ────────────────────────────────────────────────────
print(f"\nPhase 3: AB experiment ({N_AB_DAYS} days, P_treat={P_TREAT})...")
n0 = len(latent["beta_y"])
latent["treatment"] = rng.binomial(1, P_TREAT, n0).astype(np.int32)

wk_spend_all       = np.zeros((N_WEEKS, n0), dtype=np.float32)
_wk_spend          = np.zeros(n0, dtype=np.float64)
first_ab_order_day = np.full(n0, -1, dtype=np.int32)
weekly_del_t       = np.zeros(N_WEEKS, dtype=np.int64)
weekly_mat_t       = np.zeros(N_WEEKS, dtype=np.int64)
weekly_del_c       = np.zeros(N_WEEKS, dtype=np.int64)
weekly_mat_c       = np.zeros(N_WEEKS, dtype=np.int64)

for ad in range(N_AB_DAYS):
    sim_day = AB_START + ad
    n_active = (state["lifecycle"] != CHURNED).sum()
    n_new = rng.poisson(params.lambda_growth * n_active)
    if n_new > 0:
        nl, ns = generate_users(params, rng, n=n_new)
        nl["treatment"] = rng.binomial(1, P_TREAT, n_new).astype(np.int32)
        latent = _append(latent, nl); state = _append(state, ns)
    n = len(latent["beta_y"]); assert n <= MAX_N

    # Predictive dispatch: treated users dispatched DISPATCH_ADVANCE min early
    d_prep = (latent["treatment"][:n] * DISPATCH_ADVANCE).astype(np.float64)
    n_ord, _, orders = draw_daily_orders(latent, state, params, rng, sim_day,
                                         expected_matching_time=e_match,
                                         delta_prep_per_user=d_prep)
    dlyd, rspend, mkt = run_marketplace(orders, n, params, rng, sim_day)
    if mkt["n_matched"] > 0:
        e_match = (1 - E_MATCH_ALPHA) * e_match + E_MATCH_ALPHA * mkt["avg_match_time"]

    matched_n0 = mkt["matched_per_user"][:n0]
    newly_ordered = (first_ab_order_day == -1) & (matched_n0 > 0)
    first_ab_order_day[newly_ordered] = ad

    # Weekly delay-rate breakdown by arm (full randomized population, ITT)
    w_ad = ad // 7
    if w_ad < N_WEEKS:
        treat_flag = (latent["treatment"][:n] == 1)
        del_day    = mkt["delayed_per_user"][:n].astype(np.int64)
        mat_day    = mkt["matched_per_user"][:n].astype(np.int64)
        weekly_del_t[w_ad] += int(del_day[ treat_flag].sum())
        weekly_mat_t[w_ad] += int(mat_day[ treat_flag].sum())
        weekly_del_c[w_ad] += int(del_day[~treat_flag].sum())
        weekly_mat_c[w_ad] += int(mat_day[~treat_flag].sum())

    # Accumulate weekly spending (per-user totals, for ANCOVA)
    _wk_spend += rspend[:n0].astype(np.float64)
    if (ad % 7 == 6) or (ad == N_AB_DAYS - 1):
        w = ad // 7
        if w < N_WEEKS:
            wk_spend_all[w] = _wk_spend.astype(np.float32)
        _wk_spend[:] = 0.

    update_daily_state(state, latent, n_ord, rspend, dlyd, sim_day, params)
    draw_lifecycle_transitions(state, latent, params, rng)
    _update_delay_buffers(n, mkt, sim_day, _buf30, _sum30, _buf7, _ord_buf7, _sum7, _ord_sum7)
    if ad % 20 == 0 or ad == N_AB_DAYS - 1:
        print(f"  ab  {ad:3d}: n={n:,}  delay={mkt['delay_rate']*100:.1f}%  e_match={e_match:.1f}min")

print(f"\n  Simulation done in {(time.time()-t0)/60:.1f} min  (n0={n0:,})")

# ── Active cohort and AB ground truth ─────────────────────────────────────────
active_mask       = (first_ab_order_day >= 0) & (first_ab_order_day < K_ENTRY)
active_treat_mask = active_mask & (latent["treatment"][:n0] == 1)
active_ctrl_mask  = active_mask & (latent["treatment"][:n0] == 0)
n_act   = int(active_mask.sum())
n_act_t = int(active_treat_mask.sum())
n_act_c = int(active_ctrl_mask.sum())

# Last-4-week delay-rate difference — the Table-1 ΔLTV multiplier
_lt = slice(N_WEEKS - LAST_K_DAYS // 7, N_WEEKS)
_nt = int(weekly_mat_t[_lt].sum()); _nc = int(weekly_mat_c[_lt].sum())
delta_dr_AB = (weekly_del_t[_lt].sum() / max(_nt, 1)
                   - weekly_del_c[_lt].sum() / max(_nc, 1)) * 100.

print(f"  AB cohort: {n_act:,}/{n0:,} ({n_act/n0*100:.0f}%)  "
      f"treat={n_act_t:,}  ctrl={n_act_c:,}")
print(f"  Δdr_AB={delta_dr_AB:+.4f}pp")

# ── Cumulative spending matrix and pre-AB covariate ───────────────────────────
cum_spend = np.zeros((n0, N_OBS_DAYS + 1), dtype=np.float32)
cum_spend[:, 1:] = np.cumsum(spend_mat[:n0].astype(np.float64), axis=1).astype(np.float32)
del spend_mat
pre_ab_spend = (cum_spend[:n0, N_OBS_DAYS] - cum_spend[:n0, N_OBS_DAYS - N_PRE_AB]).astype(np.float64)

obs = pd.concat(obs_records, ignore_index=True)
del obs_records

# ── CBPS point estimates for each K ──────────────────────────────────────────
uid_all   = obs["user_id"].values.astype(np.int64)
td_all    = obs["obs_day"].values.astype(np.int64)
T_all     = obs["delayed"].values.astype(np.int32)
spend_all = obs["spend"].values.astype(np.float64)
dow_all   = (OBS_START + td_all).astype(float) % 7
hod_all   = obs["hour"].values.astype(float)
Z_all     = np.column_stack([
    np.ones(len(obs)),
    obs["delay_rate_30d"].values.astype(float),
    obs["delay_rate_7d"].values.astype(float),
    obs["F30"].values.astype(float),
    obs["S30"].values.astype(float),
    obs["R"].values.astype(float),
    np.log1p(obs["n_orders"].values.astype(float)),
    obs["n_orders_today"].values.astype(float),
    obs["distance"].values.astype(float),
    obs["prep_time"].values.astype(float),
    obs["tip"].values.astype(float),
    np.sin(2 * np.pi * dow_all / 7),  np.cos(2 * np.pi * dow_all / 7),
    np.sin(2 * np.pi * hod_all / 24), np.cos(2 * np.pi * hod_all / 24),
])

beta_pt    = {}        # K -> full-sample CBPS β  (feeds the Figure-4 Chronos lines)
table_pt   = {}        # K -> (naive, mlp-ipw, cbps) full-sample point estimates (Table 1)
table_boot = {}        # K -> (B_BOOT, 3) bootstrap reps (Table-1 CIs); empty if RUN_BOOTSTRAP=False
print(f"\nFull-sample estimation  (K_SWEEP={K_SWEEP}, Table-1 K={K_TABLE}, "
      f"bootstrap={'ON' if RUN_BOOTSTRAP else 'OFF'}):")
for K in K_SWEEP:
    est_obs_max = N_OBS_DAYS - K - 1   # cap so the K-day forward window stays in the obs phase
    mask     = td_all <= est_obs_max
    uid_k    = uid_all[mask];   td_k    = td_all[mask]
    T_k      = T_all[mask];     spend_k = spend_all[mask]; Z_k = Z_all[mask]
    t_lo     = np.minimum(td_k + 1,     N_OBS_DAYS)
    t_hi     = np.minimum(td_k + 1 + K, N_OBS_DAYS)
    Y_k      = (cum_spend[uid_k, t_hi] - cum_spend[uid_k, t_lo]).astype(np.float64)
    t_est    = time.time()
    if K in K_TABLE:
        b_naive, b_ipw, b_cbps = _estimate(Z_k, T_k, Y_k, spend_k, SEED)
        table_pt[K] = (b_naive, b_ipw, b_cbps)
        beta_pt[K]  = b_cbps
        print(f"  K={K:2d}d  rows={len(Z_k):,}  naive={b_naive*100:+.4f}  "
              f"mlp-ipw={b_ipw*100:+.4f}  cbps={b_cbps*100:+.4f}  (%/pp)  [{time.time()-t_est:.0f}s]")
    else:
        b_cbps = _fit_cbps(Z_k, T_k, Y_k, spend_k)
        beta_pt[K] = b_cbps
        print(f"  K={K:2d}d  rows={len(Z_k):,}  cbps={b_cbps*100:+.4f}  (%/pp)  [{time.time()-t_est:.0f}s]")

    # Bootstrap CIs for the Table-1 horizons (the ~9h cost; off by default)
    if RUN_BOOTSTRAP and K in K_TABLE:
        users_k = np.unique(uid_k)
        u2r = {int(u): [] for u in users_k}
        for i, u in enumerate(uid_k):
            u2r[int(u)].append(i)
        for u in u2r:
            u2r[u] = np.array(u2r[u], dtype=np.int64)
        _BT.update(Z=Z_k, T=T_k, Y=Y_k, spend=spend_k, users=users_k, u2r=u2r)
        seeds_b = [SEED + 77777 + b for b in range(B_BOOT)]
        t_bt = time.time()
        with mp.Pool(N_WORKERS) as pool:
            table_boot[K] = np.array(list(pool.imap_unordered(_boot_one, seeds_b)))
        print(f"        bootstrap B={B_BOOT} ({N_WORKERS}w): {(time.time()-t_bt)/60:.1f} min")

del cum_spend

# ── Prepare active-cohort arrays for cache ────────────────────────────────────
# ANCOVA (OLS with pre-AB spending covariate) is run in plot_figure.py on these arrays.
D_act        = latent["treatment"][:n0][active_mask].astype(float)
pre_act      = pre_ab_spend[active_mask]
wk_spend_act = wk_spend_all[:, active_mask]

# ── Ground-truth ΔLTV%: ANCOVA over the last LAST_K_DAYS (W14-W17), active cohort ─
_X        = np.column_stack([np.ones(len(D_act)), D_act, pre_act])
_XtXinv   = np.linalg.inv(_X.T @ _X)
_y_lt     = wk_spend_act[N_WEEKS - LAST_K_DAYS // 7:].sum(axis=0).astype(np.float64)
_beta, *_ = np.linalg.lstsq(_X, _y_lt, rcond=None)
_resid    = _y_lt - _X @ _beta
_s2       = float(_resid.var(ddof=_X.shape[1]))
_cm       = _y_lt[D_act == 0].mean()
truth_dgb = float(_beta[1]) / _cm * 100.
truth_se  = float(np.sqrt(_s2 * _XtXinv[1, 1])) / _cm * 100.
truth_lo  = truth_dgb - 1.96 * truth_se
truth_hi  = truth_dgb + 1.96 * truth_se

# ── Table 1: main result table ───────────────────────────────────────────────────
print("\n" + "=" * 70)
print(f"TABLE 1   truth (ANCOVA W14-W17) ΔLTV% = {truth_dgb:+.3f}%  "
      f"95% CI [{truth_lo:+.3f}, {truth_hi:+.3f}]")
print(f"          Δdr_AB (full n) = {delta_dr_AB:+.4f} pp"
      f"   (ΔLTV% = β × Δdr_last4w)")
print("=" * 70)
_NAMES = ["Naive", "MLP-IPW", "CBPS-IPW"]
_hdr = "  K   estimator     β mean     β 95% CI                 ΔLTV%     ΔLTV 95% CI            recovery"
print(_hdr); print("-" * len(_hdr))
for K in K_TABLE:
    pt = table_pt.get(K)
    if pt is None:
        continue
    boot = table_boot.get(K)
    for i, nm in enumerate(_NAMES):
        if boot is not None and len(boot):
            bm = boot[:, i].mean()
            blo, bhi = np.percentile(boot[:, i], [2.5, 97.5])
            pred = bm * delta_dr_AB * 100.
            plo, phi = np.percentile(boot[:, i] * delta_dr_AB * 100., [2.5, 97.5])
            print(f"  {K:<3} {nm:<10} {bm*100:+8.4f}  [{blo*100:+8.4f},{bhi*100:+8.4f}]  "
                  f"{pred:+7.3f}%  [{plo:+7.3f},{phi:+7.3f}]  {pred/truth_dgb:+.3f}×")
        else:
            b    = pt[i]
            pred = b * delta_dr_AB * 100.
            print(f"  {K:<3} {nm:<10} {b*100:+8.4f}  {'(point est.; bootstrap OFF)':<23}  "
                  f"{pred:+7.3f}%  {'':<22} {pred/truth_dgb:+.3f}×")
print("=" * 70)

# ── Save cache ────────────────────────────────────────────────────────────────
CACHE_DIR.mkdir(parents=True, exist_ok=True)
save_kw = dict(
    K_SWEEP      = np.array(K_SWEEP, dtype=np.int32),
    betas        = np.array([beta_pt[K] for K in K_SWEEP], dtype=np.float64),
    weekly_del_t = weekly_del_t, weekly_mat_t = weekly_mat_t,
    weekly_del_c = weekly_del_c, weekly_mat_c = weekly_mat_c,
    wk_gb_act    = wk_spend_act.astype(np.float32),   # key kept for plot_figure.py compat
    D_act        = D_act.astype(np.int8),
    pre_act      = pre_act.astype(np.float32),
    delta_dr_AB  = np.float64(delta_dr_AB),
    truth_dgb        = np.float64(truth_dgb),
    truth_se         = np.float64(truth_se),
    n0               = np.int64(n0),
    n_act            = np.int64(n_act),
    n_act_t          = np.int64(n_act_t),
    n_act_c          = np.int64(n_act_c),
    N_AB_DAYS        = np.int32(N_AB_DAYS),
    N_WEEKS          = np.int32(N_WEEKS),
    LAMBDA_MEM       = np.float64(LAMBDA_MEM),
    SIGMA_DEMAND     = np.float64(SIGMA_DEMAND),
    P_TREAT          = np.float64(P_TREAT),
    K_ENTRY          = np.int32(K_ENTRY),
    N_USERS          = np.int32(N_USERS),
    COURIER_SCALE_BASE = np.int32(COURIER_SCALE_BASE),
    K_TABLE          = np.array(K_TABLE, dtype=np.int32),
    B_BOOT           = np.int32(B_BOOT),
    RUN_BOOTSTRAP    = np.bool_(RUN_BOOTSTRAP),
    SEED             = np.int32(SEED),
    # ── Obs-phase diagnostics (Figures 1, 2, 3, 6) ──
    trans_ontime  = trans_ontime, trans_delay = trans_delay, trans_organic = trans_organic,
    lc_final      = lc_final,     spend_final  = spend_final,
    matched_cum   = matched_cum,  delayed_cum  = delayed_cum,
    dow_hr_orders = dow_hr_orders, dow_hr_couriers = dow_hr_couriers,
    dow_hr_matched = dow_hr_matched, dow_hr_match_t = dow_hr_match_t,
    dow_hr_count  = dow_hr_count,
    lc_share_daily = lc_share_arr,
    N_WARMUP     = np.int32(N_WARMUP),
    N_OBS_DAYS   = np.int32(N_OBS_DAYS),
    STEADY_START = np.int32(STEADY_START),
    SLOTS_PER_HR = np.int32(SLOTS_PER_HR),
)
# ── Table-1 point estimates (and bootstrap reps, when RUN_BOOTSTRAP) ──
for K in K_TABLE:
    if K in table_pt:
        save_kw[f"K{K}_pt"] = np.array(table_pt[K], dtype=np.float64)
    if K in table_boot:
        save_kw[f"K{K}_boot"] = table_boot[K].astype(np.float64)
np.savez_compressed(str(CACHE_PATH), **save_kw)
print(f"\nCache saved → {CACHE_PATH}")
print(f"Total wall time: {(time.time()-t0)/60:.1f} min")
