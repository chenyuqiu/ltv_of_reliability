"""
Switchback Validation Simulation — Figure 5
=============================================
Figure 5: Results across 20 eater-resampled replications of a simulated
marketplace under an increased-supply policy change.

Runs the full simulation:
  1. Warmup + observational phase (shared, single seed)
  2. CBPS estimation on observational data
  3. 20 parallel fork runs: each fork runs a 28-day switchback experiment
     plus two 120-day long-term arms (CN / CE) for ground-truth LTV impact

The market (supply/demand shocks, courier RNG) is held identical across
all 20 forks. Only the eater-side (user) RNG varies — new-eater arrivals and
per-eater order draws — isolating pure eater-resampling noise.

Runtime: ~30–60 min wall time (20 forks × 3 simulations each, run in parallel).

Usage (from repository root):
    python switchback_validation/run_simulation.py

Output:
    switchback_validation/results/sw_validation_results.json
"""
from __future__ import annotations
import copy, json, sys, time, warnings, multiprocessing
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from sim.params import DGPParams
from sim.users import (generate_users, generate_users_warm,
                       update_daily_state, draw_lifecycle_transitions, CHURNED)
from sim.demand import draw_daily_orders
from sim.marketplace import run_marketplace

# ── Simulation parameters ─────────────────────────────────────────────────────
OUTER_SEED    = 42
LAMBDA_MEM    = 0.06    # delay-memory decay; half-life ≈ ln2/0.06 ≈ 11.6 days
SIGMA_DEMAND  = 0.1     # daily macro shock std
COURIER_SCALE = 700
SUPPLY_MULT   = 1.25    # treatment: 1.25× courier rate in treated blocks
N_USERS       = 100_000
BLOCK_HOURS   = 3       # 8 three-hour blocks per day
N_FORK_RUNS   = 20
T_WARMUP      = 100
T_OBS         = 150
T_SB          = 28      # switchback duration (days)
T_LT          = 120     # long-term follow-up duration (days)
LT_EVAL_DAYS  = 21      # last N days used for LT spending ground truth
K_FORWARD     = 56      # observational forward-spending horizon
N_EST_USERS   = 0       # CBPS user subsample (0 = full sample)
DELTA_PI      = 0.01
MAX_N         = 200_000
SYNC_SEED     = 77777   # fixed courier RNG seed (synced across forks)
E_MATCH_ALPHA = 0.05    # EWMA weight for avg_match_time (half-life ≈ 14 days)
SIM_OBS_START = T_WARMUP
EST_OBS_MAX   = T_OBS - K_FORWARD - 1
N_BLOCKS      = 24 // BLOCK_HOURS
BC_SCALE      = (BLOCK_HOURS - 1) / BLOCK_HOURS
BC_B_OVER_L   = 1 / (BLOCK_HOURS - 1)
MIN_MATCHED   = 5
N_BOOTSTRAP   = 25

RESULTS_DIR  = Path(__file__).parent / "results"
RESULTS_PATH = RESULTS_DIR / "sw_validation_results.json"

# ── Block assignment (fixed seed, reproduced identically every run) ───────────
# Stratified design: each day-of-week gets a balanced rotation of a base pattern
# (exact treated/control balance per DoW), with rows/columns permuted by a fixed
# RNG. Base pattern below is for the 8-block (BLOCK_HOURS=3) day used in the paper.
_W_BASE_8 = np.array([[1, 1, 0, 0, 1, 1, 0, 0], [0, 0, 1, 1, 0, 0, 1, 1],
                      [1, 0, 1, 0, 0, 1, 0, 1], [0, 1, 0, 1, 1, 0, 1, 0]], dtype=int)
_rng_design = np.random.default_rng(9999)
W = np.zeros((T_SB, N_BLOCKS), dtype=int)
for _dow in range(7):
    _days = [d for d in range(T_SB) if d % 7 == _dow]
    _mat  = _W_BASE_8[_rng_design.permutation(4), :][:, _rng_design.permutation(N_BLOCKS)]
    for _i, _d in enumerate(_days):
        W[_d] = _mat[_i]

# ── Shared state (populated in main, read by fork workers via fork()) ─────────
_SHARED: dict = {}

# ── Helpers ───────────────────────────────────────────────────────────────────
def _append(base, new):
    """Concatenate two latent/state dicts along axis 0."""
    return {k: np.concatenate([base[k], new[k]], 0) for k in base}

def _update_delay_buffers(n, mkt, sim_day, buf30, sum30, buf7, ord_buf7, sum7, ord_sum7):
    """
    Update the rolling 30-day and 7-day delay-count buffers used to compute
    delay_rate_30d and delay_rate_7d features for CBPS propensity estimation.

    buf30 / sum30       : circular 30-slot buffer of daily delayed counts
    buf7  / sum7        : circular  7-slot buffer of daily delayed counts
    ord_buf7 / ord_sum7 : circular 7-slot buffer of daily matched counts (denominator)
    """
    delayed = mkt["delayed_per_user"][:n].astype(np.int32)
    matched = mkt["matched_per_user"][:n].astype(np.int32)
    s30 = sim_day % 30
    sum30[:n] -= buf30[:n, s30]; buf30[:n, s30] = delayed; sum30[:n] += delayed
    s7  = sim_day % 7
    sum7[:n]     -= buf7[:n, s7];     buf7[:n, s7]     = delayed; sum7[:n]     += delayed
    ord_sum7[:n] -= ord_buf7[:n, s7]; ord_buf7[:n, s7] = matched; ord_sum7[:n] += matched

def _collect_sb(mkt):
    oh      = mkt["order_hour"].astype(int)
    og      = mkt["order_spend"]
    mm      = mkt["matched_per_order"].astype(bool)
    dl      = mkt["delayed_per_order"].astype(bool)
    blk     = oh // BLOCK_HOURS
    hib     = oh % BLOCK_HOURS
    excl    = hib >= 1
    delay_r = np.full(N_BLOCKS, np.nan)
    burnin  = np.full(N_BLOCKS, np.nan)
    spend_r = np.full(N_BLOCKS, np.nan)
    np_r    = np.full(N_BLOCKS, np.nan)
    for b in range(N_BLOCKS):
        mask_e = (blk == b) & excl & mm
        n_mat  = mask_e.sum()
        if n_mat >= MIN_MATCHED:
            delay_r[b] = (mask_e & dl).sum() / n_mat
        placed_e = (blk == b) & excl
        if placed_e.any():
            spend_r[b] = og[mm & placed_e].sum()
            np_r[b]    = float(placed_e.sum())
        mask_bi = (blk == b) & ~excl & mm
        n_bi = mask_bi.sum()
        if n_bi >= MIN_MATCHED:
            burnin[b] = (mask_bi & dl).sum() / n_bi
    return delay_r, burnin, spend_r, np_r

def _prev_arm(d, blk):
    pd_, pb = (d, blk-1) if blk > 0 else (d-1, N_BLOCKS-1)
    return None if pd_ < 0 else int(W[pd_, pb])

def _compute_all(sb_dr, sb_bi, sb_spend, sb_np, drop_day=None):
    keep = np.ones((T_SB, N_BLOCKS), dtype=bool)
    if drop_day is not None:
        keep[drop_day, :] = False
    valid_dr = ~np.isnan(sb_dr)
    trt_dr   = sb_dr[(W==1) & valid_dr & keep]
    ctl_dr   = sb_dr[(W==0) & valid_dr & keep]
    tau_dm   = float(trt_dr.mean() - ctl_dr.mean()) if len(trt_dr) and len(ctl_dr) else np.nan
    bi_11, bi_00 = [], []
    for d in range(T_SB):
        if drop_day is not None and d == drop_day:
            continue
        for blk in range(N_BLOCKS):
            prev = _prev_arm(d, blk)
            if prev is None: continue
            curr = int(W[d, blk])
            if prev != curr: continue
            bi = sb_bi[d, blk]
            if np.isnan(bi): continue
            (bi_11 if curr == 1 else bi_00).append(bi)
    tau_bc = (BC_SCALE * tau_dm + BC_B_OVER_L * (np.mean(bi_11) - np.mean(bi_00))
              if bi_11 and bi_00 and not np.isnan(tau_dm) else np.nan)
    valid_spend = ~np.isnan(sb_spend) & ~np.isnan(sb_np) & (sb_np > 0)
    spend_rate  = np.where(valid_spend, sb_spend / sb_np, np.nan)
    trt_rate    = spend_rate[(W==1) & valid_spend & keep]
    ctl_rate    = spend_rate[(W==0) & valid_spend & keep]
    if len(trt_rate) and len(ctl_rate) and ctl_rate.mean() > 0:
        delta_spend_pct = float((trt_rate.mean() - ctl_rate.mean()) / ctl_rate.mean() * 100)
    else:
        delta_spend_pct = np.nan
    return tau_dm, tau_bc, delta_spend_pct

def _jk_se_all(sb_dr, sb_bi, sb_spend, sb_np):
    tau_dm, tau_bc, delta_spend_pct = _compute_all(sb_dr, sb_bi, sb_spend, sb_np)
    loo_dm    = np.full(T_SB, np.nan)
    loo_bc    = np.full(T_SB, np.nan)
    loo_spend = np.full(T_SB, np.nan)
    for d in range(T_SB):
        dm, bc, spend = _compute_all(sb_dr, sb_bi, sb_spend, sb_np, drop_day=d)
        loo_dm[d] = dm; loo_bc[d] = bc; loo_spend[d] = spend
    def _se(loo):
        vl = loo[~np.isnan(loo)]
        return float(np.sqrt(((len(vl)-1)/len(vl)) * np.sum((vl-vl.mean())**2))) if len(vl) >= 2 else np.nan
    return tau_dm, tau_bc, delta_spend_pct, _se(loo_dm), _se(loo_bc), _se(loo_spend)

def _cbps_obj(Z, W_arm, dpi):
    n = len(W_arm)
    def _f(theta):
        z = Z @ theta; enZ = np.exp(np.clip(-z, -50., 50.))
        return (np.mean(dpi * (W_arm * enZ + (1.-W_arm) * z)),
                (Z.T @ (dpi * ((1.-W_arm) - W_arm * enZ))) / n)
    return _f

def _cbps_weights(Z, W_arm, dpi):
    res = minimize(_cbps_obj(Z, W_arm, dpi), np.zeros(Z.shape[1]),
                   method="BFGS", jac=True, options={"maxiter": 500, "gtol": 1e-6})
    return 1. + np.exp(np.clip(-Z @ res.x, -50., 50.))

def _policy_gradient(T, Y, spend, w0, w1):
    return float(((T==1)*w1*Y - (T==0)*w0*Y).sum() * DELTA_PI / spend.sum())

def _build_ZTY(obs_df, cum_spend):
    uid  = obs_df["user_id"].values.astype(np.int64)
    td   = obs_df["obs_day"].values.astype(np.int64)
    t_lo = np.minimum(td+1, T_OBS)
    t_hi = np.minimum(td+1+K_FORWARD, T_OBS)
    Y    = (cum_spend[uid, t_hi] - cum_spend[uid, t_lo]).astype(np.float64)
    n_obs = len(obs_df)
    T_arm = obs_df["delayed"].values.astype(np.int32)
    spend = obs_df["spend"].values.astype(np.float64)
    dow   = (SIM_OBS_START + obs_df["obs_day"].values).astype(float) % 7
    hod   = obs_df["hour"].values.astype(float)
    Z_raw = np.column_stack([
        np.ones(n_obs),
        obs_df["delay_rate_30d"].values.astype(float),
        obs_df["delay_rate_7d"].values.astype(float),
        obs_df["F30"].values.astype(float),
        obs_df["S30"].values.astype(float),
        obs_df["R"].values.astype(float),
        np.log1p(obs_df["n_orders"].values.astype(float)),
        obs_df["n_orders_today"].values.astype(float),
        obs_df["distance"].values.astype(float),
        obs_df["prep_time"].values.astype(float),
        obs_df["tip"].values.astype(float),
        np.sin(2*np.pi*dow/7),  np.cos(2*np.pi*dow/7),
        np.sin(2*np.pi*hod/24), np.cos(2*np.pi*hod/24),
    ])
    Z_mu = Z_raw.mean(0); Z_mu[0] = 0.
    Z_sd = Z_raw.std(0);  Z_sd[0] = 1.; Z_sd[Z_sd==0] = 1.
    return (Z_raw - Z_mu) / Z_sd, T_arm, Y, spend

def _run_cbps_on(obs_df, cum_spend):
    Z, T_arm, Y, spend = _build_ZTY(obs_df, cum_spend)
    dpi = np.full(len(obs_df), DELTA_PI)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        w0 = _cbps_weights(Z, (T_arm==0).astype(float), dpi)
        w1 = _cbps_weights(Z, (T_arm==1).astype(float), dpi)
    return _policy_gradient(T_arm, Y, spend, w0, w1)

# ── Fork worker ───────────────────────────────────────────────────────────────
def _run_fork(fork_id):
    latent_0  = _SHARED["latent_0"]
    state_0   = _SHARED["state_0"]
    e_match_0 = _SHARED["e_match_0"]
    params    = _SHARED["params"]
    supply_s  = _SHARED["supply_fixed"]
    demand_s  = _SHARED["demand_fixed"]
    t0 = time.time()

    # ── Switchback fork ────────────────────────────────────────────────────────
    rng_sb   = np.random.default_rng(fork_id * 100 + 10)
    latent   = copy.deepcopy(latent_0); state = copy.deepcopy(state_0); e_match = e_match_0
    sb_dr    = np.full((T_SB, N_BLOCKS), np.nan)
    sb_bi    = np.full((T_SB, N_BLOCKS), np.nan)
    sb_spend = np.full((T_SB, N_BLOCKS), np.nan)
    sb_np    = np.full((T_SB, N_BLOCKS), np.nan)

    for d in range(T_SB):
        hm = np.array([SUPPLY_MULT if W[d, h // BLOCK_HOURS] == 1 else 1.0 for h in range(24)])
        n_active = (state["lifecycle"] != CHURNED).sum()
        n_new = rng_sb.poisson(params.lambda_growth * n_active)
        if n_new > 0:
            nl, ns = generate_users(params, rng_sb, n=n_new)
            latent = _append(latent, nl); state = _append(state, ns)
        n_s = len(latent["beta_y"])
        n_ord, _, orders = draw_daily_orders(latent, state, params, rng_sb,
                                             SIM_OBS_START+T_OBS+d, e_match,
                                             demand_shock_override=demand_s[d])
        dlyd, rspend, mkt = run_marketplace(orders, n_s, params,
                                            np.random.default_rng(SYNC_SEED+d),
                                            SIM_OBS_START+T_OBS+d,
                                            hourly_multipliers=hm,
                                            supply_shock_override=supply_s[d])
        if mkt["n_matched"] > 0:
            e_match = (1 - E_MATCH_ALPHA)*e_match + E_MATCH_ALPHA*mkt["avg_match_time"]
        dr_b, bi_b, spend_b, np_b = _collect_sb(mkt)
        sb_dr[d] = dr_b; sb_bi[d] = bi_b; sb_spend[d] = spend_b; sb_np[d] = np_b
        update_daily_state(state, latent, n_ord, rspend, dlyd, SIM_OBS_START+T_OBS+d, params)
        draw_lifecycle_transitions(state, latent, params, rng_sb)

    tau_dm, tau_bc, delta_spend_pct, se_dm, se_bc, se_spend = _jk_se_all(
        sb_dr, sb_bi, sb_spend, sb_np)

    # ── Long-term forks (CN / CE) ──────────────────────────────────────────────
    def _run_lt(seed_offset, hm_trt):
        rng_f = np.random.default_rng(fork_id*100 + seed_offset)
        lat_f = copy.deepcopy(latent_0); st_f = copy.deepcopy(state_0); em_f = e_match_0
        spend_day = np.zeros(T_LT); dr_day = np.zeros(T_LT)
        for d in range(T_LT):
            n_active = (st_f["lifecycle"] != CHURNED).sum()
            n_new = rng_f.poisson(params.lambda_growth * n_active)
            if n_new > 0:
                nl, ns = generate_users(params, rng_f, n=n_new)
                lat_f = _append(lat_f, nl); st_f = _append(st_f, ns)
            n_f = len(lat_f["beta_y"])
            n_ord, _, orders = draw_daily_orders(lat_f, st_f, params, rng_f,
                                                 SIM_OBS_START+T_OBS+d, em_f,
                                                 demand_shock_override=demand_s[d])
            dlyd, rspend, mkt = run_marketplace(orders, n_f, params,
                                                np.random.default_rng(SYNC_SEED+d),
                                                SIM_OBS_START+T_OBS+d,
                                                hourly_multipliers=hm_trt,
                                                supply_shock_override=supply_s[d])
            if mkt["n_matched"] > 0:
                em_f = (1 - E_MATCH_ALPHA)*em_f + E_MATCH_ALPHA*mkt["avg_match_time"]
            n_act2 = (st_f["lifecycle"] != CHURNED).sum()
            spend_day[d] = float(rspend[:n_f].sum()) / max(n_act2, 1)
            dr_day[d]    = mkt["delay_rate"]
            update_daily_state(st_f, lat_f, n_ord, rspend, dlyd, SIM_OBS_START+T_OBS+d, params)
            draw_lifecycle_transitions(st_f, lat_f, params, rng_f)
        return spend_day, dr_day

    cn_spend, cn_dr = _run_lt(10, None)
    ce_spend, ce_dr = _run_lt(10, np.full(24, SUPPLY_MULT))

    tau_GT_28d  = float((ce_dr[:T_SB].mean() - cn_dr[:T_SB].mean()) * 100)
    tau_GT_7d   = float((ce_dr[:7].mean()    - cn_dr[:7].mean())    * 100)
    tau_GT_14d  = float((ce_dr[:14].mean()   - cn_dr[:14].mean())   * 100)
    tau_GT_full = float((ce_dr.mean()        - cn_dr.mean())        * 100)
    _daily_diffs = (ce_dr[:T_SB] - cn_dr[:T_SB]) * 100
    se_GT_28d    = float(_daily_diffs.std() / np.sqrt(T_SB))
    cn_eval      = float(cn_spend[-LT_EVAL_DAYS:].mean())
    ce_eval      = float(ce_spend[-LT_EVAL_DAYS:].mean())
    gt_lt        = float((ce_eval - cn_eval) / cn_eval * 100)
    gt_lt_full   = float((ce_spend.mean() - cn_spend.mean()) / cn_spend.mean() * 100)

    elapsed = time.time() - t0
    print(f"  fork {fork_id:2d}: τ_GT={tau_GT_28d:+.2f}pp  τ_DM={tau_dm*100 if not np.isnan(tau_dm) else float('nan'):+.2f}pp  "
          f"δS_SB={delta_spend_pct:+.2f}%  GT_LT={gt_lt:+.2f}%  [{elapsed:.0f}s]", flush=True)

    return {
        "fork_id":         fork_id,
        "tau_GT_7d":       tau_GT_7d,
        "tau_GT_14d":      tau_GT_14d,
        "tau_GT_28d":      tau_GT_28d,
        "tau_GT_full":     tau_GT_full,
        "se_GT_28d":       se_GT_28d,
        "tau_DM":          float(tau_dm*100)         if not np.isnan(tau_dm)         else None,
        "se_DM":           float(se_dm*100)           if not np.isnan(se_dm)           else None,
        "tau_BC":          float(tau_bc*100)          if not np.isnan(tau_bc)          else None,
        "se_BC":           float(se_bc*100)           if not np.isnan(se_bc)           else None,
        "delta_spend_pct": float(delta_spend_pct)    if not np.isnan(delta_spend_pct) else None,
        "se_spend_pct":    float(se_spend)            if not np.isnan(se_spend)        else None,
        "gt_lt":           gt_lt,
        "gt_lt_full":      gt_lt_full,
        "elapsed_s":       elapsed,
    }

# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print("Switchback Validation Simulation")
    print(f"  OUTER_SEED={OUTER_SEED}  lambda_mem={LAMBDA_MEM}")
    print(f"  supply_mult={SUPPLY_MULT}×  block_hours={BLOCK_HOURS}  N_forks={N_FORK_RUNS}")
    print(f"  T={T_WARMUP}+{T_OBS}+{T_SB}+{T_LT}d  K_fwd={K_FORWARD}d  N_est={N_EST_USERS:,}")
    t_total = time.time()

    rng    = np.random.default_rng(OUTER_SEED)
    params = DGPParams(n_users_initial=N_USERS, seed=OUTER_SEED,
                       lambda_mem=LAMBDA_MEM,
                       courier_scale_base=COURIER_SCALE,
                       sigma_demand=SIGMA_DEMAND)
    latent, state = generate_users_warm(params, rng)

    spend_mat = np.zeros((MAX_N, T_OBS), dtype=np.float32)
    _buf30    = np.zeros((MAX_N, 30), dtype=np.int32); _sum30    = np.zeros(MAX_N, dtype=np.int32)
    _buf7     = np.zeros((MAX_N,  7), dtype=np.int32); _ord_buf7 = np.zeros((MAX_N, 7), dtype=np.int32)
    _sum7     = np.zeros(MAX_N, dtype=np.int32);       _ord_sum7 = np.zeros(MAX_N, dtype=np.int32)
    e_match   = 10.0

    print(f"\n[1/4] Warmup ({T_WARMUP} days)...")
    t0 = time.time()
    for day in range(T_WARMUP):
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
            e_match = (1 - E_MATCH_ALPHA)*e_match + E_MATCH_ALPHA*mkt["avg_match_time"]
        update_daily_state(state, latent, n_ord, rspend, dlyd, day, params)
        draw_lifecycle_transitions(state, latent, params, rng)
        _update_delay_buffers(n, mkt, day, _buf30, _sum30, _buf7, _ord_buf7, _sum7, _ord_sum7)
    print(f"  done ({time.time()-t0:.0f}s)  n={len(latent['beta_y']):,}  e_match={e_match:.1f}min")

    print(f"\n[2/4] Observational phase ({T_OBS} days)...")
    t0 = time.time()
    obs_records = []
    for od in range(T_OBS):
        sim_day = SIM_OBS_START + od
        n_active = (state["lifecycle"] != CHURNED).sum()
        n_new = rng.poisson(params.lambda_growth * n_active)
        if n_new > 0:
            nl, ns = generate_users(params, rng, n=n_new)
            latent = _append(latent, nl); state = _append(state, ns)
        n = len(latent["beta_y"]); assert n <= MAX_N
        dr30 = _sum30[:n] / np.maximum(state["F30"][:n], 1.)
        dr7  = _sum7[:n]  / np.maximum(_ord_sum7[:n], 1.)
        n_ord, _, orders = draw_daily_orders(latent, state, params, rng, sim_day,
                                             expected_matching_time=e_match)
        dlyd, rspend, mkt = run_marketplace(orders, n, params, rng, sim_day)
        if mkt["n_matched"] > 0:
            e_match = (1 - E_MATCH_ALPHA)*e_match + E_MATCH_ALPHA*mkt["avg_match_time"]
        spend_mat[:n, od] = rspend[:n].astype(np.float32)
        if mkt["n_matched"] > 0:
            mm = mkt["matched_per_order"].astype(bool)
            ou = mkt["order_user_idx"][mm]
            obs_records.append(pd.DataFrame({
                "user_id":        ou.astype(np.int32),
                "obs_day":        np.full(len(ou), od, dtype=np.int32),
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
        update_daily_state(state, latent, n_ord, rspend, dlyd, sim_day, params)
        draw_lifecycle_transitions(state, latent, params, rng)
        _update_delay_buffers(n, mkt, sim_day, _buf30, _sum30, _buf7, _ord_buf7, _sum7, _ord_sum7)
    obs_full = pd.concat(obs_records, ignore_index=True)
    n_fork   = len(latent["beta_y"])
    print(f"  done ({time.time()-t0:.0f}s)  obs_rows={len(obs_full):,}  n={n_fork:,}")

    cum_spend = np.zeros((n_fork, T_OBS+1), dtype=np.float32)
    cum_spend[:,1:] = np.cumsum(spend_mat[:n_fork].astype(np.float64), axis=1).astype(np.float32)
    del spend_mat

    obs_est  = obs_full[obs_full["obs_day"] <= EST_OBS_MAX].copy()
    rng_s    = np.random.default_rng(OUTER_SEED + 9999)
    all_uids = obs_est["user_id"].unique()
    if N_EST_USERS > 0 and N_EST_USERS < len(all_uids):
        samp_u  = rng_s.choice(all_uids, size=N_EST_USERS, replace=False)
        obs_est = obs_est[obs_est["user_id"].isin(set(samp_u))].copy()
    print(f"  estimation sample: {len(obs_est):,} rows from {obs_est['user_id'].nunique():,} users")

    print(f"\n[3/4] CBPS estimation + {N_BOOTSTRAP} bootstrap CIs...")
    t0 = time.time()
    beta_full = _run_cbps_on(obs_est, cum_spend)
    t_one = time.time() - t0
    print(f"  full-data CBPS: β={beta_full*100:+.4f}%/pp  ({t_one:.1f}s)")

    uid_col    = obs_est["user_id"].values
    uid_unique = np.unique(uid_col)
    uid_to_idx = {u: np.where(uid_col == u)[0] for u in uid_unique}
    rng_bs     = np.random.default_rng(OUTER_SEED + 777)
    bs_betas   = np.zeros(N_BOOTSTRAP)
    for b in range(N_BOOTSTRAP):
        boot_u      = rng_bs.choice(uid_unique, size=len(uid_unique), replace=True)
        idx_b       = np.concatenate([uid_to_idx[u] for u in boot_u])
        bs_betas[b] = _run_cbps_on(obs_est.iloc[idx_b], cum_spend)
        if (b+1) % 5 == 0:
            print(f"  bootstrap {b+1}/{N_BOOTSTRAP}  β={bs_betas[b]*100:+.4f}%/pp", flush=True)

    beta_mean  = float(beta_full * 100)
    beta_se    = float(bs_betas.std() * 100)
    beta_ci_lo = float(np.percentile(bs_betas, 2.5)  * 100)
    beta_ci_hi = float(np.percentile(bs_betas, 97.5) * 100)
    print(f"  β={beta_mean:+.4f}  SE={beta_se:.4f}  95%CI=[{beta_ci_lo:.4f}, {beta_ci_hi:.4f}]%/pp"
          f"  ({time.time()-t0:.0f}s)")

    _SHARED["latent_0"]     = copy.deepcopy(latent)
    _SHARED["state_0"]      = copy.deepcopy(state)
    _SHARED["e_match_0"]    = e_match
    _SHARED["params"]       = params
    rng_crn = np.random.default_rng(50)
    _SHARED["supply_fixed"] = rng_crn.lognormal(0., params.sigma_supply, max(T_SB, T_LT))
    _SHARED["demand_fixed"] = rng_crn.lognormal(0., params.sigma_demand, max(T_SB, T_LT))

    print(f"\n[4/4] Fork phase ({N_FORK_RUNS} forks, {multiprocessing.cpu_count()} CPUs available)...")
    t0 = time.time()
    with multiprocessing.Pool(N_FORK_RUNS) as pool:
        fork_results = pool.map(_run_fork, list(range(N_FORK_RUNS)))
    fork_results.sort(key=lambda r: r["fork_id"])
    print(f"  done ({time.time()-t0:.0f}s wall time)")

    out = {
        "config": {
            "outer_seed":    OUTER_SEED,
            "lambda_mem":    LAMBDA_MEM,    "supply_mult":  SUPPLY_MULT,
            "courier_scale": COURIER_SCALE, "n_fork_runs":  N_FORK_RUNS,
            "n_bootstrap":   N_BOOTSTRAP,   "k_forward":    K_FORWARD,
            "sigma_demand":  SIGMA_DEMAND,  "n_est_users":  N_EST_USERS,
            "block_h":       BLOCK_HOURS,
        },
        "beta_cbps": {
            "mean_pp": beta_mean, "se_pp": beta_se,
            "ci_lo_pp": beta_ci_lo, "ci_hi_pp": beta_ci_hi,
            "bootstrap_betas_pp": (bs_betas * 100).tolist(),
        },
        "fork_results": fork_results,
    }
    with open(str(RESULTS_PATH), "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved → {RESULTS_PATH}")
    print(f"Total wall time: {(time.time()-t_total)/60:.1f} min")
