"""
Switchback Validation Simulation — Table 2
============================================
Runs 200 independent market simulations and records, for each, the three
switchback estimators against that run's own paired ground truth:

  1. Switchback delay   : tau_DM          vs tau_GT_28d   (pp)
  2. Switchback ΔLTV     : delta_spend_pct vs gt_lt        (%)
  3. Chronos prediction : tau_DM × β       vs gt_lt        (%)

Here every run draws its own supply / demand / courier RNG streams, so the
200 runs are fully independent market realizations — the honest sampling distribution
used for Table 2's bias / RMSE / coverage.

Within each run the switchback arm and the two long-term arms (CN / CE) share that
run's user-RNG seed, supply/demand shocks, and courier seed (CRN), so the per-run
ground truth is the matched counterfactual for that run's market.

β (the Chronos CBPS-IPW slope) is deterministic in the observational phase and is
identical to the box-plot run, so it is reused from that run's results rather than
recomputed. Run run_simulation.py first so sw_validation_results.json exists.

Runtime: a few hours wall time (200 runs × 3 arms each, parallel across cores).

Usage (from repository root):
    python switchback_validation/run_table_simulation.py

Output:
    switchback_validation/results/table2_results.json   (raw per-run estimates)
    Table 2 (bias / RMSE / coverage) is printed to stdout at the end of the run.
"""
from __future__ import annotations
import copy, json, os, sys, time, multiprocessing
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np

from sim.params import DGPParams
from sim.users import (generate_users, generate_users_warm,
                       update_daily_state, draw_lifecycle_transitions, CHURNED)
from sim.demand import draw_daily_orders
from sim.marketplace import run_marketplace

# ── Simulation parameters (paper config — identical to run_simulation.py) ──────
OUTER_SEED    = 42
LAMBDA_MEM    = 0.06    # delay-memory decay; half-life ≈ ln2/0.06 ≈ 11.6 days
SIGMA_DEMAND  = 0.1     # daily macro shock std
COURIER_SCALE = 700
SUPPLY_MULT   = 1.25    # treatment: 1.25× courier rate in treated blocks
N_USERS       = 100_000
BLOCK_HOURS   = 3       # 8 three-hour blocks per day
N_RUNS        = int(os.environ.get("N_RUNS", "200"))  # independent market replications
T_WARMUP      = 100
T_OBS         = 150
T_SB          = 28      # switchback duration (days)
T_LT          = 120     # long-term follow-up duration (days)
LT_EVAL_DAYS  = 21      # last N days used for LT spending ground truth
MAX_N         = 200_000
E_MATCH_ALPHA = 0.05    # EWMA weight for avg_match_time
SIM_OBS_START = T_WARMUP
N_BLOCKS      = 24 // BLOCK_HOURS
BC_SCALE      = (BLOCK_HOURS - 1) / BLOCK_HOURS
BC_B_OVER_L   = 1 / (BLOCK_HOURS - 1)
MIN_MATCHED   = 5
N_WORKERS     = min(N_RUNS, multiprocessing.cpu_count())

RESULTS_DIR  = Path(__file__).parent / "results"
RESULTS_PATH = RESULTS_DIR / "table2_results.json"
BETA_PATH    = RESULTS_DIR / "sw_validation_results.json"  # reuse deterministic β

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

def _collect_sb(mkt):
    """Per-block delay rate, burn-in rate, total spend, n_placed (all excl 1h burn-in)."""
    oh      = mkt["order_hour"].astype(int)
    og      = mkt["order_spend"]
    mm      = mkt["matched_per_order"].astype(bool)
    dl      = mkt["delayed_per_order"].astype(bool)
    blk     = oh // BLOCK_HOURS
    hib     = oh % BLOCK_HOURS
    excl    = hib >= 1                       # exclude first hour of each block
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
    """Day-level cluster jackknife SE: drop all blocks from one day at a time."""
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

# ── Fork worker ───────────────────────────────────────────────────────────────
def _run_fork(fork_id):
    latent_0  = _SHARED["latent_0"]
    state_0   = _SHARED["state_0"]
    e_match_0 = _SHARED["e_match_0"]
    params    = _SHARED["params"]
    # INDEPENDENT market: each run draws its OWN supply & demand shocks from a
    # per-run seed → every run is a fresh, fully independent market realisation.
    # (SB/CN/CE within this run still share these shocks → GT stays a valid
    #  CRN-paired truth for THIS run's market.)
    rng_shock = np.random.default_rng([fork_id, 30000])
    supply_s  = rng_shock.lognormal(0., params.sigma_supply, max(T_SB, T_LT))
    demand_s  = rng_shock.lognormal(0., params.sigma_demand, max(T_SB, T_LT))
    t0 = time.time()

    # ── Switchback arm ───────────────────────────────────────────────────────────
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
        # INDEPENDENT couriers: per-run seed [fork_id, 20000, d] — independent across
        # runs, shared by SB/CN/CE within this run (CRN).
        dlyd, rspend, mkt = run_marketplace(orders, n_s, params,
                                            np.random.default_rng([fork_id, 20000, d]),
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

    # ── Long-term arms (CN / CE) ───────────────────────────────────────────────
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
                                                np.random.default_rng([fork_id, 20000, d]),
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
    print(f"  run {fork_id:3d}: τ_GT={tau_GT_28d:+.2f}pp  "
          f"τ_DM={tau_dm*100 if not np.isnan(tau_dm) else float('nan'):+.2f}pp  "
          f"δLTV_SB={delta_spend_pct:+.2f}%  GT_LT={gt_lt:+.2f}%  [{elapsed:.0f}s]", flush=True)

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
        "se_BC":           float(se_bc*100)           if not np.isnan(se_bc)          else None,
        "delta_spend_pct": float(delta_spend_pct)    if not np.isnan(delta_spend_pct) else None,
        "se_spend_pct":    float(se_spend)            if not np.isnan(se_spend)        else None,
        "gt_lt":           gt_lt,
        "gt_lt_full":      gt_lt_full,
        "elapsed_s":       elapsed,
    }

# ── Table 2 (printed to stdout / run log) ─────────────────────────────────────
def print_table2(fork_results, beta_cbps):
    """Aggregate the runs into Table 2: bias / RMSE / 95%-CI coverage for the three
    estimators, each scored against its OWN per-run ground truth (CRN-paired):

      1. Switchback delay  : tau_DM          vs tau_GT_28d  (pp)
      2. Switchback ΔLTV    : delta_spend_pct vs gt_lt       (%)
      3. Chronos prediction : tau_DM × β       vs gt_lt       (%)

    coverage = fraction of runs whose 95% CI (est ± 1.96·se) contains the truth.
    Chronos se via delta method: sqrt(β²·se_DM² + τ_DM²·se_β²)."""
    beta, se_beta = beta_cbps["mean_pp"], beta_cbps["se_pp"]
    fr = [r for r in fork_results
          if None not in (r["tau_DM"], r["se_DM"], r["delta_spend_pct"],
                          r["se_spend_pct"], r["gt_lt"], r["tau_GT_28d"])]
    g = lambda k: np.array([r[k] for r in fr], float)
    tau_dm,   se_dm     = g("tau_DM"),          g("se_DM")
    dspend,   se_dspend = g("delta_spend_pct"), g("se_spend_pct")
    gt_delay, gt_lt     = g("tau_GT_28d"),      g("gt_lt")
    pred    = tau_dm * beta
    se_pred = np.sqrt(beta**2 * se_dm**2 + tau_dm**2 * se_beta**2)

    def metrics(est, se, truth):
        err = est - truth
        cov = np.mean((truth >= est - 1.96*se) & (truth <= est + 1.96*se)) * 100
        return est.mean(), truth.mean(), err.mean(), np.sqrt((err**2).mean()), cov

    rows = [
        ("Switchback delay  (tau_DM)",   tau_dm, se_dm,     gt_delay, "pp"),
        ("Switchback dLTV   (dspend)",   dspend, se_dspend, gt_lt,    "%"),
        ("Chronos dLTV      (tau_DM*b)", pred,   se_pred,   gt_lt,    "%"),
    ]
    print(f"\n=== Table 2: estimator performance across {len(fr)} independent markets ===")
    print(f"beta = {beta:+.4f} %/pp (se {se_beta:.4f})")
    print(f"{'Estimator':<28}{'Mean Est':>11}{'Mean True':>12}{'Bias':>9}{'RMSE':>8}{'Cov95%':>8}")
    for name, est, se, truth, u in rows:
        me, mt, b, r, c = metrics(est, se, truth)
        print(f"{name:<28}{f'{me:+.2f} {u}':>11}{f'{mt:+.2f} {u}':>12}"
              f"{b:>+9.2f}{r:>8.2f}{c:>7.0f}%")

# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if not BETA_PATH.exists():
        raise FileNotFoundError(
            f"β source not found: {BETA_PATH}\n"
            "Run switchback_validation/run_simulation.py first (it estimates β once)."
        )
    beta_cbps = json.load(open(str(BETA_PATH)))["beta_cbps"]

    print("Switchback Validation Simulation — Table 2 (200 independent markets)")
    print(f"  OUTER_SEED={OUTER_SEED}  lambda_mem={LAMBDA_MEM}  supply_mult={SUPPLY_MULT}×")
    print(f"  block_hours={BLOCK_HOURS}  N_runs={N_RUNS}  T={T_WARMUP}+{T_OBS}+{T_SB}+{T_LT}d")
    print(f"  β reused from {BETA_PATH.name}: {beta_cbps['mean_pp']:+.4f}%/pp (se {beta_cbps['se_pp']:.4f})")
    t_total = time.time()

    rng    = np.random.default_rng(OUTER_SEED)
    params = DGPParams(n_users_initial=N_USERS, seed=OUTER_SEED,
                       lambda_mem=LAMBDA_MEM,
                       courier_scale_base=COURIER_SCALE,
                       sigma_demand=SIGMA_DEMAND)
    latent, state = generate_users_warm(params, rng)
    e_match = 10.0

    # The warmup + observational phases evolve the shared population that every run
    # forks from. They are deterministic in OUTER_SEED (identical to run_simulation.py);
    # β is reused from the box-plot run, so the CBPS fit is not repeated here.
    print(f"\n[1/3] Warmup ({T_WARMUP} days)...")
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
    print(f"  done ({time.time()-t0:.0f}s)  n={len(latent['beta_y']):,}  e_match={e_match:.1f}min")

    print(f"\n[2/3] Observational phase ({T_OBS} days)...")
    t0 = time.time()
    for od in range(T_OBS):
        sim_day = SIM_OBS_START + od
        n_active = (state["lifecycle"] != CHURNED).sum()
        n_new = rng.poisson(params.lambda_growth * n_active)
        if n_new > 0:
            nl, ns = generate_users(params, rng, n=n_new)
            latent = _append(latent, nl); state = _append(state, ns)
        n = len(latent["beta_y"]); assert n <= MAX_N
        n_ord, _, orders = draw_daily_orders(latent, state, params, rng, sim_day,
                                             expected_matching_time=e_match)
        dlyd, rspend, mkt = run_marketplace(orders, n, params, rng, sim_day)
        if mkt["n_matched"] > 0:
            e_match = (1 - E_MATCH_ALPHA)*e_match + E_MATCH_ALPHA*mkt["avg_match_time"]
        update_daily_state(state, latent, n_ord, rspend, dlyd, sim_day, params)
        draw_lifecycle_transitions(state, latent, params, rng)
    print(f"  done ({time.time()-t0:.0f}s)  n={len(latent['beta_y']):,}")

    _SHARED["latent_0"]  = copy.deepcopy(latent)
    _SHARED["state_0"]   = copy.deepcopy(state)
    _SHARED["e_match_0"] = e_match
    _SHARED["params"]    = params

    print(f"\n[3/3] Run phase ({N_RUNS} independent markets, {N_WORKERS} workers)...")
    t0 = time.time()
    with multiprocessing.Pool(N_WORKERS) as pool:
        fork_results = pool.map(_run_fork, list(range(N_RUNS)))
    fork_results.sort(key=lambda r: r["fork_id"])
    print(f"  done ({time.time()-t0:.0f}s wall time)")

    out = {
        "config": {
            "outer_seed":    OUTER_SEED,
            "lambda_mem":    LAMBDA_MEM,    "supply_mult":  SUPPLY_MULT,
            "courier_scale": COURIER_SCALE, "n_runs":       N_RUNS,
            "sigma_demand":  SIGMA_DEMAND,  "block_h":      BLOCK_HOURS,
            "k_forward":     56,            "n_est_users":  0,
        },
        "beta_cbps": beta_cbps,
        "fork_results": fork_results,
    }
    with open(str(RESULTS_PATH), "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved → {RESULTS_PATH}")
    print_table2(fork_results, beta_cbps)
    print(f"\nTotal wall time: {(time.time()-t_total)/60:.1f} min")
