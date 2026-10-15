"""Isolated, paired switchback burn-in sensitivity.

This is a copy of the paper's Table-2 simulator with its courier/order DGP
unchanged. In each independent fork, a single 28-day switchback realization
is analyzed at 60 and 80 minutes after each three-hour block starts,
using actual continuous placement timestamps. The original 60-minute
eligibility is asserted to match hour-binned published code exactly.
Orders are assigned by placement block; their fulfillment may continue
after a switch.

The long-term baseline/treatment arms share each fork's daily macro shocks
and courier draws as in Table 2. They are run once per fork, not once per
burn-in. The reported paired difference may combine carryover and shifts in
the within-block hours represented by the estimator.

Usage: SW_BURN_USERS=100000 SW_BURN_N_RUNS=200 SW_BURN_WORKERS=8 \
    python -B switchback_validation/carryover_sensitivity/run_different_burn_in.py
"""
from __future__ import annotations
import copy, json, os, sys, time, multiprocessing
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
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
N_USERS       = int(os.environ.get("SW_BURN_USERS", "100000"))
BLOCK_HOURS   = 3       # 8 three-hour blocks per day
N_RUNS        = int(os.environ.get("SW_BURN_N_RUNS", "200"))
BURNS_MIN     = tuple(int(v) for v in os.environ.get(
    "SW_BURN_MINUTES", "60,80").split(","))
QUEUE_TIMEOUT_MIN = int(os.environ.get("SW_BURN_QUEUE_TIMEOUT", "80"))
if BURNS_MIN != (60, 80) or QUEUE_TIMEOUT_MIN <= 0:
    raise ValueError("This response analysis uses 60/80-minute burn-ins "
                     "and a positive queue timeout.")
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
N_WORKERS     = min(N_RUNS, int(os.environ.get("SW_BURN_WORKERS", "8")))

RESULTS_DIR  = Path(__file__).parent / "results"
_suffix = f"_q{QUEUE_TIMEOUT_MIN}_b{'-'.join(map(str,BURNS_MIN))}"
RESULTS_PATH = RESULTS_DIR / (
    f"paired_burnin_{N_USERS}_eaters_{N_RUNS}_markets{_suffix}.json"
)

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

def _collect_sb(mkt, order_time, burn_min):
    """Block outcomes attributed by placement, with minute-resolution burn-in.

    The published 60-minute eligible mask is reproduced exactly. Both
    masks act on the same simulated orders; no marketplace is rerun per burn.
    """
    oh      = mkt["order_hour"].astype(int)
    og      = mkt["order_spend"]
    mm      = mkt["matched_per_order"].astype(bool)
    dl      = mkt["delayed_per_order"].astype(bool)
    if not np.array_equal(oh, np.floor(order_time / 60).astype(int)):
        raise AssertionError("Marketplace and placed-order timestamps not aligned")
    blk     = oh // BLOCK_HOURS
    mins    = order_time - blk * (60 * BLOCK_HOURS)
    excl    = mins >= burn_min
    if burn_min == 60 and not np.array_equal(excl, (oh % BLOCK_HOURS) >= 1):
        raise AssertionError("60-minute eligibility must reproduce original code")
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
    sb_by_burn = {
        burn: {
            "dr": np.full((T_SB, N_BLOCKS), np.nan),
            "bi": np.full((T_SB, N_BLOCKS), np.nan),
            "spend": np.full((T_SB, N_BLOCKS), np.nan),
            "np": np.full((T_SB, N_BLOCKS), np.nan),
        }
        for burn in BURNS_MIN
    }

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
        for burn, acc in sb_by_burn.items():
            dr_b, bi_b, spend_b, np_b = _collect_sb(
                mkt, orders["order_time"], burn)
            acc["dr"][d] = dr_b; acc["bi"][d] = bi_b
            acc["spend"][d] = spend_b; acc["np"][d] = np_b
        update_daily_state(state, latent, n_ord, rspend, dlyd, SIM_OBS_START+T_OBS+d, params)
        draw_lifecycle_transitions(state, latent, params, rng_sb)

    burn_estimates = {}
    for burn, acc in sb_by_burn.items():
        tau_dm, tau_bc, delta_spend_pct, se_dm, se_bc, se_spend = _jk_se_all(
            acc["dr"], acc["bi"], acc["spend"], acc["np"]
        )
        burn_estimates[burn] = {
            "tau_DM": float(tau_dm * 100) if np.isfinite(tau_dm) else None,
            "se_DM": float(se_dm * 100) if np.isfinite(se_dm) else None,
            "delta_spend_pct": (float(delta_spend_pct)
                                if np.isfinite(delta_spend_pct) else None),
            "se_spend_pct": float(se_spend) if np.isfinite(se_spend) else None,
            # BC weights in the original code were derived for one hour.
            "tau_BC": (float(tau_bc * 100)
                       if burn == 60 and np.isfinite(tau_bc) else None),
        }

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
    short = burn_estimates[60]
    print(f"  run {fork_id:3d}: τ_GT={tau_GT_28d:+.2f}pp  "
          f"τ_DM(60)={short['tau_DM'] if short['tau_DM'] is not None else float('nan'):+.2f}pp  "
          f"δLTV_SB(60)={short['delta_spend_pct'] if short['delta_spend_pct'] is not None else float('nan'):+.2f}%  "
          f"GT_LT={gt_lt:+.2f}%  [{elapsed:.0f}s]", flush=True)

    return {
        "fork_id":         fork_id,
        "tau_GT_7d":       tau_GT_7d,
        "tau_GT_14d":      tau_GT_14d,
        "tau_GT_28d":      tau_GT_28d,
        "tau_GT_full":     tau_GT_full,
        "se_GT_28d":       se_GT_28d,
        "burn_estimates":  burn_estimates,
        "gt_lt":           gt_lt,
        "gt_lt_full":      gt_lt_full,
        "elapsed_s":       elapsed,
    }

# ── Table 2 (printed to stdout / run log) ─────────────────────────────────────
def summarize_burns(fork_results):
    """Paired across identical full-size markets; no observational β refit."""
    output = []
    for burn in BURNS_MIN:
        selected = [
            r for r in fork_results
            if all(r["burn_estimates"][burn][key] is not None for key in (
                "tau_DM", "se_DM", "delta_spend_pct", "se_spend_pct"))
        ]
        if len(selected) != len(fork_results):
            raise ValueError(f"Missing block-level estimates at burn={burn}")
        estimates = [r["burn_estimates"][burn] for r in selected]
        est = np.array([e["tau_DM"] for e in estimates])
        se = np.array([e["se_DM"] for e in estimates])
        truth = np.array([r["tau_GT_28d"] for r in selected])
        spend = np.array([e["delta_spend_pct"] for e in estimates])
        spend_se = np.array([e["se_spend_pct"] for e in estimates])
        spend_truth = np.array([r["gt_lt"] for r in selected])
        output.append({
            "burn_min": burn, "markets": len(selected),
            "delay_mean_pp": float(est.mean()),
            "delay_truth_mean_pp": float(truth.mean()),
            "delay_bias_pp": float(np.mean(est - truth)),
            "delay_rmse_pp": float(np.sqrt(np.mean((est-truth)**2))),
            "delay_coverage_95_pct": float(100 * np.mean(
                (truth >= est - 1.96*se) & (truth <= est + 1.96*se))),
            "spend_mean_pct": float(spend.mean()),
            "long_run_spend_truth_pct": float(spend_truth.mean()),
            "spend_gap_pp": float(np.mean(spend - spend_truth)),
            "spend_coverage_95_pct": float(100 * np.mean(
                (spend_truth >= spend - 1.96*spend_se) &
                (spend_truth <= spend + 1.96*spend_se))),
        })
    for r in output:
        print(f"burn={r['burn_min']:3d} min  "
              f"delay={r['delay_mean_pp']:+.3f} pp  "
              f"bias={r['delay_bias_pp']:+.3f} pp  "
              f"rmse={r['delay_rmse_pp']:.3f} pp  "
              f"coverage={r['delay_coverage_95_pct']:.0f}%  "
              f"spend={r['spend_mean_pct']:+.3f}%",
              flush=True)
    return output

# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print("Local matched-market switchback burn-in sensitivity")
    print(f"  OUTER_SEED={OUTER_SEED}  lambda_mem={LAMBDA_MEM}  supply_mult={SUPPLY_MULT}×")
    print(f"  block_hours={BLOCK_HOURS}  burns={BURNS_MIN}min  "
          f"N_runs={N_RUNS}, N_users={N_USERS:,}, workers={N_WORKERS}")
    print(f"  T={T_WARMUP}+{T_OBS}+{T_SB}+{T_LT}d")
    t_total = time.time()

    rng    = np.random.default_rng(OUTER_SEED)
    params = DGPParams(n_users_initial=N_USERS, seed=OUTER_SEED,
                       lambda_mem=LAMBDA_MEM,
                       courier_scale_base=COURIER_SCALE,
                       sigma_demand=SIGMA_DEMAND)
    params.max_order_wait_min = float(QUEUE_TIMEOUT_MIN)
    latent, state = generate_users_warm(params, rng)
    e_match = 10.0

    # The warmup + observational phases evolve the shared population that every run
    # forks from. They are deterministic in OUTER_SEED (identical to run_simulation.py);
    # This burn-in check concerns switchback DM/spend, not observational β.
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
            "burns_min":     BURNS_MIN,      "n_users":      N_USERS,
            "max_order_wait_min": params.max_order_wait_min,
            "max_courier_wait_min": params.max_courier_wait_min,
        },
        "fork_results": fork_results,
    }
    out["summary"] = summarize_burns(fork_results)
    with open(str(RESULTS_PATH), "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved → {RESULTS_PATH}")
    print(f"\nTotal wall time: {(time.time()-t_total)/60:.1f} min")
