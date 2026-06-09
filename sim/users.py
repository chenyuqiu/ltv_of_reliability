"""
User population: initial cohort generation and daily state updates.

Two dicts are returned by generate_users():
  latent — fixed per-user draws from joint LogNormal; never mutated after creation
  state  — mutable per-user state; updated each simulation day
"""
from __future__ import annotations

import numpy as np

from .params import DGPParams

# ── Lifecycle state constants ──────────────────────────────────────────────────
NEW, CASUAL, POWER, AT_RISK, CHURNED = 0, 1, 2, 3, 4
N_STATES = 5


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def generate_users(
    params: DGPParams, rng: np.random.Generator, n: int | None = None
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """
    Draw a user cohort from the joint LogNormal population distribution.

    Returns
    -------
    latent : dict
        Fixed per-user parameters.
        Keys: beta_y, beta_d, beta_spend, beta_tip, sigma_y, mu_dist, mu_prep
    state : dict
        Mutable per-user state.
        Keys: lifecycle, D, R, F30, S30, spend_total, n_orders, ordered_today,
              _orders_buf (n×30), _spend_buf (n×30)
    """
    n = params.n_users_initial if n is None else n

    log_means = np.array([
        params.log_beta_y_mean, params.log_beta_d_mean, params.log_beta_spend_mean,
        params.log_beta_tip_mean, params.log_sigma_y_mean,
        params.log_mu_dist_mean, params.log_mu_prep_mean,
    ])
    log_stds = np.array([
        params.log_beta_y_std, params.log_beta_d_std, params.log_beta_spend_std,
        params.log_beta_tip_std, params.log_sigma_y_std,
        params.log_mu_dist_std, params.log_mu_prep_std,
    ])
    cov = log_stds[:, None] * params.user_corr * log_stds[None, :]
    log_draws = rng.multivariate_normal(log_means, cov, size=n)  # (n, 7)

    latent: dict[str, np.ndarray] = {
        "beta_y":   np.exp(log_draws[:, 0]),
        "beta_d":   np.exp(log_draws[:, 1]),
        "beta_spend":  np.exp(log_draws[:, 2]),
        "beta_tip": np.exp(log_draws[:, 3]),
        "sigma_y":  np.exp(log_draws[:, 4]),
        "mu_dist":  np.exp(log_draws[:, 5]),
        "mu_prep":  np.exp(log_draws[:, 6]),
    }

    state: dict[str, np.ndarray] = {
        "lifecycle":     np.full(n, NEW, dtype=np.int8),
        "D":             np.ones(n, dtype=np.float64),
        "R":             np.zeros(n, dtype=np.int32),
        "F30":           np.zeros(n, dtype=np.int32),
        "S30":          np.zeros(n, dtype=np.float64),
        "spend_total":      np.zeros(n, dtype=np.float64),
        "n_orders":      np.zeros(n, dtype=np.int32),
        "ordered_today": np.zeros(n, dtype=np.bool_),
        "_orders_buf":   np.zeros((n, 30), dtype=np.int32),
        "_spend_buf":       np.zeros((n, 30), dtype=np.float64),
    }

    return latent, state


def update_daily_state(
    state: dict,
    latent: dict,
    n_orders_today: np.ndarray,
    gb_today: np.ndarray,
    delayed_mask: np.ndarray,
    sim_day: int,
    params: DGPParams,
) -> None:
    """
    Update all mutable per-user state at end of day. Mutates state in-place.

    Parameters
    ----------
    n_orders_today : int array (n_users,)
    gb_today       : float array (n_users,)
    delayed_mask   : bool array (n_users,)  True if ≥1 delayed order today
    sim_day        : 0-indexed simulation day (rotates the 30-day circular buffer)

    F30 / S30 maintained via circular buffers: on day d, slot d % 30 is
    overwritten with today's counts and the running sum updated accordingly.

    Delay memory:  log D(t+1) = exp(−λ_mem) · log D(t) − β_d · 1[delayed]
    """
    ordered = n_orders_today > 0

    buf_idx = sim_day % 30
    state["F30"]  -= state["_orders_buf"][:, buf_idx]
    state["S30"] -= state["_spend_buf"][:, buf_idx]
    state["_orders_buf"][:, buf_idx] = n_orders_today
    state["_spend_buf"][:, buf_idx]     = gb_today
    state["F30"]  += n_orders_today
    state["S30"] += gb_today

    state["spend_total"]  += gb_today
    state["n_orders"]  += n_orders_today
    state["ordered_today"] = ordered
    state["R"] = np.where(ordered, 0, state["R"] + 1)

    log_D  = np.exp(-params.lambda_mem) * np.log(state["D"])
    log_D -= latent["beta_d"] * delayed_mask
    state["D"] = np.exp(log_D)


def draw_lifecycle_transitions(
    state: dict,
    latent: dict,
    params: DGPParams,
    rng: np.random.Generator,
) -> None:
    """
    Evaluate and draw Markov-chain lifecycle transitions for all users.
    Mutates state["lifecycle"] in-place.

    Transition graph:
      New → Casual → Power
       ↓       ↓       ↓
       └──→ AtRisk ←───┘
               ↓ (recovery)
             Casual
               ↓
             Churned → New (reactivation)

    Conflict resolution (two transitions fire same day):
      New:    Casual > AtRisk
      Casual: Power  > AtRisk
      AtRisk: Casual (recovery) > Churned
    """
    p = params
    lc = state["lifecycle"]

    # log1p(n_orders) capped at log1p(30): prevents unbounded tenure signal
    # from accumulating and locking long-tenured users in Power permanently.
    log1p_n   = np.log1p(np.minimum(state["n_orders"].astype(float), 30.0))
    neg_log_D = -np.log(np.maximum(state["D"], 1e-9))
    ord_f     = state["ordered_today"].astype(float)

    new_lc = lc.copy()

    # ── New → Casual ──────────────────────────────────────────────────────────
    # Hard gate: must have ordered today.
    new_casual_mask = (lc == NEW) & state["ordered_today"]
    if new_casual_mask.any():
        idx = np.where(new_casual_mask)[0]
        logit = (p.gamma_NC
                 + p.delta_n   * log1p_n[new_casual_mask]
                 + p.delta_ord * ord_f[new_casual_mask]
                 - p.delta_D   * neg_log_D[new_casual_mask])
        new_lc[idx[rng.random(new_casual_mask.sum()) < _sigmoid(logit)]] = CASUAL

    # ── New → AtRisk ──────────────────────────────────────────────────────────
    # Evaluated for all users still New after the N→C draw.
    new_atrisk_mask = (new_lc == NEW)
    if new_atrisk_mask.any():
        logit = (p.gamma_NA
                 + p.delta_R   * state["R"][new_atrisk_mask]
                 + p.delta_D   * neg_log_D[new_atrisk_mask]
                 - p.delta_n   * log1p_n[new_atrisk_mask]
                 - p.delta_ord * ord_f[new_atrisk_mask])
        fires = rng.random(new_atrisk_mask.sum()) < _sigmoid(logit)
        new_lc[np.where(new_atrisk_mask)[0][fires]] = AT_RISK

    # ── Casual → Power / AtRisk ───────────────────────────────────────────────
    # Power gate: ordered_today AND F30 ≥ power_min_F30.
    mask = lc == CASUAL
    if mask.any():
        idx = np.where(mask)[0]
        logit_power = (p.gamma_CP
                       + p.delta_F   * state["F30"][mask]
                       + p.delta_S  * state["S30"][mask]
                       - p.delta_R   * state["R"][mask]
                       + p.delta_n   * log1p_n[mask]
                       + p.delta_ord * ord_f[mask]
                       - p.delta_D   * neg_log_D[mask])
        logit_atrisk = (p.gamma_CA
                        + p.delta_R   * state["R"][mask]
                        + p.delta_D   * neg_log_D[mask]
                        - p.delta_n   * log1p_n[mask]
                        - p.delta_ord * ord_f[mask])
        fire_power  = (rng.random(mask.sum()) < _sigmoid(logit_power)) \
                      & ord_f[mask].astype(bool) \
                      & (state["F30"][mask] >= p.power_min_F30)
        fire_atrisk = (rng.random(mask.sum()) < _sigmoid(logit_atrisk)) & ~fire_power
        new_lc[idx[fire_power]]  = POWER
        new_lc[idx[fire_atrisk]] = AT_RISK

    # ── Power → AtRisk / Casual ───────────────────────────────────────────────
    # delta_n excluded from Power→AtRisk: lifetime orders should not protect
    # Power users from disengagement; only current frequency (F30) and recency (R) do.
    # Conflict resolution is delay-conditional:
    #   D ≥ 0.95 (clean history) → Power→Casual wins (drifting, not frustrated)
    #   D < 0.95 (recent delay)  → Power→AtRisk wins (escalate to at-risk)
    mask = lc == POWER
    if mask.any():
        idx = np.where(mask)[0]
        logit_pa = (p.gamma_PA
                    + p.delta_R   * state["R"][mask]
                    + p.delta_D   * neg_log_D[mask]
                    - p.delta_F   * state["F30"][mask]
                    - p.delta_S  * state["S30"][mask])
        logit_pc = (p.gamma_PC
                    + p.delta_D   * neg_log_D[mask]
                    + p.delta_R   * state["R"][mask]
                    - p.delta_F   * state["F30"][mask]
                    - p.delta_S  * state["S30"][mask])
        fire_atrisk = rng.random(mask.sum()) < _sigmoid(logit_pa)
        fire_casual = rng.random(mask.sum()) < _sigmoid(logit_pc)
        has_delay   = state["D"][mask] < 0.95
        fire_casual[has_delay]  &= ~fire_atrisk[has_delay]
        fire_atrisk[~has_delay] &= ~fire_casual[~has_delay]
        new_lc[idx[fire_atrisk]] = AT_RISK
        new_lc[idx[fire_casual]] = CASUAL

    # ── AtRisk → Casual (recovery) / Churned ─────────────────────────────────
    mask = lc == AT_RISK
    if mask.any():
        idx = np.where(mask)[0]
        logit_rec = (p.gamma_RAC
                     - p.delta_R     * state["R"][mask]
                     + p.delta_D_rec * np.log(state["D"][mask])
                     + p.delta_F     * state["F30"][mask]
                     + p.delta_S    * state["S30"][mask]
                     + p.delta_n     * log1p_n[mask]
                     + p.delta_ord   * ord_f[mask])
        logit_churn = (p.gamma_RC
                       + p.delta_R   * state["R"][mask]
                       + p.delta_D   * neg_log_D[mask]
                       - p.delta_F   * state["F30"][mask]
                       - p.delta_S  * state["S30"][mask]
                       - p.delta_n   * log1p_n[mask]
                       - p.delta_ord * ord_f[mask])
        fire_rec   = rng.random(mask.sum()) < _sigmoid(logit_rec)
        fire_churn = (rng.random(mask.sum()) < _sigmoid(logit_churn)) & ~fire_rec
        new_lc[idx[fire_rec]]   = CASUAL
        new_lc[idx[fire_churn]] = CHURNED

    # ── Churned → New (reactivation) ──────────────────────────────────────────
    # Constant daily probability; makes the chain ergodic.
    # R reset on reactivation; D recovers naturally during the churned period.
    if params.p_reactivate > 0:
        mask = lc == CHURNED
        if mask.any():
            fires = rng.random(mask.sum()) < params.p_reactivate
            idx = np.where(mask)[0][fires]
            new_lc[idx] = NEW
            state["R"][idx] = 0

    state["lifecycle"] = new_lc


def generate_users_warm(
    params: DGPParams,
    rng: np.random.Generator,
    lc_dist: np.ndarray | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """
    Initialize the cohort at the near-steady-state lifecycle distribution
    instead of all-New, avoiding the transient warm-up wave.

    Parameters
    ----------
    lc_dist : probabilities for [New, Casual, Power, AtRisk, Churned].
              Defaults to the calibrated steady-state distribution.
    """
    n = params.n_users_initial
    latent, state = generate_users(params, rng, n)

    if lc_dist is None:
        # Calibrated steady-state fractions of all tracked users:
        # New=13%  Casual=34%  Power=14%  AtRisk=11%  Churned=29%
        lc_dist = np.array([0.13, 0.34, 0.14, 0.11, 0.29])
    lc_dist = np.asarray(lc_dist, dtype=float)
    lc_dist /= lc_dist.sum()

    lifecycles = rng.choice(N_STATES, size=n, p=lc_dist)
    state["lifecycle"] = lifecycles.astype(np.int8)

    # Per-state initialization of (n_orders, F30, R, D).
    # Format: (n_orders_mean, f30_mean, r_lo, r_hi, d_mu, d_std)
    calib = {
        CASUAL:  (47,  4,  0, 10, 0.92, 0.04),
        POWER:   (78, 12,  0,  3, 0.80, 0.04),
        AT_RISK: (34,  2, 15, 40, 0.98, 0.02),
        CHURNED: ( 9,  0, 30, 90, 0.99, 0.01),
    }
    for s, (n_ord_mu, f30_mu, r_lo, r_hi, d_mu, d_std) in calib.items():
        mask = lifecycles == s
        m = mask.sum()
        if m == 0:
            continue
        n_ord = rng.poisson(n_ord_mu, m).astype(np.int32)
        f30   = rng.poisson(f30_mu,   m).astype(np.int32)
        r     = rng.integers(r_lo, max(r_lo + 1, r_hi), m).astype(np.int32)
        d     = np.clip(rng.normal(d_mu, d_std, m), 0.01, 1.0)

        gb_per_order = latent["beta_spend"][mask]
        state["n_orders"][mask] = n_ord
        state["F30"][mask]      = f30
        state["R"][mask]        = r
        state["D"][mask]        = d
        state["S30"][mask]     = f30   * gb_per_order
        state["spend_total"][mask] = n_ord * gb_per_order

        n_full, remainder = np.divmod(f30, 30)
        orders_buf = np.tile(n_full[:, None], (1, 30))
        for j in range(m):
            if remainder[j] > 0:
                chosen = rng.choice(30, size=int(remainder[j]), replace=False)
                orders_buf[j, chosen] += 1
        state["_orders_buf"][mask] = orders_buf
        state["_spend_buf"][mask]     = orders_buf * gb_per_order[:, None]

    return latent, state
