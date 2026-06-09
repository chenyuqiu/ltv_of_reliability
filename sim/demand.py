"""
Order placement model — Stage 1 (daily count) and Stage 2 (per-order characteristics).

draw_daily_orders() is called once per simulation day for all users.

Stage 1 — Daily order count
    n_i(t) ~ Poisson(λ_i(t))
    λ_i(t) = β_y_i · D_i(t) · m_{s_t} · κ_t

    β_y_i   — baseline order rate (latent per-user)
    D_i(t)  — delay memory ∈ (0,1]; 1 = no suppression
    m_{s_t} — lifecycle state multiplier (params.state_multipliers)
    κ_t     — daily macro shock ~ LogNormal(0, σ_demand)

    E_match (expected queue wait) enters only the ETA formula, not demand,
    because the demand rate does not include a congestion deterrence term.

Stage 2 — Per-order characteristics
    order_time ~ mixture(lunch, dinner, late-night, background, overnight)
    distance   ~ LogNormal(log μ_dist_i, σ_y_i)   [km]
    prep_time  ~ LogNormal(log μ_prep_i, σ_y_i)   [min]
    spend      ~ LogNormal(log β_spend_i, σ_y_i)  [$]
    tip        ~ LogNormal(log β_tip_i,   σ_y_i)  [$]
    ETA        = τ_base + τ_dist·distance + prep_time + E_match + η_buffer  [min]

Returns
-------
n_orders_today : int array   (n_users,)
spend_today    : float array (n_users,)  per-user total spending today
orders         : dict of parallel arrays per order placed today
                 Keys: user_idx, order_time, distance, prep_time, prep_time_orig,
                       spend, tip, ETA  — all shape (total_orders,)
"""
from __future__ import annotations

import numpy as np

from .params import DGPParams


def draw_daily_orders(
    latent: dict,
    state: dict,
    params: DGPParams,
    rng: np.random.Generator,
    sim_day: int,
    expected_matching_time: float | np.ndarray = 0.0,
    delta_prep_per_user: np.ndarray | None = None,
    demand_shock_override: float | None = None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Draw all orders placed today for the entire user population.

    Parameters
    ----------
    expected_matching_time : scalar or (n_users,) array — expected queue wait (min).
                             Used in the ETA formula; does not affect demand rates.
    delta_prep_per_user    : (n_users,) proactive dispatch advance (min).
                             All orders enter the marketplace FIFO queue at
                             food-ready time = order_time + prep_time.
                             Control:   prep_time = realized kitchen time (food-ready dispatch)
                             Treatment: prep_time = max(kitchen_time − delta_prep, 0)
                             so the courier is dispatched delta_prep minutes before
                             food is ready and travels concurrently with kitchen prep.
                             The ETA formula uses the original kitchen_time for both
                             arms, so the promised delivery time is unchanged.
    demand_shock_override  : if set, use this value instead of drawing κ_t; the RNG
                             still advances one draw to keep streams consistent.
    """
    n = len(latent["beta_y"])

    # ── Stage 1: Daily order count ────────────────────────────────────────────
    if demand_shock_override is not None:
        kappa_t = float(demand_shock_override)
        rng.lognormal(0.0, params.sigma_demand)  # keep RNG stream consistent
    else:
        kappa_t = rng.lognormal(0.0, params.sigma_demand)

    multipliers    = params.state_multipliers[state["lifecycle"]]
    lam            = latent["beta_y"] * state["D"] * multipliers * kappa_t
    n_orders_today = rng.poisson(lam).astype(np.int32)

    # ── Stage 2: Per-order characteristics ───────────────────────────────────
    total_orders = int(n_orders_today.sum())
    if total_orders == 0:
        empty = np.empty(0, dtype=np.float64)
        return n_orders_today, np.zeros(n, dtype=np.float64), {
            "user_idx": np.empty(0, dtype=np.int32), "order_time": empty.copy(),
            "distance": empty.copy(), "prep_time": empty.copy(),
            "prep_time_orig": empty.copy(), "spend": empty.copy(),
            "tip": empty.copy(), "ETA": empty.copy(),
        }

    user_idx   = np.repeat(np.arange(n, dtype=np.int32), n_orders_today)
    mu_dist    = latent["mu_dist"][user_idx]
    mu_prep    = latent["mu_prep"][user_idx]
    beta_spend = latent["beta_spend"][user_idx]
    beta_tip   = latent["beta_tip"][user_idx]
    sigma_y    = latent["sigma_y"][user_idx]

    distances  = rng.lognormal(np.log(mu_dist), sigma_y)
    prep_times = rng.lognormal(np.log(mu_prep),  sigma_y)
    if delta_prep_per_user is not None:
        prep_times_eff = np.maximum(prep_times - delta_prep_per_user[user_idx], 0.0)
    else:
        prep_times_eff = prep_times

    spends = rng.lognormal(np.log(beta_spend), sigma_y)
    tips   = rng.lognormal(np.log(beta_tip),   sigma_y)
    order_times = _draw_order_times(total_orders, params, rng)

    if np.isscalar(expected_matching_time):
        exp_match: float | np.ndarray = float(expected_matching_time)
    else:
        exp_match = np.asarray(expected_matching_time)[user_idx]

    ETAs = params.tau_base + params.tau_dist * distances + prep_times + exp_match + params.eta_buffer

    spend_today = np.zeros(n, dtype=np.float64)
    np.add.at(spend_today, user_idx, spends)

    orders: dict[str, np.ndarray] = {
        "user_idx":       user_idx,
        "order_time":     order_times,
        "distance":       distances,
        "prep_time":      prep_times_eff,
        "prep_time_orig": prep_times,
        "spend":          spends,
        "tip":            tips,
        "ETA":            ETAs,
    }
    return n_orders_today, spend_today, orders


def _draw_order_times(
    n_orders: int, params: DGPParams, rng: np.random.Generator
) -> np.ndarray:
    """
    Draw n_orders placement times (minutes from midnight) from a 5-component
    mixture: Gaussian lunch (12:00), Gaussian dinner (19:00), Gaussian late-night
    (22:30), Uniform background (07:00–23:00), Uniform overnight (00:00–06:00).
    """
    component = rng.choice(len(params.tod_weights), size=n_orders, p=params.tod_weights)
    times = np.empty(n_orders, dtype=np.float64)

    gaussian_params = [
        (params.tod_lunch_mean,     params.tod_lunch_std),
        (params.tod_dinner_mean,    params.tod_dinner_std),
        (params.tod_latenight_mean, params.tod_latenight_std),
    ]
    for i, (mu, sigma) in enumerate(gaussian_params):
        mask = component == i
        if mask.any():
            times[mask] = rng.normal(mu, sigma, mask.sum())

    mask = component == 3
    if mask.any():
        times[mask] = rng.uniform(params.tod_bg_lo, params.tod_bg_hi, mask.sum())

    mask = component == 4
    if mask.any():
        times[mask] = rng.uniform(params.tod_overnight_lo, params.tod_overnight_hi, mask.sum())

    return np.clip(times, 0.0, 1439.0)
