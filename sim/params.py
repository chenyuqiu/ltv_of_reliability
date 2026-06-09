"""
Data-generating process parameters for the delivery platform simulator.
Parameters are grouped by the module that reads them.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def _make_user_corr() -> np.ndarray:
    """
    Correlation matrix for the 7 latent user parameters.
    Order: [β_y=0, β_d=1, β_spend=2, β_tip=3, σ_y=4, μ_dist=5, μ_prep=6]
    Verified positive-definite (min eigenvalue ≈ 0.22).

    Pair              Value   Rationale
    ──────────────────────────────────────────────────────────────────
    β_y  ↔ β_d       -0.30   Frequent orderers are less delay-sensitive
    β_spend ↔ β_d       -0.40   High-value users are less delay-sensitive
    β_y  ↔ σ_y       +0.30   Frequent orderers explore more variety
    β_spend ↔ σ_y       -0.30   High-GB users are loyal to known restaurants
    β_spend ↔ β_tip     +0.50   Larger baskets produce larger absolute tips
    μ_dist ↔ β_tip   +0.30   Far-distance orders receive larger tips
    μ_prep ↔ β_tip   +0.25   Slow-restaurant users tip more generously
    β_y  ↔ β_spend      +0.35   High-frequency and high-spend co-occur
    β_y  ↔ β_tip     +0.20   Frequent users tip more habitually
    β_d  ↔ β_tip     -0.15   Delay-sensitive users tip slightly less
    β_d  ↔ μ_dist    -0.20   Delay-sensitive users self-select shorter distances
    β_spend ↔ μ_dist    +0.20   High-value users occasionally order from farther restaurants
    β_spend ↔ μ_prep    +0.30   High-value users favor higher-tier, slower restaurants
    σ_y  ↔ μ_dist    +0.30   Exploratory users venture farther from home
    σ_y  ↔ μ_prep    +0.15   Exploratory users sample varied restaurant tiers
    μ_dist ↔ μ_prep  +0.40   Geographically distant restaurants tend to be higher-tier
    ──────────────────────────────────────────────────────────────────
    All other pairs: 0.00
    """
    c = np.eye(7)
    c[0, 1] = c[1, 0] = -0.30   # β_y ↔ β_d
    c[1, 2] = c[2, 1] = -0.40   # β_spend ↔ β_d
    c[0, 4] = c[4, 0] = +0.30   # β_y ↔ σ_y
    c[2, 4] = c[4, 2] = -0.30   # β_spend ↔ σ_y
    c[2, 3] = c[3, 2] = +0.50   # β_spend ↔ β_tip
    c[3, 5] = c[5, 3] = +0.30   # μ_dist ↔ β_tip
    c[3, 6] = c[6, 3] = +0.25   # μ_prep ↔ β_tip
    c[0, 2] = c[2, 0] = +0.35   # β_y ↔ β_spend
    c[0, 3] = c[3, 0] = +0.20   # β_y ↔ β_tip
    c[1, 3] = c[3, 1] = -0.15   # β_d ↔ β_tip
    c[1, 5] = c[5, 1] = -0.20   # β_d ↔ μ_dist
    c[2, 5] = c[5, 2] = +0.20   # β_spend ↔ μ_dist
    c[2, 6] = c[6, 2] = +0.30   # β_spend ↔ μ_prep
    c[4, 5] = c[5, 4] = +0.30   # σ_y ↔ μ_dist
    c[4, 6] = c[6, 4] = +0.15   # σ_y ↔ μ_prep
    c[5, 6] = c[6, 5] = +0.40   # μ_dist ↔ μ_prep
    assert np.all(np.linalg.eigvalsh(c) > 0), "user_corr is not positive definite"
    return c


_DEFAULT_USER_CORR = _make_user_corr()


def _make_courier_rate_table() -> np.ndarray:
    """
    Mean couriers per 5-minute slot by (day_of_week, hour_of_day). Shape: (7, 24).
    Row 0 = Monday, row 6 = Sunday. Column = hour of day (0 = 00:00–00:59).
    """
    base = np.array([
        0.20, 0.12, 0.09, 0.09, 0.09, 0.18,   # 00–05: overnight
        0.36, 0.67, 0.89, 0.89, 1.11, 1.65,   # 06–11: morning build
        1.56, 1.33, 1.11, 0.89, 1.11, 1.33,   # 12–17: lunch peak + afternoon
        1.78, 2.00, 1.56, 1.11, 0.95, 0.36,   # 18–23: dinner peak + late decline
    ], dtype=np.float64)
    dow_mult = np.array([1.00, 1.00, 1.00, 1.00, 1.10, 1.25, 1.15])
    return dow_mult[:, None] * base[None, :]  # (7, 24)


_DEFAULT_COURIER_RATE = _make_courier_rate_table()


@dataclass
class DGPParams:
    # ── Simulation scope ───────────────────────────────────────────────────────
    n_users_initial: int = 1_000
    seed: int = 42

    # ── users.py: User population (joint LogNormal) ───────────────────────────
    # Latent params drawn as z ~ N(mu_pop, Sigma_pop), param = exp(z).
    # Order: [β_y, β_d, β_spend, β_tip, σ_y, μ_dist, μ_prep]

    # β_y: baseline log-rate of ordering (enters log λ_i directly)
    log_beta_y_mean: float = -0.70   # exp(-0.70) ≈ 0.50 orders/day at Power state
    log_beta_y_std:  float =  0.40

    # β_d: delay sensitivity (governs D(t) decay and lifecycle transitions)
    log_beta_d_mean: float = -2.50   # exp(-2.50) ≈ 0.082
    log_beta_d_std:  float =  0.50

    # β_spend: log-mean of per-order user spending ($)
    log_beta_spend_mean: float = 3.00   # exp(3) ≈ $20 typical order
    log_beta_spend_std:  float = 0.40

    # β_tip: log-mean of per-order tip ($)
    log_beta_tip_mean: float = 0.70  # exp(0.70) ≈ $2 typical tip
    log_beta_tip_std:  float = 0.40

    # σ_y: explorativeness; controls per-order characteristic variance
    log_sigma_y_mean: float = -0.70  # exp(-0.70) ≈ 0.50
    log_sigma_y_std:  float =  0.30

    # μ_dist: log-mean of order distance (km)
    log_mu_dist_mean: float = 1.60   # exp(1.60) ≈ 5 km
    log_mu_dist_std:  float = 0.40

    # μ_prep: log-mean of restaurant prep time (min)
    log_mu_prep_mean: float = 2.90   # exp(2.90) ≈ 18 min
    log_mu_prep_std:  float = 0.30

    # 7×7 positive-definite correlation matrix (see _make_user_corr).
    user_corr: np.ndarray = field(default_factory=lambda: _DEFAULT_USER_CORR.copy())

    # ── users.py: Delay memory ────────────────────────────────────────────────
    # D(t) update: log D(t+1) = exp(-λ_mem)·log D(t) − β_d·1[delayed]
    lambda_mem: float = 0.08   # decay rate; half-life ≈ ln2/0.08 ≈ 8.7 days

    # ── users.py: New-user arrivals ───────────────────────────────────────────
    # n_new ~ Poisson(lambda_growth · n_active) each day. Proportional arrivals
    # keep the New-state share self-stabilizing as the platform grows.
    lambda_growth: float = 0.001   # per active user per day

    # ── users.py: Reactivation ────────────────────────────────────────────────
    # Constant daily probability that a Churned user re-enters as New.
    # Makes the lifecycle Markov chain ergodic → finite mixing time.
    # R is reset on reactivation; D recovers naturally via lambda_mem decay.
    p_reactivate: float = 0.01   # 1%/day ≈ avg 100 days before reactivation

    # ── users.py: Lifecycle state multipliers ─────────────────────────────────
    # Multiplier on baseline order rate λ_i(t) by lifecycle state.
    # Index: [New=0, Casual=1, Power=2, AtRisk=3, Churned=4]
    state_multipliers: np.ndarray = field(
        default_factory=lambda: np.array([0.3, 0.6, 1.0, 0.2, 0.0])
    )

    # ── users.py: Lifecycle transition parameters ─────────────────────────────
    # Each transition: logit(p) = γ + Σ ±δ·covariate (see users.py for equations).
    # Coefficients (δ) are all positive; sign is applied per-transition.
    #
    # Structural constraints (hard-coded in users.py, not here):
    #   1. log1p(n_orders) is capped at log1p(30): prevents unbounded tenure signal
    #      from locking long-tenured Casual users into Power indefinitely.
    #   2. delta_n excluded from Power→AtRisk: lifetime orders should not protect
    #      Power users from current-period disengagement; only F30 and R govern it.

    # Intercepts (more negative = rarer without supporting signal)
    gamma_NC:  float = -3.00   # New → Casual
    gamma_NA:  float = -3.50   # New → AtRisk
    gamma_CP:  float = -5.50   # Casual → Power
    gamma_CA:  float = -2.50   # Casual → AtRisk
    gamma_PA:  float = -3.50   # Power → AtRisk
    gamma_PC:  float = -2.00   # Power → Casual
    gamma_RAC: float = -1.00   # AtRisk → Casual (recovery)
    gamma_RC:  float = -7.00   # AtRisk → Churned

    # Shared coefficients
    delta_n:     float = 0.50    # log(1 + n_orders): habit depth
    delta_ord:   float = 0.80    # ordered_today: daily engagement signal
    delta_F:     float = 0.05    # F30: 30-day order frequency
    delta_S:     float = 0.002   # S30: 30-day user spending ($)
    delta_R:     float = 0.05    # R: days since last order
    delta_D:     float = 15.0    # −log D: delay-memory pressure
    delta_D_rec: float = 10.0    # log D in AtRisk→Casual only (decoupled from delta_D)

    # Hard gate: Casual→Power requires F30 ≥ power_min_F30 AND ordered_today.
    # Ensures the Power state reflects sustained current frequency, not just tenure.
    power_min_F30: int = 6

    # ── demand.py ─────────────────────────────────────────────────────────────
    # Daily demand shock: κ_t ~ LogNormal(0, sigma_demand), drawn once per day,
    # applied uniformly to all users. E[κ_t] = exp(σ²/2) ≈ 1.01.
    sigma_demand: float = 0.15

    # ETA = tau_base + tau_dist·distance + prep_time + E_match + eta_buffer
    # eta_buffer pads the promise to the ~80th-percentile delivery time so that
    # near-miss lateness (< buffer) is not classified as a delay.
    tau_base:   float = 5.0    # fixed platform overhead (min)
    tau_dist:   float = 3.0    # per-km travel coefficient (min/km)
    eta_buffer: float = 5.0    # ETA padding above median (min)

    # Time-of-day order placement: 5-component mixture (minutes from midnight).
    # Components: [lunch, dinner, late-night, background, overnight]
    tod_weights: np.ndarray = field(
        default_factory=lambda: np.array([0.34, 0.44, 0.10, 0.08, 0.04])
    )
    tod_lunch_mean:     float = 720.0    # 12:00
    tod_lunch_std:      float = 45.0
    tod_dinner_mean:    float = 1140.0   # 19:00
    tod_dinner_std:     float = 60.0
    tod_latenight_mean: float = 1350.0   # 22:30
    tod_latenight_std:  float = 30.0
    tod_bg_lo:          float = 420.0    # 07:00
    tod_bg_hi:          float = 1380.0   # 23:00
    tod_overnight_lo:   float = 0.0      # 00:00
    tod_overnight_hi:   float = 360.0    # 06:00

    # ── marketplace.py ────────────────────────────────────────────────────────
    # Daily supply shock: ξ_t ~ LogNormal(0, sigma_supply).
    # σ=0.10 → P(ξ < 0.85) ≈ 5%; IQR ≈ [0.93, 1.07].
    sigma_supply: float = 0.10

    # Courier arrival rate table: mean couriers per 5-min slot by (dow, hour).
    # Scaled by n_users / courier_scale_base in marketplace.py.
    courier_rate_table: np.ndarray = field(
        default_factory=lambda: _DEFAULT_COURIER_RATE.copy()
    )
    courier_scale_base: int = 700

    # Courier arrival overdispersion: NegBin replaces Poisson when > 0.
    # overdispersion = 1/r; at large n, slot-level CV ≈ sqrt(overdispersion).
    courier_overdispersion: float = 0.12

    # Courier acceptance model:
    #   p_accept = σ(α_c − γ_dist·distance + γ_tip·tip − peak_adj)
    # α_c ~ N(mu_courier, sigma_courier) drawn per arriving courier.
    # peak_adj = courier_peak_penalty during lunch (11–13h) and dinner (18–21h).
    mu_courier:           float = 0.50
    sigma_courier:        float = 0.50
    gamma_dist:           float = 0.10   # logit penalty per km
    gamma_tip:            float = 0.30   # logit bonus per $ of tip
    courier_show_k:       int   = 4      # orders shown per courier (FIFO top-k)
    courier_peak_penalty: float = 0.80   # extra logit reduction at peak hours

    # Couriers wait up to max_courier_wait_min without a match before leaving.
    max_courier_wait_min: float = 30.0

    # Orders abandoned after max_order_wait_min in queue (no GB, no delay).
    max_order_wait_min: float = 80.0

    # Delay classification: actual > ETA + delay_buffer_min → delayed.
    # Avoids classifying near-misses as delays, consistent with platform SLA rules.
    delay_buffer_min: float = 20.0

    # Delivery noise: actual durations drawn with log-normal multipliers.
    #   actual_travel = tau_dist · distance · speed_noise,  speed_noise ~ LN(0, σ)
    #   actual_prep   = prep_time · kitchen_noise,          kitchen_noise ~ LN(0, σ)
    sigma_travel:  float = 0.35
    sigma_kitchen: float = 0.40
