"""
Marketplace inner loop — courier supply, FIFO queue matching, delay assignment.

run_marketplace() processes one simulation day through a 5-minute slot inner loop.

Architecture
------------
Per day:
  1. Draw daily supply shock: ξ_t ~ LogNormal(0, σ_supply)
  2. Draw per-order travel and kitchen noise
  3. Index all orders in a FIFO queue by food-ready time
     (= order placement time + realized preparation time); run 288 slot iterations

Per slot (5 min):
  1. Insert orders whose food is ready (queue-entry time < slot end)
  2. Expire orders waiting > max_order_wait_min → unfulfilled (no spend, no delay)
  3. Expire couriers waiting > max_courier_wait_min → leave
  4. Draw new courier arrivals: NegBin(r, r/(r+λ)) or Poisson(λ)
  5. Match: each courier (FIFO) scans top-k orders;
       p_accept = σ(α_c − γ_dist·d + γ_tip·tip − peak_adj)
     First accepted order is taken; courier leaves. Unmatched couriers stay.

Persistent courier pool
-----------------------
Couriers accumulate across slots and retry until matched or timed out. This
produces realistic spillover: slack periods build courier inventory; demand
surges grow the queue. Matched couriers do not return within the same day.

Delay assignment
----------------
  ETA (promised) = τ_base + τ_dist·d + prep_time + E_match + η_buffer
  actual         = τ_base + τ_dist·d·speed_k + max(prep_orig·kitchen_k, prep_eff + match_time)
  delayed        = 1[actual > ETA + delay_buffer_min]

  The max(·) term models concurrent dispatch: when prep_eff < prep_orig (predictive
  dispatch), courier travel and kitchen prep overlap, reducing actual duration.

Switchback treatment
--------------------
Pass hourly_multipliers as a (24,) array to scale courier supply per hour.
Treated blocks receive SUPPLY_MULT×; control blocks and all non-switchback
phases use the default (None → all-ones).
"""
from __future__ import annotations

import numpy as np

from .params import DGPParams

N_SLOTS  = 288   # 1440 min / 5 min
SLOT_DUR = 5.0   # minutes per slot


def run_marketplace(
    orders: dict,
    n_users: int,
    params: DGPParams,
    rng: np.random.Generator,
    sim_day: int,
    supply_shock_override: float | None = None,
    hourly_multipliers: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Run the 5-minute marketplace inner loop for one simulation day.

    Parameters
    ----------
    orders                : dict from demand.draw_daily_orders
    n_users               : current total tracked users (for output array sizing)
    supply_shock_override : if set, use this value instead of drawing ξ_t;
                            the RNG still advances to keep streams consistent.
    hourly_multipliers    : (24,) per-hour courier-rate multipliers.
                            None → all hours use 1.0 (no modification).

    Returns
    -------
    delayed_mask  : bool array (n_users,)
    realized_spend: float array (n_users,)
    stats         : dict of market-level and per-order metrics
    """
    total = int(len(orders["user_idx"]))
    if total == 0:
        return (
            np.zeros(n_users, dtype=np.bool_),
            np.zeros(n_users, dtype=np.float64),
            _empty_stats(n_users),
        )

    dow = sim_day % 7

    # ── Daily supply shock ────────────────────────────────────────────────────
    if supply_shock_override is not None:
        supply_shock = supply_shock_override
        _ = rng.lognormal(0.0, params.sigma_supply)  # keep RNG stream consistent
    else:
        supply_shock = rng.lognormal(0.0, params.sigma_supply)

    # ── Per-order delivery noise ──────────────────────────────────────────────
    speed_k   = rng.lognormal(0.0, params.sigma_travel,  total)
    kitchen_k = rng.lognormal(0.0, params.sigma_kitchen, total)

    # ── FIFO queue indexed by food-ready time ────────────────────────────────
    # Queue-entry time = order_time + prep_time. In the control arm, prep_time
    # is the full realized kitchen time, so the courier is dispatched when food
    # is ready. In the AB treatment arm, delta_prep_per_user reduces prep_time
    # so the courier is dispatched early and travels concurrently with kitchen
    # preparation. The actual delivery formula uses prep_time_orig (the true
    # kitchen time) to correctly credit or charge for any courier-wait at pickup.
    o_time_raw  = orders["order_time"]
    o_prep_raw  = orders["prep_time"]
    o_qtime_raw = o_time_raw + o_prep_raw

    sidx      = np.argsort(o_qtime_raw)
    _inv_sidx = np.argsort(sidx)
    o_qtime   = o_qtime_raw[sidx]
    o_time    = o_time_raw[sidx]
    o_dist    = orders["distance"][sidx]
    o_prep    = orders["prep_time"][sidx]
    o_prep_orig = orders.get("prep_time_orig", orders["prep_time"])[sidx]
    o_spend   = orders["spend"][sidx]
    o_tip     = orders["tip"][sidx]
    o_eta     = orders["ETA"][sidx]
    o_user    = orders["user_idx"][sidx]

    base_actual   = params.tau_base + params.tau_dist * o_dist * speed_k
    order_contrib = -params.gamma_dist * o_dist + params.gamma_tip * o_tip

    match_time = np.full(total, np.nan, dtype=np.float64)

    scale = n_users / params.courier_scale_base

    # ── 5-minute slot inner loop ──────────────────────────────────────────────
    queue: list[int] = []
    ptr = 0
    courier_pool: list[list[float]] = []

    # Per-slot diagnostic accumulators (time-of-day / day-of-week patterns)
    orders_in_by_slot      = np.zeros(N_SLOTS, dtype=np.int32)
    couriers_by_slot       = np.zeros(N_SLOTS, dtype=np.int32)
    matched_by_slot        = np.zeros(N_SLOTS, dtype=np.int32)
    match_time_sum_by_slot = np.zeros(N_SLOTS, dtype=np.float64)

    for slot in range(N_SLOTS):
        t_start = slot * SLOT_DUR
        t_end   = t_start + SLOT_DUR

        while ptr < total and o_qtime[ptr] < t_end:
            queue.append(ptr)
            ptr += 1
            orders_in_by_slot[slot] += 1

        while queue and (t_start - o_qtime[queue[0]]) >= params.max_order_wait_min:
            queue.pop(0)
        courier_pool = [c for c in courier_pool
                        if (t_start - c[0]) < params.max_courier_wait_min]

        hour     = min(int(t_start) // 60, 23)
        slot_mult = hourly_multipliers[hour] if hourly_multipliers is not None else 1.0
        lam = params.courier_rate_table[dow, hour] * supply_shock * scale * slot_mult
        if params.courier_overdispersion > 0 and lam > 0:
            r     = 1.0 / params.courier_overdispersion
            n_new = int(rng.negative_binomial(r, r / (r + lam)))
        else:
            n_new = int(rng.poisson(lam))
        couriers_by_slot[slot] = n_new
        if n_new > 0:
            new_alphas = rng.normal(params.mu_courier, params.sigma_courier, n_new)
            for alpha in new_alphas:
                courier_pool.append([t_start, float(alpha)])

        if not queue or not courier_pool:
            continue

        is_peak  = (11 <= hour <= 13) or (18 <= hour <= 21)
        peak_adj = params.courier_peak_penalty if is_peak else 0.0
        still_waiting: list[list[float]] = []
        for c in courier_pool:
            if not queue:
                still_waiting.append(c)
                continue
            alpha   = c[1]
            k_show  = min(params.courier_show_k, len(queue))
            matched = False
            for ki in range(k_show):
                oi    = queue[ki]
                logit = alpha + order_contrib[oi] - peak_adj
                if rng.random() < 1.0 / (1.0 + np.exp(-logit)):
                    queue.pop(ki)
                    match_time[oi] = t_end - o_qtime[oi]
                    matched_by_slot[slot] += 1
                    match_time_sum_by_slot[slot] += match_time[oi]
                    matched = True
                    break
            if not matched:
                still_waiting.append(c)
        courier_pool = still_waiting

    # ── Delay assignment ──────────────────────────────────────────────────────
    matched     = ~np.isnan(match_time)
    _match_safe = np.where(matched, match_time, 0.0)
    actual_dur  = base_actual + np.maximum(o_prep_orig * kitchen_k, o_prep + _match_safe)
    delayed_ord = matched & (actual_dur > o_eta + params.delay_buffer_min)

    # ── Aggregate to per-user arrays ──────────────────────────────────────────
    delayed_mask   = np.zeros(n_users, dtype=np.bool_)
    realized_spend = np.zeros(n_users, dtype=np.float64)
    matched_per_user = np.zeros(n_users, dtype=np.int32)
    delayed_per_user = np.zeros(n_users, dtype=np.int32)

    np.add.at(realized_spend,  o_user[matched],      o_spend[matched])
    delayed_mask[o_user[delayed_ord]] = True
    np.add.at(matched_per_user, o_user[matched],     1)
    np.add.at(delayed_per_user, o_user[delayed_ord], 1)

    # ── Stats ─────────────────────────────────────────────────────────────────
    n_matched  = int(matched.sum())
    n_delayed  = int(delayed_ord.sum())
    avg_match  = float(match_time[matched].mean()) if n_matched > 0 else 0.0
    delay_rate = n_delayed / n_matched if n_matched > 0 else 0.0

    stats = {
        "n_matched":         n_matched,
        "avg_match_time":    avg_match,
        "delay_rate":        delay_rate,
        "matched_per_user":  matched_per_user,
        "delayed_per_user":  delayed_per_user,
        "matched_per_order": matched[_inv_sidx],
        "delayed_per_order": delayed_ord[_inv_sidx],
        "order_user_idx":    o_user[_inv_sidx],
        "order_distance":    o_dist[_inv_sidx],
        "order_prep_time":   o_prep[_inv_sidx],
        "order_tip":         o_tip[_inv_sidx],
        "order_spend":       o_spend[_inv_sidx],
        "order_hour":        (o_time_raw // 60).astype(np.int32),
        "orders_in_by_slot":      orders_in_by_slot,
        "couriers_by_slot":       couriers_by_slot,
        "matched_by_slot":        matched_by_slot,
        "match_time_sum_by_slot": match_time_sum_by_slot,
    }

    return delayed_mask, realized_spend, stats


def _empty_stats(n_users: int = 0) -> dict:
    return {
        "n_matched":         0,
        "avg_match_time":    0.0,
        "delay_rate":        0.0,
        "matched_per_user":  np.zeros(n_users, dtype=np.int32),
        "delayed_per_user":  np.zeros(n_users, dtype=np.int32),
        "matched_per_order": np.zeros(0, dtype=np.bool_),
        "delayed_per_order": np.zeros(0, dtype=np.bool_),
        "order_user_idx":    np.zeros(0, dtype=np.int32),
        "order_distance":    np.zeros(0, dtype=np.float64),
        "order_prep_time":   np.zeros(0, dtype=np.float64),
        "order_tip":         np.zeros(0, dtype=np.float64),
        "order_spend":       np.zeros(0, dtype=np.float64),
        "order_hour":        np.zeros(0, dtype=np.int32),
        "orders_in_by_slot":      np.zeros(N_SLOTS, dtype=np.int32),
        "couriers_by_slot":       np.zeros(N_SLOTS, dtype=np.int32),
        "matched_by_slot":        np.zeros(N_SLOTS, dtype=np.int32),
        "match_time_sum_by_slot": np.zeros(N_SLOTS, dtype=np.float64),
    }
