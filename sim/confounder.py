"""Optional iid bad-service shocks and their finite-duration demand effect.

The daily shock is independent over eaters and days. Only its effect on demand
persists, through a rolling buffer of the previous M matched bad experiences.
"""
from __future__ import annotations

import numpy as np

from .params import DGPParams


def draw_bad_service(
    orders: dict,
    n_users: int,
    params: DGPParams,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw the day's hidden shock and attach it to orders before fulfillment."""
    if not 0.0 <= params.quality_bad_prob <= 1.0:
        raise ValueError("quality_bad_prob must be in [0, 1]")
    if params.quality_bad_prob == 0.0:
        bad = np.zeros(n_users, dtype=np.int8)
    else:
        bad = (rng.random(n_users) < params.quality_bad_prob).astype(np.int8)
    orders["hidden_bad_quality"] = bad[orders["user_idx"]]
    return bad


def record_bad_service(
    state: dict,
    matched_per_user: np.ndarray,
    bad: np.ndarray,
    sim_day: int,
) -> None:
    """Record matched bad experiences; effect applies on days t+1 through t+M."""
    history = state["_quality_history"]
    if history.shape[1] < 1:
        raise ValueError("quality_effect_days must be positive")
    history[:, sim_day % history.shape[1]] = (
        (matched_per_user > 0) & (bad > 0)
    ).astype(np.int8)
