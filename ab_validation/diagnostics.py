"""
Observed overlap, CBPS balance and weight-stability diagnostics.
"""
from __future__ import annotations

import warnings

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold


FEATURE_NAMES = np.array([
    "past delay rate (30d)", "past delay rate (7d)",
    "orders (30d)", "spend (30d)", "recency",
    "log lifetime orders",
    "distance", "quoted prep time", "tip",
    "day-of-week sine", "day-of-week cosine",
    "hour sine", "hour cosine",
])
PROPENSITY_BINS = np.array([
    0., .01, .02, .05, .10, .15, .20, .30, .40, .50,
    .60, .70, .80, .90, .95, .98, .99, 1.,
])


def standardized_differences(
    covariates: np.ndarray, delayed: np.ndarray,
    delayed_weights: np.ndarray | None = None,
    ontime_weights: np.ndarray | None = None,
) -> np.ndarray:
    """Weighted delayed-minus-on-time means / unweighted pooled SD."""
    delayed = np.asarray(delayed, dtype=bool)
    x = np.asarray(covariates, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    if len(x) != len(delayed) or not delayed.any() or delayed.all():
        raise ValueError("Need both delayed and on-time orders")
    sd = np.sqrt((x[delayed].var(axis=0) + x[~delayed].var(axis=0)) / 2)
    sd[sd == 0] = 1.0
    w1 = np.ones(delayed.sum()) if delayed_weights is None else np.asarray(delayed_weights)
    w0 = np.ones((~delayed).sum()) if ontime_weights is None else np.asarray(ontime_weights)
    return ((np.average(x[delayed], axis=0, weights=w1)
             - np.average(x[~delayed], axis=0, weights=w0)) / sd)


def effective_size(weights: np.ndarray) -> float:
    w = np.asarray(weights, dtype=float)
    return float(w.sum() ** 2 / np.square(w).sum())


def weight_summary(weights: np.ndarray, outcomes: np.ndarray) -> np.ndarray:
    """ESS, median, p95, p99, max, top-1% weight and |w*Y| mass shares."""
    w = np.asarray(weights, dtype=float)
    y = np.asarray(outcomes, dtype=float)
    if w.ndim != 1 or len(w) != len(y) or not np.all(np.isfinite(w)):
        raise ValueError("Weights and outcomes must be finite 1d arrays of equal length")
    k = max(1, int(np.ceil(len(w) * .01)))
    top_w = np.argpartition(w, -k)[-k:]
    contribution = np.abs(w * y)
    top_contrib = np.argpartition(contribution, -k)[-k:]
    return np.array([
        effective_size(w), *np.quantile(w, [.5, .95, .99]), w.max(),
        w[top_w].sum() / w.sum(),
        contribution[top_contrib].sum() / max(contribution.sum(), 1e-12),
    ], dtype=float)


def cross_fitted_overlap(
    standardized_covariates: np.ndarray, delayed: np.ndarray,
    users: np.ndarray, max_train: int = 150_000,
) -> dict[str, np.ndarray]:
    """Out-of-eater-fold logistic delay score"""
    t = np.asarray(delayed, dtype=np.int32)
    groups = np.asarray(users)
    score = np.empty(len(t), dtype=float)
    training_rng = np.random.default_rng(1943)
    for train, test in GroupKFold(n_splits=3).split(
        standardized_covariates, t, groups=groups
    ):
        # The diagnostic is separate from estimation. Fit on a reproducible,
        # randomly sampled training subset at full-paper scale; predict every
        # held-out order (never the training eaters).
        if len(train) > max_train:
            train = training_rng.choice(train, max_train, replace=False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clf = LogisticRegression(max_iter=200, solver="lbfgs")
            clf.fit(standardized_covariates[train], t[train])
        score[test] = clf.predict_proba(standardized_covariates[test])[:, 1]
    bins = PROPENSITY_BINS
    hist = np.stack([np.histogram(score[t == arm], bins)[0]
                     for arm in (0, 1)])
    pct = np.stack([np.quantile(score[t == arm], [.01, .05, .5, .95, .99])
                    for arm in (0, 1)])
    tails = np.array([
        np.mean(score[t == 0] < .05), np.mean(score[t == 1] < .05),
        np.mean(score[t == 0] > .95), np.mean(score[t == 1] > .95),
    ])
    return dict(hist=hist, quantiles=pct, tails=tails, bins=bins)
