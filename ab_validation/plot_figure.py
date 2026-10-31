"""
AB Validation Plots
===================
Reads the single main-market AB cache and renders paper Figures 1–6 and 8.
Usage (from repository root):
    python ab_validation/plot_figure.py

Input:  ab_validation/cache/ab_validation_cache.npz
Output: figures/figure{1,2,3,4,5,6,8}_*.png
"""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator
from scipy.integrate import trapezoid
from scipy.ndimage import gaussian_filter1d

# ── Paths ─────────────────────────────────────────────────────────────────────
CACHE_PATH = Path(__file__).parent / "cache" / "ab_validation_cache.npz"
OUT_DIR    = ROOT / "figures"

# ── Shared style ──────────────────────────────────────────────────────────────
STATE_NAMES  = ["New", "Casual", "Power", "AtRisk", "Churned"]
STATE_COLORS = ["#4C9BE8", "#5DBB63", "#F4A623", "#E05C5C", "#AAAAAA"]
N_STATES     = 5

# Figure 5 (Chronos) style
K_COLORS = {28: "#E67E22", 42: "#9B59B6", 56: "#2980B9", 70: "#E91E63"}
K_SOLID  = 56   # K rendered as thick solid; others dashed

# ── Load cache ────────────────────────────────────────────────────────────────
if not CACHE_PATH.exists():
    raise FileNotFoundError(
        f"Cache not found: {CACHE_PATH}\n"
        "Run ab_validation/run_simulation.py first."
    )
z = np.load(str(CACHE_PATH))

# Chronos (paper Figure 5)
K_SWEEP        = [int(k) for k in z["K_SWEEP"]]
betas          = {int(k): float(b) for k, b in zip(z["K_SWEEP"], z["betas"])}
weekly_del_t   = z["weekly_del_t"]; weekly_mat_t = z["weekly_mat_t"]
weekly_del_c   = z["weekly_del_c"]; weekly_mat_c = z["weekly_mat_c"]
wk_gb_act      = z["wk_gb_act"]
D_act          = z["D_act"].astype(float)
pre_act        = z["pre_act"].astype(np.float64)
n_act          = int(z["n_act"])
N_WEEKS        = int(z["N_WEEKS"])

# Obs-phase diagnostics (paper Figures 1, 2, 3, 8)
trans_ontime    = z["trans_ontime"]
trans_delay     = z["trans_delay"]
trans_organic   = z["trans_organic"]
lc_final        = z["lc_final"].astype(int)
spend_final     = z["spend_final"].astype(np.float64)
matched_cum     = z["matched_cum"]
delayed_cum     = z["delayed_cum"]
lc_share_daily  = z["lc_share_daily"]
dow_hr_orders   = z["dow_hr_orders"]
dow_hr_couriers = z["dow_hr_couriers"]
dow_hr_matched  = z["dow_hr_matched"]
dow_hr_match_t  = z["dow_hr_match_t"]
dow_hr_count    = z["dow_hr_count"]
N_OBS_DAYS      = int(z["N_OBS_DAYS"])

OUT_DIR.mkdir(parents=True, exist_ok=True)
print(f"Loaded cache: K_SWEEP={K_SWEEP}  n_act={n_act:,}  N_WEEKS={N_WEEKS}")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 1: Lifecycle transition probabilities (3-panel)
# ══════════════════════════════════════════════════════════════════════════════
def _row_norm(mat):
    return mat / mat.sum(axis=1, keepdims=True).clip(1)

def _plot_matrix(ax, prob_mat, title):
    im = ax.imshow(prob_mat, vmin=0, vmax=1, cmap="Blues", aspect="auto")
    for i in range(N_STATES):
        for j in range(N_STATES):
            v = prob_mat[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    fontsize=15, color="white" if v > 0.55 else "black")
    ax.set_xticks(range(N_STATES))
    ax.set_xticklabels(STATE_NAMES, fontsize=16)
    ax.set_yticks(range(N_STATES)); ax.set_yticklabels(STATE_NAMES, fontsize=16)
    ax.set_title(title, fontsize=19)
    return im

fig1, axes1 = plt.subplots(1, 3, figsize=(18, 5.9), sharey=True, layout="constrained")
im = _plot_matrix(axes1[0], _row_norm(trans_organic.astype(float)), "Did Not Order")
_plot_matrix(axes1[1], _row_norm(trans_ontime.astype(float)), "Ordered, On-Time")
_plot_matrix(axes1[2], _row_norm(trans_delay.astype(float)), "Ordered, Delayed")
fig1.supxlabel("State at end of day (t+1)", fontsize=17)
fig1.supylabel("State at start of day (t)", fontsize=17)
cb = fig1.colorbar(im, ax=axes1, shrink=0.75, pad=0.02)
cb.set_label("Transition probability", fontsize=16)
cb.ax.tick_params(labelsize=14)
fig1.suptitle("Daily Lifecycle Transition Probabilities", fontsize=21)
out1 = OUT_DIR / "figure1_lifecycle_transition.png"
fig1.savefig(str(out1), dpi=150, bbox_inches="tight"); plt.close(fig1)
print(f"Saved → {out1}")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 2: Eater-level outcomes by lifecycle state (day-250 snapshot)
# ══════════════════════════════════════════════════════════════════════════════
with np.errstate(invalid="ignore", divide="ignore"):
    ss_dr = np.where(matched_cum > 0, delayed_cum / matched_cum, np.nan)

dr_means, val_means = [], []
for s in range(N_STATES):
    mask_dr = (lc_final == s) & ~np.isnan(ss_dr)
    dr_s  = ss_dr[mask_dr] * 100
    val_s = spend_final[lc_final == s]
    dr_means.append(dr_s.mean()  if len(dr_s)  > 0 else 0.0)
    val_means.append(val_s.mean() if len(val_s) > 0 else 0.0)

fig2, (ax_dr, ax_val) = plt.subplots(1, 2, figsize=(14, 5))

ax_dr.bar(STATE_NAMES, dr_means, color=STATE_COLORS, alpha=0.85)
ax_dr.set_ylabel("Avg delay rate (%)", fontsize=15)
ax_dr.set_title("Delay Rate by Lifecycle State", fontsize=17)
ax_dr.tick_params(axis="both", labelsize=13); ax_dr.set_ylim(0, None)

ax_val.bar(STATE_NAMES, val_means, color=STATE_COLORS, alpha=0.85)
ax_val.set_ylabel("Avg lifetime value ($)", fontsize=15)
ax_val.set_title("Lifetime Value by Lifecycle State", fontsize=17)
ax_val.tick_params(axis="both", labelsize=13); ax_val.set_ylim(0, None)

fig2.suptitle("Delay Rate and Lifetime Value by Lifecycle State", fontsize=19)
fig2.tight_layout()
out2 = OUT_DIR / "figure2_outcomes_by_state.png"
fig2.savefig(str(out2), dpi=150, bbox_inches="tight"); plt.close(fig2)
print(f"Saved → {out2}")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 3: publication-style marketplace volume and congestion
# ══════════════════════════════════════════════════════════════════════════════
day_counts = np.asarray(dow_hr_count, float)
orders = np.asarray(dow_hr_orders, float)
couriers = np.asarray(dow_hr_couriers, float)
matched = np.asarray(dow_hr_matched, float)
match_minutes = np.asarray(dow_hr_match_t, float)
assert day_counts.shape == (7,) and (day_counts > 0).all()
assert all(a.shape == (7, 24) for a in (orders, couriers, matched, match_minutes))
tod_orders = orders.sum(0) / day_counts.sum()
tod_couriers = couriers.sum(0) / day_counts.sum()
with np.errstate(invalid="ignore", divide="ignore"):
    tod_match = np.where(matched.sum(0) > 0,
                         match_minutes.sum(0) / matched.sum(0), np.nan)
    dow_match = np.where(matched.sum(1) > 0,
                         match_minutes.sum(1) / matched.sum(1), np.nan)
dow_orders = orders.sum(1) / day_counts
dow_couriers = couriers.sum(1) / day_counts

original_style = plt.rcParams.copy()
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 17,
    "axes.labelsize": 18, "xtick.labelsize": 17, "ytick.labelsize": 17,
    "figure.facecolor": "white", "axes.facecolor": "white",
    "savefig.facecolor": "white",
})
fig3, (ax_tod, ax_dow) = plt.subplots(1, 2, figsize=(18.5, 7.0))
fig3.subplots_adjust(left=.07, right=.94, bottom=.14, top=.73, wspace=.39)
COLORS = {"orders": "#4C9BE8", "couriers": "#F4A623", "match": "#5DBB63"}
hr, dow = np.arange(24), np.arange(7)

tod_r = ax_tod.twinx()
l1, = ax_tod.plot(hr, tod_orders, color=COLORS["orders"], lw=2.5, marker="o", ms=5.5)
l2, = ax_tod.plot(hr, tod_couriers, color=COLORS["couriers"], lw=2.5, marker="s", ms=5.5)
l3, = tod_r.plot(hr, tod_match, color=COLORS["match"], lw=2.5,
                 ls="--", marker="^", ms=6)
ax_tod.set(xlabel="Hour of day", ylabel="Avg count per hour")
tod_r.set_ylabel("Match time (min)")
tod_r.set_ylim(0, None)
ax_tod.set_xticks(hr[::2])
ax_tod.set_title("Time-of-Day Marketplace Pattern", fontsize=20, pad=14)

dow_r = ax_dow.twinx()
ax_dow.plot(dow, dow_orders, color=COLORS["orders"], lw=2.5, marker="o", ms=7.5)
ax_dow.plot(dow, dow_couriers, color=COLORS["couriers"], lw=2.5, marker="s", ms=7.5)
dow_r.plot(dow, dow_match, color=COLORS["match"], lw=2.5, ls="--", marker="^", ms=8)
ax_dow.set(xlabel="Day of week", ylabel="Avg daily count")
dow_r.set_ylabel("Match time (min)")
dow_r.set_ylim(0, None)
ax_dow.set_xticks(dow, ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
ax_dow.set_title("Day-of-Week Marketplace Pattern", fontsize=20, pad=14)

fig3.suptitle("Marketplace Volume & Congestion Patterns", fontsize=22, y=.99)
fig3.legend([l1, l2, l3], ["Orders", "Couriers", "Match time (min)"],
            loc="upper center", bbox_to_anchor=(.5, .91), ncol=3,
            fontsize=17, frameon=False, columnspacing=2.2,
            handlelength=2.4, markerscale=1.3)
out3 = OUT_DIR / "figure3_marketplace_patterns.png"
fig3.savefig(out3, dpi=180, facecolor="white", bbox_inches="tight", pad_inches=.06)
plt.close(fig3)
plt.rcParams.update(original_style)
print(f"Saved → {out3}")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 4: observed balance and overlap in the main simulation market (K=56)
# ══════════════════════════════════════════════════════════════════════════════
features = list(map(str, z["diag_feature_names"]))
assert len(features) == 13 and "orders already placed today" not in features
lookup = {name: j for j, name in enumerate(features)}
raw = np.asarray(z["K56_diag_observed_smd_raw"], float)
weighted = np.asarray(z["K56_diag_observed_smd_weighted"], float)

def _most_imbalanced_time_component(prefix, label):
    """Display one cyclic component; both remain in CBPS and overlap checks."""
    names = (f"{prefix} sine", f"{prefix} cosine")
    chosen = max(names, key=lambda name: abs(raw[lookup[name]]))
    return chosen, f"{label} ({chosen.rsplit(' ', 1)[-1]})"

selected = [
    ("past delay rate (30d)", "Past delay rate (30d)"),
    ("spend (30d)", "Past spending (30d)"),
    ("distance", "Distance"),
    ("quoted prep time", "Restaurant prep time"),
    ("tip", "Tip"),
    _most_imbalanced_time_component("day-of-week", "Day of week"),
    _most_imbalanced_time_component("hour", "Hour of day"),
]
# Rank by pre-weighting imbalance
selected.sort(key=lambda item: abs(raw[lookup[item[0]]]), reverse=True)
before = np.array([abs(raw[lookup[name]]) for name, _ in selected])
after = np.array([abs(weighted[lookup[name]]) for name, _ in selected])
labels = [label for _, label in selected]
n_arm = np.asarray(z["K56_diag_n_by_arm"], int)
bins = np.asarray(z["diag_overlap_bins"], float)
hist = np.asarray(z["diag_overlap_hist"], float)
assert hist.shape == (2, len(bins)-1)
assert np.array_equal(hist.sum(axis=1), n_arm)

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 16,
    "axes.titlesize": 19, "axes.labelsize": 16,
    "xtick.labelsize": 16, "ytick.labelsize": 16,
    "figure.facecolor": "white", "axes.facecolor": "white",
    "savefig.facecolor": "white",
})
fig4, (left, right) = plt.subplots(1, 2, figsize=(17.2, 6.4))
fig4.subplots_adjust(left=.20, right=.975, top=.88, bottom=.27, wspace=.35)
yy = np.arange(len(labels))
cbps_color = "#008c95"
left.barh(yy-.19, before, height=.30, color="#aebcc6", label="Before weighting")
left.barh(yy+.19, after, height=.30, color=cbps_color, label="After CBPS")
margin = max(.006, .015 * before.max())
for j, (x, y) in enumerate(zip(before, after)):
    left.text(x+margin, j-.19, f"{x:.3f}", fontsize=14, va="center")
    left.text(y+margin, j+.19, f"{y:.4f}", fontsize=14, va="center",
              color=cbps_color, fontweight="bold")
left.set_yticks(yy, labels)
left.invert_yaxis()
left.set_xlim(0, max(.12, float(max(before.max(), after.max()))*1.28))
left.set_xlabel("Absolute standardized mean difference")
left.set_title("Abs SMD Before and After CBPS", pad=13)
handles, legend_labels = left.get_legend_handles_labels()
fig4.legend(handles, legend_labels, frameon=False, fontsize=15,
            loc="lower center", bbox_to_anchor=(.37, .015), ncol=2)
left.grid(axis="x", color="#dce4e7", lw=.8)
left.set_axisbelow(True)
left.spines[["top", "right"]].set_visible(False)

# Approximate each density from the saved bins;
# Smooth the CDF-spline derivative to avoid kinks at coarse bin edges.
score_grid = np.linspace(0, 1, 501)
for arm, (color, label) in enumerate((("#335c81", "On time"), ("#bf5e35", "Delayed"))):
    cdf_at_bins = np.r_[0., np.cumsum(hist[arm] / n_arm[arm])]
    cdf_at_bins[-1] = 1.
    density = np.maximum(PchipInterpolator(bins, cdf_at_bins).derivative()(score_grid), 0.)
    density = gaussian_filter1d(density, sigma=0.012 / np.diff(score_grid)[0])
    density /= trapezoid(density, score_grid)
    right.fill_between(score_grid, density, color=color, alpha=.085)
    right.plot(score_grid, density, color=color, lw=2.3, label=label)
right.set_xlim(0, 1)
right.set_ylim(bottom=0)
right.set_xticks(np.arange(0, 1.01, .2))
right.set_xlabel("Predicted delay probability")
right.set_ylabel("Estimated density")
right.set_title("Overlap Checks", pad=13)
right.legend(frameon=False, fontsize=15)
right.grid(axis="y", color="#dce4e7", lw=.8)
right.set_axisbelow(True)
right.spines[["top", "right"]].set_visible(False)
out4 = OUT_DIR / "figure4_balance_overlap.png"
fig4.savefig(out4, dpi=200, facecolor="white", bbox_inches="tight", pad_inches=.06)
plt.close(fig4)
plt.rcParams.update(original_style)
print(f"Saved → {out4}")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 5: Chronos estimator vs. AB ground truth
# ══════════════════════════════════════════════════════════════════════════════
def _delta_dr(slc):
    nt = int(weekly_mat_t[slc].sum()); nc = int(weekly_mat_c[slc].sum())
    dt = weekly_del_t[slc].sum() / max(nt, 1) * 100.
    dc = weekly_del_c[slc].sum() / max(nc, 1) * 100.
    return dt - dc

delta_dr = _delta_dr(slice(N_WEEKS - 4, N_WEEKS))
print(f"Δdr (last 4 AB weeks) = {delta_dr:+.3f}pp")

# Per-week adjustments (OLS with pre-AB spending covariate)
n_a     = len(D_act)
X       = np.column_stack([np.ones(n_a), D_act, pre_act])
XtX_inv = np.linalg.inv(X.T @ X)

anc_dgb_wk = np.zeros(N_WEEKS); anc_se_wk = np.zeros(N_WEEKS)
for w in range(N_WEEKS):
    y_w = wk_gb_act[w].astype(np.float64)
    beta_w, *_ = np.linalg.lstsq(X, y_w, rcond=None)
    resid = y_w - X @ beta_w
    s2    = float(resid.var(ddof=X.shape[1]))
    tau   = float(beta_w[1])
    se    = float(np.sqrt(s2 * XtX_inv[1, 1]))
    cm    = y_w[D_act == 0].mean()
    anc_dgb_wk[w] = tau / cm * 100. if cm > 0 else np.nan
    anc_se_wk[w]  = se  / cm * 100. if cm > 0 else np.nan

preds = {K: betas[K] * delta_dr * 100. for K in K_SWEEP}
print("Chronos predictions (beta × Δdr):")
for K, p in preds.items():
    print(f"  K={K:2d}d  beta={betas[K]*100:+.4f}%/pp  pred={p:+.3f}%")

weeks     = np.arange(N_WEEKS)
week_lbls = [f"W{w+1}" for w in range(N_WEEKS)]

fig5, ax = plt.subplots(figsize=(11, 5.5))
ax.plot(weeks, anc_dgb_wk, color="#5DBB63", lw=2.4, marker="o", ms=5, label="Weekly ΔLTV%")
ax.fill_between(weeks, anc_dgb_wk - 1.96*anc_se_wk, anc_dgb_wk + 1.96*anc_se_wk,
                color="#5DBB63", alpha=0.20, label="Weekly ΔLTV 95% CI")
ax.axhline(0, color="gray", lw=0.8, ls="--")
for K in K_SWEEP:
    p  = preds[K]
    ls = "--" if K != K_SOLID else (0, (6, 2))
    lw = 2.6  if K == K_SOLID else 2.0
    ax.axhline(p, color=K_COLORS[K], lw=lw, ls=ls, label=f"K={K}d (W{K//7}) Chronos pred")
ax.set_xticks(weeks); ax.set_xticklabels(week_lbls, fontsize=10)
ax.set_title("Weekly LTV% with 95% CI + Chronos Predictions", fontsize=16)
ax.set_xlabel("Week since AB start", fontsize=14)
ax.set_ylabel("ΔLTV%", fontsize=14)
ax.tick_params(axis="y", labelsize=13)
ax.legend(fontsize=12, loc="lower right")
fig5.tight_layout()
out5 = OUT_DIR / "figure5_chronos_validation.png"
fig5.savefig(str(out5), dpi=150, bbox_inches="tight"); plt.close(fig5)
print(f"Saved → {out5}")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 6: Chronos response by K on a fixed cohort
# ══════════════════════════════════════════════════════════════════════════════
K_h = np.asarray(z["horizon_K"], int)
assert K_h.tolist() == [28, 42, 56, 70, 84, 100]
assert int(z["horizon_common_obs_max"]) == 49
beta_h = np.asarray(z["horizon_beta_pt"], float)
boot_h = np.asarray(z["horizon_boot_cbps"], float)
assert beta_h.shape == (len(K_h),)
assert boot_h.shape == (100, len(K_h)) and np.isfinite(boot_h).all()
assert np.isclose(float(z["delta_dr_AB"]), delta_dr)
pred_h = boot_h * delta_dr * 100.
mean_h = pred_h.mean(0)
lo_h, hi_h = np.percentile(pred_h, [2.5, 97.5], axis=0)

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 13, "axes.labelsize": 13,
    "xtick.labelsize": 13, "ytick.labelsize": 13,
    "figure.facecolor": "white", "axes.facecolor": "white",
    "savefig.facecolor": "white",
})
fig6, ax6 = plt.subplots(figsize=(7.4, 4.9))
fig6.subplots_adjust(left=.12, right=.98, top=.97, bottom=.14)
teal = "#008c95"
ci = ax6.errorbar(K_h, mean_h, yerr=[mean_h-lo_h, hi_h-mean_h],
                   fmt="none", ecolor=teal, elinewidth=1.9,
                   capsize=5.5, capthick=1.9, zorder=2)
curve, = ax6.plot(K_h, mean_h, "-o", color=teal, lw=2.5,
                  markersize=6.5, zorder=3)
ax6.legend([curve, ci], ["Chronos estimate", "95% bootstrap CI"],
           frameon=False, fontsize=12, loc="lower right")
ax6.set_xticks(K_h)
ax6.set_xlim(22, 106)
ax6.set_ylim(0, float(hi_h.max()) * 1.12)
ax6.set_xlabel("Outcome horizon K (days)")
ax6.set_ylabel(r"$\Delta\,\mathrm{LTV}\%$")
ax6.spines[["top", "right"]].set_visible(False)
out6 = OUT_DIR / "figure6_horizon_k28_100.png"
fig6.savefig(out6, dpi=200, facecolor="white", bbox_inches="tight", pad_inches=.06)
plt.close(fig6)
plt.rcParams.update(original_style)
print(f"Saved → {out6}")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 8: Lifecycle composition across simulation days (stacked area)
# ══════════════════════════════════════════════════════════════════════════════
n_days_total = lc_share_daily.shape[0]
days = np.arange(n_days_total)

fig8, ax8 = plt.subplots(figsize=(12, 5))
ax8.stackplot(days, lc_share_daily.T, labels=STATE_NAMES, colors=STATE_COLORS, alpha=0.9)

final = lc_share_daily[-1]; cum = np.cumsum(final)
for s, name in enumerate(STATE_NAMES):
    if final[s] > 0.02:
        ax8.text(n_days_total + 1.0, cum[s] - final[s] / 2, name, va="center", ha="left",
                 fontsize=13, color=STATE_COLORS[s], fontweight="bold")

ax8.set_xlabel("Simulation day", fontsize=15)
ax8.set_ylabel("Population share", fontsize=15)
ax8.set_xlim(0, n_days_total + 12); ax8.set_ylim(0, 1)
ax8.tick_params(axis="both", labelsize=14)
ax8.legend(loc="upper left", fontsize=13, ncol=N_STATES, frameon=True)
fig8.suptitle(f"Lifecycle Composition Across {n_days_total} Simulation Days", fontsize=19)
fig8.tight_layout()
out8 = OUT_DIR / "figure8_lifecycle_composition.png"
fig8.savefig(str(out8), dpi=150, bbox_inches="tight"); plt.close(fig8)
print(f"Saved → {out8}")

print("\nDone.")
