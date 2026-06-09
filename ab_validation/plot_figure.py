"""
AB Validation Plots
===================
Reads the cache produced by run_simulation.py and renders the paper figures for
the AB-validation section:

  Figure 1  lifecycle transition probabilities          → fig1_lifecycle_transition.png
  Figure 2  eater-level outcomes by lifecycle state      → fig2_outcomes_by_state.png
  Figure 3  marketplace volume & congestion patterns     → fig3_marketplace_patterns.png
  Figure 4  Chronos estimator vs. AB ground truth        → fig4_chronos_validation.png
  Figure 6  lifecycle composition over 250 days          → fig6_lifecycle_composition.png

Usage (from repository root):
    python ab_validation/plot_figure.py

Input:  ab_validation/cache/ab_validation_cache.npz
Output: figures/fig{1,2,3,4,6}_*.png
"""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Paths ─────────────────────────────────────────────────────────────────────
CACHE_PATH = Path(__file__).parent / "cache" / "ab_validation_cache.npz"
OUT_DIR    = ROOT / "figures"

# ── Shared style ──────────────────────────────────────────────────────────────
STATE_NAMES  = ["New", "Casual", "Power", "AtRisk", "Churned"]
STATE_COLORS = ["#4C9BE8", "#5DBB63", "#F4A623", "#E05C5C", "#AAAAAA"]
DOW_NAMES    = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
N_STATES     = 5

# Figure 4 (Chronos) style
K_COLORS = {28: "#E67E22", 42: "#9B59B6", 56: "#2980B9", 70: "#E91E63"}
K_SOLID  = 56   # K rendered as thick solid; others dashed

# ── Load cache ────────────────────────────────────────────────────────────────
if not CACHE_PATH.exists():
    raise FileNotFoundError(
        f"Cache not found: {CACHE_PATH}\n"
        "Run ab_validation/run_simulation.py first."
    )
z = np.load(str(CACHE_PATH))

# Chronos (Figure 4)
K_SWEEP        = [int(k) for k in z["K_SWEEP"]]
betas          = {int(k): float(b) for k, b in zip(z["K_SWEEP"], z["betas"])}
weekly_del_t   = z["weekly_del_t"]; weekly_mat_t = z["weekly_mat_t"]
weekly_del_c   = z["weekly_del_c"]; weekly_mat_c = z["weekly_mat_c"]
wk_gb_act      = z["wk_gb_act"]
D_act          = z["D_act"].astype(float)
pre_act        = z["pre_act"].astype(np.float64)
n_act          = int(z["n_act"])
N_WEEKS        = int(z["N_WEEKS"])

# Obs-phase diagnostics (Figures 1, 2, 3, 6)
trans_ontime    = z["trans_ontime"]
trans_delay     = z["trans_delay"]
trans_organic   = z["trans_organic"]
lc_final        = z["lc_final"].astype(int)
spend_final     = z["spend_final"].astype(np.float64)
matched_cum     = z["matched_cum"]
delayed_cum     = z["delayed_cum"]
dow_hr_orders   = z["dow_hr_orders"]
dow_hr_couriers = z["dow_hr_couriers"]
dow_hr_matched  = z["dow_hr_matched"]
dow_hr_match_t  = z["dow_hr_match_t"]
dow_hr_count    = z["dow_hr_count"]
lc_share_daily  = z["lc_share_daily"]
N_OBS_DAYS      = int(z["N_OBS_DAYS"])

OUT_DIR.mkdir(parents=True, exist_ok=True)
print(f"Loaded cache: K_SWEEP={K_SWEEP}  n_act={n_act:,}  N_WEEKS={N_WEEKS}")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 1: Lifecycle transition probabilities (3-panel)
# ══════════════════════════════════════════════════════════════════════════════
def _row_norm(mat):
    return mat / mat.sum(axis=1, keepdims=True).clip(1)

def _plot_matrix(ax, prob_mat, count_mat, title):
    im = ax.imshow(prob_mat, vmin=0, vmax=1, cmap="Blues", aspect="auto")
    for i in range(N_STATES):
        for j in range(N_STATES):
            v = prob_mat[i, j]; c = int(count_mat[i, j])
            ax.text(j, i, f"{v:.2f}\n(n={c:,})", ha="center", va="center",
                    fontsize=11, color="white" if v > 0.55 else "black")
    ax.set_xticks(range(N_STATES)); ax.set_xticklabels(STATE_NAMES, fontsize=13)
    ax.set_yticks(range(N_STATES)); ax.set_yticklabels(STATE_NAMES, fontsize=13)
    ax.set_xlabel("State at end of day (t+1)", fontsize=15)
    ax.set_ylabel("State at start of day (t)", fontsize=15)
    ax.set_title(title, fontsize=17)
    cb = plt.colorbar(im, ax=ax, shrink=0.75)
    cb.set_label("Transition probability", fontsize=14)
    cb.ax.tick_params(labelsize=12)

fig1, axes1 = plt.subplots(1, 3, figsize=(22, 6))
_plot_matrix(axes1[0], _row_norm(trans_organic.astype(float)), trans_organic, "Did Not Order")
_plot_matrix(axes1[1], _row_norm(trans_ontime.astype(float)),  trans_ontime,  "Ordered, On-Time")
_plot_matrix(axes1[2], _row_norm(trans_delay.astype(float)),   trans_delay,   "Ordered, Delayed")
fig1.suptitle("Daily Lifecycle Transition Probabilities", fontsize=19)
fig1.tight_layout()
out1 = OUT_DIR / "fig1_lifecycle_transition.png"
fig1.savefig(str(out1), dpi=150, bbox_inches="tight"); plt.close(fig1)
print(f"Saved → {out1}")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 2: Eater-level outcomes by lifecycle state (day-250 snapshot)
# ══════════════════════════════════════════════════════════════════════════════
with np.errstate(invalid="ignore", divide="ignore"):
    ss_dr = np.where(matched_cum > 0, delayed_cum / matched_cum, np.nan)

dr_means, dr_errs, dr_ns = [], [], []
val_means, val_errs      = [], []
for s in range(N_STATES):
    mask_dr = (lc_final == s) & ~np.isnan(ss_dr)
    dr_s  = ss_dr[mask_dr] * 100
    val_s = spend_final[lc_final == s]
    dr_means.append(dr_s.mean()  if len(dr_s)  > 0 else 0.0)
    dr_errs.append(dr_s.std()  / np.sqrt(max(len(dr_s), 1)))
    dr_ns.append(len(dr_s))
    val_means.append(val_s.mean() if len(val_s) > 0 else 0.0)
    val_errs.append(val_s.std() / np.sqrt(max(len(val_s), 1)))

fig2, (ax_dr, ax_val) = plt.subplots(1, 2, figsize=(14, 5))

bars_dr = ax_dr.bar(STATE_NAMES, dr_means, color=STATE_COLORS, alpha=0.85, yerr=dr_errs, capsize=5)
for bar, n_b in zip(bars_dr, dr_ns):
    ax_dr.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(dr_errs) * 0.1 + 0.15,
               f"n={n_b:,}", ha="center", va="bottom", fontsize=12)
ax_dr.set_ylabel("Avg delay rate (%)", fontsize=15)
ax_dr.set_title("Delay Rate by Lifecycle State", fontsize=17)
ax_dr.tick_params(axis="both", labelsize=13); ax_dr.set_ylim(0, None)

bars_val = ax_val.bar(STATE_NAMES, val_means, color=STATE_COLORS, alpha=0.85, yerr=val_errs, capsize=5)
for bar, v_m in zip(bars_val, val_means):
    ax_val.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(val_errs) * 0.1 + 0.5,
                f"${v_m:.0f}", ha="center", va="bottom", fontsize=12)
ax_val.set_ylabel("Avg lifetime value ($)", fontsize=15)
ax_val.set_title("Lifetime Value by Lifecycle State", fontsize=17)
ax_val.tick_params(axis="both", labelsize=13); ax_val.set_ylim(0, None)

fig2.suptitle("Delay Rate and Lifetime Value by Lifecycle State", fontsize=19)
fig2.tight_layout()
out2 = OUT_DIR / "fig2_outcomes_by_state.png"
fig2.savefig(str(out2), dpi=150, bbox_inches="tight"); plt.close(fig2)
print(f"Saved → {out2}")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 3: Marketplace volume & congestion patterns (TOD / DOW)
# ══════════════════════════════════════════════════════════════════════════════
hours = np.arange(24); dows = np.arange(7)
total_ss_days = int(dow_hr_count.sum())

tod_orders   = dow_hr_orders.sum(0)   / max(total_ss_days, 1)
tod_couriers = dow_hr_couriers.sum(0) / max(total_ss_days, 1)
with np.errstate(invalid="ignore", divide="ignore"):
    tod_match_t = np.where(dow_hr_matched.sum(0) > 0,
                           dow_hr_match_t.sum(0) / dow_hr_matched.sum(0), np.nan)

dow_orders_daily   = dow_hr_orders.sum(1)   / dow_hr_count.clip(1)
dow_couriers_daily = dow_hr_couriers.sum(1) / dow_hr_count.clip(1)
with np.errstate(invalid="ignore", divide="ignore"):
    dow_match_t_avg = np.where(dow_hr_matched.sum(1) > 0,
                               dow_hr_match_t.sum(1) / dow_hr_matched.sum(1), np.nan)

fig3, (ax_tod, ax_dow) = plt.subplots(1, 2, figsize=(16, 5))

ax_tod_r = ax_tod.twinx()
l1, = ax_tod.plot(hours, tod_orders,   color="#4C9BE8", lw=2, marker="o", ms=4, label="Orders/hr")
l2, = ax_tod.plot(hours, tod_couriers, color="#F4A623", lw=2, marker="s", ms=4, label="Couriers/hr")
l3, = ax_tod_r.plot(hours, tod_match_t, color="#5DBB63", lw=2, marker="^", ms=4,
                    linestyle="--", label="Match time (min)")
ax_tod.set_xlabel("Hour of day", fontsize=15)
ax_tod.set_ylabel("Avg count per hour", fontsize=15)
ax_tod_r.set_ylabel("Match time (min)", fontsize=15); ax_tod_r.set_ylim(0, None)
ax_tod.set_xticks(hours[::2])
ax_tod.tick_params(axis="both", labelsize=14); ax_tod_r.tick_params(axis="y", labelsize=14)
ax_tod.set_title("Time-of-Day Marketplace Pattern", fontsize=17)
ax_tod.legend([l1, l2, l3], [l.get_label() for l in [l1, l2, l3]], fontsize=14, loc="upper left")

ax_dow_r = ax_dow.twinx()
l4, = ax_dow.plot(dows, dow_orders_daily,   color="#4C9BE8", lw=2, marker="o", ms=6, label="Daily orders")
l5, = ax_dow.plot(dows, dow_couriers_daily, color="#F4A623", lw=2, marker="s", ms=6, label="Daily couriers")
l6, = ax_dow_r.plot(dows, dow_match_t_avg, color="#5DBB63", lw=2, marker="^", ms=6,
                    linestyle="--", label="Avg match time (min)")
ax_dow.set_xlabel("Day of week", fontsize=15)
ax_dow.set_ylabel("Avg daily count", fontsize=15)
ax_dow_r.set_ylabel("Match time (min)", fontsize=15); ax_dow_r.set_ylim(0, None)
ax_dow.set_xticks(dows); ax_dow.set_xticklabels(DOW_NAMES, fontsize=14)
ax_dow.tick_params(axis="both", labelsize=14); ax_dow_r.tick_params(axis="y", labelsize=14)
ax_dow.set_title("Day-of-Week Marketplace Pattern", fontsize=17)
ax_dow.legend([l4, l5, l6], [l.get_label() for l in [l4, l5, l6]], fontsize=14, loc="upper left")

fig3.suptitle("Marketplace Volume & Congestion Patterns", fontsize=19)
fig3.tight_layout()
out3 = OUT_DIR / "fig3_marketplace_patterns.png"
fig3.savefig(str(out3), dpi=150, bbox_inches="tight"); plt.close(fig3)
print(f"Saved → {out3}")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 4: Chronos estimator vs. AB ground truth
# ══════════════════════════════════════════════════════════════════════════════
def _delta_dr(slc):
    nt = int(weekly_mat_t[slc].sum()); nc = int(weekly_mat_c[slc].sum())
    dt = weekly_del_t[slc].sum() / max(nt, 1) * 100.
    dc = weekly_del_c[slc].sum() / max(nc, 1) * 100.
    return dt - dc

delta_dr = _delta_dr(slice(N_WEEKS - 4, N_WEEKS))
print(f"Δdr (last 4 AB weeks) = {delta_dr:+.3f}pp")

# Per-week ANCOVA (OLS with pre-AB spending covariate)
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

fig4, ax = plt.subplots(figsize=(11, 5.5))
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
fig4.tight_layout()
out4 = OUT_DIR / "fig4_chronos_validation.png"
fig4.savefig(str(out4), dpi=150, bbox_inches="tight"); plt.close(fig4)
print(f"Saved → {out4}")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 6: Lifecycle composition across simulation days (stacked area)
# ══════════════════════════════════════════════════════════════════════════════
n_days_total = lc_share_daily.shape[0]
days = np.arange(n_days_total)

fig6, ax6 = plt.subplots(figsize=(12, 5))
ax6.stackplot(days, lc_share_daily.T, labels=STATE_NAMES, colors=STATE_COLORS, alpha=0.9)

final = lc_share_daily[-1]; cum = np.cumsum(final)
for s, name in enumerate(STATE_NAMES):
    if final[s] > 0.02:
        ax6.text(n_days_total + 1.0, cum[s] - final[s] / 2, name, va="center", ha="left",
                 fontsize=13, color=STATE_COLORS[s], fontweight="bold")

ax6.set_xlabel("Simulation day", fontsize=15)
ax6.set_ylabel("Population share", fontsize=15)
ax6.set_xlim(0, n_days_total + 12); ax6.set_ylim(0, 1)
ax6.tick_params(axis="both", labelsize=14)
ax6.legend(loc="upper left", fontsize=13, ncol=N_STATES, frameon=True)
fig6.suptitle(f"Lifecycle Composition Across {n_days_total} Simulation Days", fontsize=19)
fig6.tight_layout()
out6 = OUT_DIR / "fig6_lifecycle_composition.png"
fig6.savefig(str(out6), dpi=150, bbox_inches="tight"); plt.close(fig6)
print(f"Saved → {out6}")

print("\nDone.")
