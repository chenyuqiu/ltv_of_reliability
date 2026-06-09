"""
Switchback Validation Plot — Figure 5
=======================================
Reads results from run_simulation.py and generates the publication figure:
two-panel box plot showing (left) 28-day delay impact and (right) long-term
value impact across 20 eater-resampled fork runs.

Usage (from repository root):
    python switchback_validation/plot_figure.py

Input:  switchback_validation/results/sw_validation_results.json
Output: figures/fig5_switchback_validation.png
"""
from __future__ import annotations
import json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS_PATH = Path(__file__).parent / "results" / "sw_validation_results.json"
OUT_DIR      = ROOT / "figures"
OUT_PATH     = OUT_DIR / "fig5_switchback_validation.png"

C_GT     = "#2196F3"   # blue  = ground truth
C_SB     = "#F44336"   # red   = switchback delay estimate (left)
C_SB_LTV = "#9C27B0"   # purple = switchback LTV estimate (right)
C_CHR    = "#4CAF50"   # green = Chronos prediction

if not RESULTS_PATH.exists():
    raise FileNotFoundError(
        f"Results not found: {RESULTS_PATH}\n"
        "Run switchback_validation/run_simulation.py first."
    )
with open(str(RESULTS_PATH)) as f:
    d = json.load(f)

beta = d["beta_cbps"]["mean_pp"]      # %/pp point estimate
fr   = d["fork_results"]
print(f"Loaded {len(fr)} fork results  β_CBPS={beta:+.4f}%/pp")

tau_dm   = np.array([r["tau_DM"]        for r in fr if r["tau_DM"]        is not None])
dspend_sb = np.array([r["delta_spend_pct"] for r in fr if r["delta_spend_pct"] is not None])
pred     = np.array([r["tau_DM"] * beta for r in fr if r["tau_DM"]        is not None])
gt_delay = float(np.mean([r["tau_GT_28d"] for r in fr]))
gt_lt    = float(np.mean([r["gt_lt"]      for r in fr]))

print(f"  τ_DM:    mean={tau_dm.mean():+.2f}  std={tau_dm.std():.2f}  (GT {gt_delay:+.2f} pp)")
print(f"  δS_SB:   mean={dspend_sb.mean():+.2f}  std={dspend_sb.std():.2f}")
print(f"  Chronos: mean={pred.mean():+.2f}  std={pred.std():.2f}  (GT {gt_lt:+.2f}%)")

def _box(ax, data_list, positions, colors):
    bp = ax.boxplot(data_list, positions=positions, widths=0.45, patch_artist=True,
                    showmeans=True, meanline=True,
                    meanprops=dict(color="black", lw=2.0),
                    medianprops=dict(visible=False),
                    whiskerprops=dict(color="#555"), capprops=dict(color="#555"),
                    flierprops=dict(marker="o", ms=4, mfc="#999", mec="none", alpha=0.5))
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c); patch.set_alpha(0.45); patch.set_edgecolor(c)

fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 5.5))

# ── Left: 28-day delay impact ─────────────────────────────────────────────────
_box(axL, [tau_dm], [1], [C_SB])
axL.axhline(gt_delay, color=C_GT, ls="--", lw=2.5, label="True Delay Impact")
axL.axhline(0, color="gray", lw=1.0, ls="--")
axL.set_xticks([1]); axL.set_xticklabels(["Switchback\nestimate"], fontsize=17)
axL.set_xlim(0.5, 1.5)
axL.set_ylim(-12, 1)
axL.tick_params(axis="y", labelsize=14)
axL.set_ylabel("28-day delay rate change (pp)", fontsize=15)
axL.set_title("Delay impact", fontsize=17)
axL.legend(fontsize=14, loc="upper left")
axL.grid(axis="y", alpha=0.3)
for sp in ["top", "right"]: axL.spines[sp].set_visible(False)

# ── Right: long-term value impact ─────────────────────────────────────────────
_box(axR, [dspend_sb, pred], [1, 2], [C_SB_LTV, C_CHR])
axR.axhline(gt_lt, color=C_GT, ls="--", lw=2.5, label="True LTV Impact")
axR.axhline(0, color="gray", lw=0.7, ls=":")
axR.set_xticks([1, 2])
axR.set_xticklabels(["Switchback\nestimate", "Chronos\nprediction"], fontsize=17)
axR.set_xlim(0.5, 2.5)
axR.tick_params(axis="y", labelsize=14)
axR.set_ylabel("Long-term value impact (%)", fontsize=15)
axR.set_title("Long-term value impact", fontsize=17)
axR.legend(fontsize=14, loc="best")
axR.grid(axis="y", alpha=0.3)
for sp in ["top", "right"]: axR.spines[sp].set_visible(False)

fig.suptitle(f"Switchback validation across {len(fr)} eater-resampled runs", fontsize=18)
fig.tight_layout(rect=[0, 0, 1, 0.96])

OUT_DIR.mkdir(parents=True, exist_ok=True)
fig.savefig(str(OUT_PATH), dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\nSaved → {OUT_PATH}")
