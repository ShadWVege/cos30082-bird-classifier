"""Generate updated comprehensive model_comparison.png showing all 7 experiments on the frozen development split."""
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9.5,
    "axes.titlesize": 10.5,
    "axes.labelsize": 9.5,
    "figure.titlesize": 12,
})

fig, axes = plt.subplots(1, 3, figsize=(17, 5.2), dpi=300, layout="constrained")

experiments = [
    "E1 Baseline\n(Custom 4-Block)",
    "E2 Reg.\n(Dropout 0.4)",
    "E4 Ablation\n(Dropout 0.0)",
    "E3 ResNet18\n[Pre-Committed]",
    "E5 Label Smooth\n(ls=0.05)",
    "E6 Enhanced\n(layer3+4+fc)",
    "E7 ConvNeXt\n[Current App]"
]

clean_train = [80.18, 16.61, 30.47, 99.90, 99.90, 98.74, 99.82]
val_top1 = [21.69, 10.43, 14.39, 69.66, 68.72, 68.40, 80.40]
val_avg = [21.29, 9.63, 13.93, 69.10, 68.02, 67.88, 79.75]
val_top5 = [42.75, 27.11, 34.83, 89.57, 88.11, 88.95, 96.35]
gaps = [58.49, 6.19, 16.08, 30.24, 31.18, 30.34, 19.42]

x = np.arange(len(experiments))

# =========================================================================
# Subplot (a): Clean-Train vs Validation Top-1 Accuracy
# =========================================================================
ax0 = axes[0]
w = 0.35
b_tr = ax0.bar(x - w/2, clean_train, w, label="Clean-Train Top-1 (%)", color="#aec7e8", edgecolor="black", linewidth=0.5)
b_val = ax0.bar(x + w/2, val_top1, w, label="Val Top-1 (%)", color="#1f77b4", edgecolor="black", linewidth=0.5)

for bar in b_val:
    h = bar.get_height()
    ax0.text(bar.get_x() + bar.get_width()/2, h + 1.2, f"{h:.1f}%", ha="center", va="bottom", fontsize=7.5, fontweight="bold")

ax0.set_xticks(x)
ax0.set_xticklabels(experiments, fontsize=8, rotation=20, ha="right")
ax0.set_ylim(0, 112)
ax0.set_ylabel("Accuracy (%)")
ax0.set_title("(a) Clean-Train vs Validation Top-1 (N=3,870 / 959)", fontweight="bold")
ax0.grid(axis="y", linestyle=":", alpha=0.6)
ax0.legend(loc="upper left", fontsize=8)

# =========================================================================
# Subplot (b): Validation Metrics: Top-1 vs Average Accuracy / Class
# =========================================================================
ax1 = axes[1]
w = 0.26
b1 = ax1.bar(x - w, val_top1, w, label="Val Top-1 (%)", color="#1f77b4", edgecolor="black", linewidth=0.5)
b2 = ax1.bar(x, val_avg, w, label="Val Avg / Class (%)", color="#2ca02c", edgecolor="black", linewidth=0.5)
b3 = ax1.bar(x + w, val_top5, w, label="Val Top-5 (%)", color="#9467bd", edgecolor="black", linewidth=0.5)

for bar in b2:
    h = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2, h + 1.2, f"{h:.1f}%", ha="center", va="bottom", fontsize=7.5, fontweight="bold")

ax1.set_xticks(x)
ax1.set_xticklabels(experiments, fontsize=8, rotation=20, ha="right")
ax1.set_ylim(0, 112)
ax1.set_ylabel("Validation Score (%)")
ax1.set_title("(b) Validation Top-1, Avg/Class, and Top-5 (N=959)", fontweight="bold")
ax1.grid(axis="y", linestyle=":", alpha=0.6)
ax1.legend(loc="upper left", fontsize=8)

# =========================================================================
# Subplot (c): Generalisation Gap Evolution (Clean-Train minus Val Top-1)
# =========================================================================
ax2 = axes[2]
colors = ["#d62728", "#2ca02c", "#ff7f0e", "#1f77b4", "#9467bd", "#8c564b", "#17becf"]
b_gap = ax2.bar(x, gaps, 0.55, color=colors, edgecolor="black", linewidth=0.6)

for bar in b_gap:
    h = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2, h + 1.0, f"{h:.1f} pp", ha="center", va="bottom", fontsize=8, fontweight="bold")

ax2.set_xticks(x)
ax2.set_xticklabels(experiments, fontsize=8, rotation=20, ha="right")
ax2.set_ylim(0, 70)
ax2.set_ylabel("Generalisation Gap (pp)")
ax2.set_title("(c) Generalisation Gap (Clean-Train - Val Top-1)\nE7 reduces gap to 19.42 pp while achieving 80.40% Val", fontweight="bold")
ax2.grid(axis="y", linestyle=":", alpha=0.6)

fig.suptitle("Complete 7-Experiment Architecture & Regularisation Progression (Frozen Split, 200 Classes)", fontsize=11.5, fontweight="bold")

out_results = ROOT / "outputs/experiment_001/figures/model_comparison.png"
out_deploy = ROOT / "deploy/e7_lightning/assets/model_comparison.png"

fig.savefig(out_results, dpi=300, bbox_inches="tight")
fig.savefig(out_deploy, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"Updated model_comparison.png saved to:\n  {out_results}\n  {out_deploy}")

