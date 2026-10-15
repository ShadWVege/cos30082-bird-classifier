"""Generate authoritative, updated official test metrics comparison figure including E7 and McNemar analysis."""
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

fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.8), dpi=300, layout="constrained")

# =========================================================================
# Subplot (a): Official Held-Out Test Set Performance (All Evaluated Models)
# =========================================================================
ax0 = axes[0]
models = [
    "E1 Baseline",
    "E2 Regularized",
    "E3 ResNet18\n[Pre-Committed]",
    "E7 ConvNeXt\n[Single-View]",
    "E7 ConvNeXt\n[2-View TTA]"
]
top1 = [21.01, 8.97, 70.43, 80.73, 80.32]
avg_cls = [20.63, 8.80, 70.12, 80.24, 79.86]
top5 = [42.03, 25.33, 89.12, 95.18, 94.93]

x = np.arange(len(models))
w = 0.26

b1 = ax0.bar(x - w, top1, w, label="Top-1 Accuracy (%)", color="#1f77b4", edgecolor="black", linewidth=0.5)
b2 = ax0.bar(x, avg_cls, w, label="Avg Acc / Class (%)", color="#2ca02c", edgecolor="black", linewidth=0.5)
b3 = ax0.bar(x + w, top5, w, label="Top-5 Accuracy (%)", color="#9467bd", edgecolor="black", linewidth=0.5)

for bars in (b1, b2, b3):
    for bar in bars:
        h = bar.get_height()
        ax0.text(bar.get_x() + bar.get_width() / 2, h + 1.2, f"{h:.1f}%", ha="center", va="bottom", fontsize=7.5, fontweight="bold")

ax0.set_xticks(x)
ax0.set_xticklabels(models, fontsize=8.5)
ax0.set_ylim(0, 108)
ax0.set_ylabel("Score (%)")
ax0.set_title("(a) Official Held-Out Test Scores (N=1,204)\nE4, E5, E6: Not evaluated on test set", fontweight="bold")
ax0.grid(axis="y", linestyle=":", alpha=0.6)
ax0.legend(loc="upper left", fontsize=8)

# =========================================================================
# Subplot (b): Paired McNemar Contingency Matrix (E3 ResNet18 vs E7 ConvNeXt)
# =========================================================================
ax1 = axes[1]
matrix = np.array([
    [784, 64],    # E7 Correct, E7 Incorrect (when E3 Correct)
    [188, 168]    # E7 Correct, E7 Incorrect (when E3 Incorrect)
])
# Labels: row = E3 (Correct, Incorrect), col = E7 (Correct, Incorrect)
# Actually:
# E3 Correct & E7 Correct: 784
# E3 Correct & E7 Incorrect: 64
# E3 Incorrect & E7 Correct: 188
# E3 Incorrect & E7 Incorrect: 168
cax = ax1.imshow(matrix, cmap="Blues", interpolation="nearest")
for i in range(2):
    for j in range(2):
        val = matrix[i, j]
        pct = val / 1204 * 100
        color = "white" if val > 400 else "black"
        lbl = f"{val}\n({pct:.1f}%)"
        if i == 0 and j == 1:
            lbl += "\n(E3 win)"
        elif i == 1 and j == 0:
            lbl += "\n(E7 win)"
        ax1.text(j, i, lbl, ha="center", va="center", color=color, fontsize=9.5, fontweight="bold")

ax1.set_xticks([0, 1])
ax1.set_yticks([0, 1])
ax1.set_xticklabels(["E7 Correct", "E7 Incorrect"], fontsize=9, fontweight="bold")
ax1.set_yticklabels(["E3 Correct", "E3 Incorrect"], fontsize=9, fontweight="bold")
ax1.set_title("(b) Paired McNemar 2x2 Matrix (N=1,204)\nDiscordant: 188 vs 64 | Exact p = 2.48e-15", fontweight="bold")
ax1.text(0.5, -0.22, "McNemar Chi2 = 60.04 (p < 0.0001)\nNet gain: +124 images (+10.30 pp Top-1)",
         ha="center", va="top", transform=ax1.transAxes, fontsize=8.5,
         bbox=dict(boxstyle="round,pad=0.3", facecolor="#e8f4f8", edgecolor="#b0d4de"))

# =========================================================================
# Subplot (c): Development Validation vs Held-Out Test Consistency Parity
# =========================================================================
ax2 = axes[2]
comp_models = ["E3 ResNet18", "E7 ConvNeXt-Tiny"]
val_top1 = [69.66, 80.40]
test_top1 = [70.43, 80.73]
val_avg = [69.10, 79.75]
test_avg = [70.12, 80.24]

xc = np.arange(len(comp_models))
wc = 0.2

b_vt = ax2.bar(xc - 1.5 * wc, val_top1, wc, label="Val Top-1 (N=959)", color="#3182bd", edgecolor="black", linewidth=0.5)
b_tt = ax2.bar(xc - 0.5 * wc, test_top1, wc, label="Test Top-1 (N=1,204)", color="#08519c", edgecolor="black", linewidth=0.5)
b_va = ax2.bar(xc + 0.5 * wc, val_avg, wc, label="Val Avg/Class", color="#74c476", edgecolor="black", linewidth=0.5)
b_ta = ax2.bar(xc + 1.5 * wc, test_avg, wc, label="Test Avg/Class", color="#238b45", edgecolor="black", linewidth=0.5)

for bars in (b_vt, b_tt, b_va, b_ta):
    for bar in bars:
        h = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2, h + 1.2, f"{h:.1f}%", ha="center", va="bottom", fontsize=7.5, fontweight="bold")

ax2.set_xticks(xc)
ax2.set_xticklabels(comp_models, fontsize=9.5, fontweight="bold")
ax2.set_ylim(0, 108)
ax2.set_ylabel("Score (%)")
ax2.set_title("(c) Validation vs Held-Out Test Generalisation Parity\nΔ(Test - Val): E3 = +0.77 pp | E7 = +0.33 pp", fontweight="bold")
ax2.grid(axis="y", linestyle=":", alpha=0.6)
ax2.legend(loc="upper left", fontsize=7.5, ncol=2)

fig.suptitle("Authoritative Official Held-Out Test Evaluation & Statistical Comparison (CUB-200, N=1,204)", fontsize=11.5, fontweight="bold")

out_results = ROOT / "outputs/final_test_results/figures/test_metrics_summary.png"
out_deploy = ROOT / "deploy/e7_lightning/assets/test_metrics_summary.png"
out_results.parent.mkdir(parents=True, exist_ok=True)
out_deploy.parent.mkdir(parents=True, exist_ok=True)

fig.savefig(out_results, dpi=300, bbox_inches="tight")
fig.savefig(out_deploy, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"Updated test_metrics_summary.png saved to:\n  {out_results}\n  {out_deploy}")

