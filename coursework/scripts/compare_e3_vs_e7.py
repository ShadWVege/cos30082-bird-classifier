"""Generate publication-quality comparison between E3 ResNet18 and E7 ConvNeXt-Tiny."""
import csv
import json
from pathlib import Path
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "figure.titlesize": 13,
})

ROOT = Path(__file__).resolve().parent.parent
e3_csv = ROOT / "outputs/experiment_001/resnet18/history.csv"
e7_csv = ROOT / "outputs/experiment_001/E7_convnext_tiny/history.csv"

def read_history(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append({
                "epoch": int(r["epoch"]),
                "stage": r["stage"],
                "train_loss": float(r["train_loss"]),
                "clean_train_loss": float(r["clean_train_loss"]),
                "val_loss": float(r["val_loss"]),
                "train_top1": float(r["train_top1"]) * 100,
                "clean_train_top1": float(r["clean_train_top1"]) * 100,
                "val_top1": float(r["val_top1"]) * 100,
                "val_avg_class": float(r["val_average_accuracy_per_class"]) * 100,
                "generalisation_gap": float(r["generalisation_gap"]) * 100,
            })
    return rows

e3_hist = read_history(e3_csv)
e7_hist = read_history(e7_csv)

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), dpi=300)

# Panel (a): Loss Curves
ax = axes[0]
ax.plot([r["epoch"] for r in e3_hist], [r["clean_train_loss"] for r in e3_hist], label="E3 ResNet18 Clean-Train Loss", color="#1f77b4", linestyle="--", alpha=0.8)
ax.plot([r["epoch"] for r in e3_hist], [r["val_loss"] for r in e3_hist], label="E3 ResNet18 Val Loss (best: 1.1895)", color="#1f77b4", linewidth=2)
ax.plot([r["epoch"] for r in e7_hist], [r["clean_train_loss"] for r in e7_hist], label="E7 ConvNeXt-Tiny Clean-Train Loss", color="#2ca02c", linestyle="--", alpha=0.8)
ax.plot([r["epoch"] for r in e7_hist], [r["val_loss"] for r in e7_hist], label="E7 ConvNeXt-Tiny Val Loss (best: 0.6985)", color="#2ca02c", linewidth=2)
ax.set_title("(a) Cross-Entropy Loss Curves")
ax.set_xlabel("Epoch")
ax.set_ylabel("Cross-Entropy Loss")
ax.grid(True, linestyle=":", alpha=0.6)
ax.legend(fontsize=8, loc="upper right")

# Panel (b): Top-1 Accuracy Curves
ax = axes[1]
ax.plot([r["epoch"] for r in e3_hist], [r["clean_train_top1"] for r in e3_hist], label="E3 Clean-Train (99.90%)", color="#1f77b4", linestyle="--", alpha=0.8)
ax.plot([r["epoch"] for r in e3_hist], [r["val_top1"] for r in e3_hist], label="E3 Val Top-1 (best: 69.66%)", color="#1f77b4", linewidth=2)
ax.plot([r["epoch"] for r in e7_hist], [r["clean_train_top1"] for r in e7_hist], label="E7 Clean-Train (99.82%)", color="#2ca02c", linestyle="--", alpha=0.8)
ax.plot([r["epoch"] for r in e7_hist], [r["val_top1"] for r in e7_hist], label="E7 Val Top-1 (best: 80.40%)", color="#2ca02c", linewidth=2)
ax.axhline(69.66, color="#1f77b4", linestyle=":", alpha=0.5)
ax.axhline(80.40, color="#2ca02c", linestyle=":", alpha=0.5)
ax.set_title("(b) Top-1 Accuracy Curves (%)")
ax.set_xlabel("Epoch")
ax.set_ylabel("Top-1 Accuracy (%)")
ax.grid(True, linestyle=":", alpha=0.6)
ax.legend(fontsize=8, loc="lower right")

# Panel (c): Generalisation Gap (Clean-Train minus Val Top-1)
ax = axes[2]
ax.plot([r["epoch"] for r in e3_hist], [r["generalisation_gap"] for r in e3_hist], label="E3 ResNet18 Gap (30.24 pp)", color="#d62728", linewidth=2)
ax.plot([r["epoch"] for r in e7_hist], [r["generalisation_gap"] for r in e7_hist], label="E7 ConvNeXt-Tiny Gap (19.42 pp)", color="#2ca02c", linewidth=2)
ax.set_title("(c) Generalisation Gap: Clean-Train - Val (%)")
ax.set_xlabel("Epoch")
ax.set_ylabel("Generalisation Gap (pp)")
ax.grid(True, linestyle=":", alpha=0.6)
ax.legend(fontsize=8, loc="center right")

plt.suptitle("Architecture Progression: E3 ResNet18 (11.28M) vs E7 ConvNeXt-Tiny (27.97M) on Frozen Development Split", y=1.02)
plt.tight_layout()

out_path = ROOT / "outputs/experiment_001/figures/ablation_e3_vs_e7.png"
fig.savefig(out_path, bbox_inches="tight", dpi=300)
plt.close(fig)
print(f"Comparison figure saved to {out_path}")

