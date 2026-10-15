"""Compare E3 (ResNet18 baseline) and E6 (Enhanced ResNet18: layer3+layer4 discriminative fine-tuning) on the frozen validation split."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(".").resolve()))
from src.utils import ROOT, read_json, write_json

os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def compute_ece(probs: np.ndarray, correct: np.ndarray, n_bins: int = 10) -> float:
    """Compute 10-bin Expected Calibration Error (ECE) on top-1 validation predictions."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(probs)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (probs >= lo) & (probs <= hi if i == n_bins - 1 else probs < hi)
        if np.any(mask):
            acc_bin = float(np.mean(correct[mask]))
            conf_bin = float(np.mean(probs[mask]))
            ece += (np.sum(mask) / n) * abs(acc_bin - conf_bin)
    return float(ece)


def main():
    e3_dir = Path("outputs/experiment_001/resnet18")
    e6_dir = Path("outputs/experiment_001/E6_resnet18_enhanced")
    fig_dir = Path("outputs/experiment_001/figures")
    fig_dir.mkdir(parents=True, exist_ok=True)

    h3 = pd.read_csv(e3_dir / "history.csv")
    h6 = pd.read_csv(e6_dir / "history.csv")
    s3 = read_json(e3_dir / "summary.json")
    s6 = read_json(e6_dir / "summary.json")
    p3 = pd.read_csv(e3_dir / "validation" / "predictions.csv")
    p6 = pd.read_csv(e6_dir / "validation" / "predictions.csv")
    c3 = pd.read_csv(e3_dir / "validation" / "per_class.csv")
    c6 = pd.read_csv(e6_dir / "validation" / "per_class.csv")

    b3_ep = int(s3["best_epoch"])
    b6_ep = int(s6["best_epoch"])
    b3_row = h3[h3["epoch"] == b3_ep].iloc[0]
    b6_row = h6[h6["epoch"] == b6_ep].iloc[0]

    ece3 = compute_ece(p3["probability"].to_numpy(), p3["correct"].to_numpy(dtype=float))
    ece6 = compute_ece(p6["probability"].to_numpy(), p6["correct"].to_numpy(dtype=float))

    conf3_all = float(p3["probability"].mean())
    conf6_all = float(p6["probability"].mean())
    conf3_corr = float(p3.loc[p3["correct"] == True, "probability"].mean())
    conf6_corr = float(p6.loc[p6["correct"] == True, "probability"].mean())
    conf3_err = float(p3.loc[p3["correct"] == False, "probability"].mean())
    conf6_err = float(p6.loc[p6["correct"] == False, "probability"].mean())

    # Per-class differences
    c_merged = pd.merge(c3[["label", "class_name", "accuracy", "support"]],
                        c6[["label", "accuracy"]],
                        on="label", suffixes=("_e3", "_e6"))
    c_merged["delta_accuracy"] = c_merged["accuracy_e6"] - c_merged["accuracy_e3"]
    improved_classes = c_merged[c_merged["delta_accuracy"] > 0].sort_values("delta_accuracy", ascending=False)
    regressed_classes = c_merged[c_merged["delta_accuracy"] < 0].sort_values("delta_accuracy", ascending=True)
    unchanged_classes = c_merged[c_merged["delta_accuracy"] == 0]

    summary = {
        "best_epoch": {"E3": b3_ep, "E6": b6_ep},
        "validation_metrics": {
            "val_top1": {
                "E3": float(b3_row["val_top1"]),
                "E6": float(b6_row["val_top1"]),
                "delta_pp": float((b6_row["val_top1"] - b3_row["val_top1"]) * 100),
            },
            "val_avg_class": {
                "E3": float(b3_row["val_average_accuracy_per_class"]),
                "E6": float(b6_row["val_average_accuracy_per_class"]),
                "delta_pp": float((b6_row["val_average_accuracy_per_class"] - b3_row["val_average_accuracy_per_class"]) * 100),
            },
            "val_loss": {
                "E3": float(b3_row["val_loss"]),
                "E6": float(b6_row["val_loss"]),
                "delta": float(b6_row["val_loss"] - b3_row["val_loss"]),
            },
            "val_top5": {
                "E3": float(s3["validation"]["top5"]),
                "E6": float(s6["validation"]["top5"]),
                "delta_pp": float((s6["validation"]["top5"] - s3["validation"]["top5"]) * 100),
            },
            "val_macro_f1": {
                "E3": float(s3["validation"]["macro_f1"]),
                "E6": float(s6["validation"]["macro_f1"]),
                "delta_pp": float((s6["validation"]["macro_f1"] - s3["validation"]["macro_f1"]) * 100),
            },
        },
        "generalisation_gap": {
            "clean_train_top1": {
                "E3": float(b3_row["clean_train_top1"]),
                "E6": float(b6_row["clean_train_top1"]),
                "delta_pp": float((b6_row["clean_train_top1"] - b3_row["clean_train_top1"]) * 100),
            },
            "gap_pp": {
                "E3": float(b3_row["generalisation_gap"] * 100),
                "E6": float(b6_row["generalisation_gap"] * 100),
                "delta_pp": float((b6_row["generalisation_gap"] - b3_row["generalisation_gap"]) * 100),
            },
            "clean_train_loss": {
                "E3": float(b3_row["clean_train_loss"]),
                "E6": float(b6_row["clean_train_loss"]),
            },
        },
        "calibration": {
            "ece_10bin": {"E3": ece3, "E6": ece6, "delta": float(ece6 - ece3)},
            "mean_confidence_all": {"E3": conf3_all, "E6": conf6_all},
            "mean_confidence_correct": {"E3": conf3_corr, "E6": conf6_corr},
            "mean_confidence_incorrect": {"E3": conf3_err, "E6": conf6_err},
        },
        "per_class_summary": {
            "total_classes": 200,
            "classes_improved": int(len(improved_classes)),
            "classes_regressed": int(len(regressed_classes)),
            "classes_unchanged": int(len(unchanged_classes)),
            "top_improved": [
                {"class_name": r["class_name"], "accuracy_e3": float(r["accuracy_e3"]), "accuracy_e6": float(r["accuracy_e6"]), "delta_pp": float(r["delta_accuracy"] * 100)}
                for _, r in improved_classes.head(10).iterrows()
            ],
            "top_regressed": [
                {"class_name": r["class_name"], "accuracy_e3": float(r["accuracy_e3"]), "accuracy_e6": float(r["accuracy_e6"]), "delta_pp": float(r["delta_accuracy"] * 100)}
                for _, r in regressed_classes.head(10).iterrows()
            ],
        },
    }

    out_json = e6_dir / "e3_vs_e6_comparison.json"
    write_json(out_json, summary)
    print(f"Comparison saved to {out_json}")

    # Generate visual figure: 4-panel comparison
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), layout="constrained")

    # Panel A: Unsmoothed Validation Loss vs Clean Train Loss
    ax = axes[0, 0]
    ax.plot(h3["epoch"], h3["val_loss"], "o-", color="#1f77b4", label="E3 Val Loss (layer4 only)", markersize=4)
    ax.plot(h6["epoch"], h6["val_loss"], "s-", color="#d62728", label="E6 Val Loss (layer3+layer4)", markersize=4)
    ax.plot(h3["epoch"], h3["clean_train_loss"], "--", color="#1f77b4", alpha=0.5, label="E3 Clean-Train Loss")
    ax.plot(h6["epoch"], h6["clean_train_loss"], "--", color="#d62728", alpha=0.5, label="E6 Clean-Train Loss")
    ax.axvline(b3_ep, color="#1f77b4", linestyle=":", alpha=0.7, label=f"E3 Best Ep {b3_ep}")
    ax.axvline(b6_ep, color="#d62728", linestyle=":", alpha=0.7, label=f"E6 Best Ep {b6_ep}")
    ax.set_title("A. Cross-Entropy Loss (Validation vs Clean-Train)", fontsize=11, fontweight="bold")
    ax.set_xlabel("Epoch", fontsize=10)
    ax.set_ylabel("Loss", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="upper right")

    # Panel B: Top-1 Accuracy Curves
    ax = axes[0, 1]
    ax.plot(h3["epoch"], h3["val_top1"] * 100, "o-", color="#1f77b4", label=f"E3 Val Top-1 (Peak {b3_row['val_top1']*100:.2f}%)", markersize=4)
    ax.plot(h6["epoch"], h6["val_top1"] * 100, "s-", color="#d62728", label=f"E6 Val Top-1 (Peak {b6_row['val_top1']*100:.2f}%)", markersize=4)
    ax.plot(h3["epoch"], h3["clean_train_top1"] * 100, "--", color="#1f77b4", alpha=0.4, label="E3 Clean-Train Top-1")
    ax.plot(h6["epoch"], h6["clean_train_top1"] * 100, "--", color="#d62728", alpha=0.4, label="E6 Clean-Train Top-1")
    ax.axvline(b3_ep, color="#1f77b4", linestyle=":", alpha=0.7)
    ax.axvline(b6_ep, color="#d62728", linestyle=":", alpha=0.7)
    ax.set_title("B. Top-1 Classification Accuracy (%)", fontsize=11, fontweight="bold")
    ax.set_xlabel("Epoch", fontsize=10)
    ax.set_ylabel("Top-1 Accuracy (%)", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")

    # Panel C: Generalisation Gap Evolution
    ax = axes[1, 0]
    ax.plot(h3["epoch"], h3["generalisation_gap"] * 100, "o-", color="#1f77b4", label=f"E3 Gap ({b3_row['generalisation_gap']*100:.2f} pp at ep {b3_ep})", markersize=4)
    ax.plot(h6["epoch"], h6["generalisation_gap"] * 100, "s-", color="#d62728", label=f"E6 Gap ({b6_row['generalisation_gap']*100:.2f} pp at ep {b6_ep})", markersize=4)
    ax.set_title("C. Generalisation Gap (Clean-Train − Val Top-1 pp)", fontsize=11, fontweight="bold")
    ax.set_xlabel("Epoch", fontsize=10)
    ax.set_ylabel("Generalisation Gap (pp)", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="upper left")

    # Panel D: Per-Class Recall Delta Distribution
    ax = axes[1, 1]
    deltas = c_merged["delta_accuracy"] * 100
    n_bins = 25
    counts, edges, patches = ax.hist(deltas, bins=n_bins, color="#2ca02c", edgecolor="black", alpha=0.7)
    for patch, left_edge in zip(patches, edges[:-1]):
        if left_edge < 0:
            patch.set_facecolor("#d62728")
        elif left_edge > 0:
            patch.set_facecolor("#1f77b4")
        else:
            patch.set_facecolor("#7f7f7f")
    ax.axvline(0, color="black", linestyle="--", linewidth=1.2)
    mean_delta = float(deltas.mean())
    ax.axvline(mean_delta, color="gold", linestyle="-", linewidth=1.8, label=f"Mean Delta = {mean_delta:+.2f} pp")
    ax.set_title(f"D. Per-Class Recall Delta (E6 − E3)\nImproved: {len(improved_classes)} | Regressed: {len(regressed_classes)} | Unchanged: {len(unchanged_classes)}", fontsize=11, fontweight="bold")
    ax.set_xlabel("Recall Delta (pp)", fontsize=10)
    ax.set_ylabel("Number of Species", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="upper right")

    fig_path = fig_dir / "ablation_e3_vs_e6.png"
    fig.savefig(fig_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved to {fig_path}")

    # Print summary table
    print("\n" + "=" * 75)
    print("  E3 (ResNet18 Baseline) vs E6 (ResNet18 Enhanced: layer3+layer4)")
    print("=" * 75)
    print(f"{'Metric':<35} | {'E3 (Baseline)':<15} | {'E6 (Enhanced)':<15} | {'Delta':<10}")
    print("-" * 75)
    vm = summary["validation_metrics"]
    print(f"{'Best Epoch':<35} | {b3_ep:<15} | {b6_ep:<15} | -")
    print(f"{'Val Top-1 Accuracy':<35} | {vm['val_top1']['E3']:.2%} | {vm['val_top1']['E6']:.2%} | {vm['val_top1']['delta_pp']:+.2f} pp")
    print(f"{'Val Average Accuracy / Class':<35} | {vm['val_avg_class']['E3']:.2%} | {vm['val_avg_class']['E6']:.2%} | {vm['val_avg_class']['delta_pp']:+.2f} pp")
    print(f"{'Val Top-5 Accuracy':<35} | {vm['val_top5']['E3']:.2%} | {vm['val_top5']['E6']:.2%} | {vm['val_top5']['delta_pp']:+.2f} pp")
    print(f"{'Val Macro-F1':<35} | {vm['val_macro_f1']['E3']:.2%} | {vm['val_macro_f1']['E6']:.2%} | {vm['val_macro_f1']['delta_pp']:+.2f} pp")
    print(f"{'Val Loss':<35} | {vm['val_loss']['E3']:.4f} | {vm['val_loss']['E6']:.4f} | {vm['val_loss']['delta']:+.4f}")
    gg = summary["generalisation_gap"]
    print(f"{'Clean-Train Top-1':<35} | {gg['clean_train_top1']['E3']:.2%} | {gg['clean_train_top1']['E6']:.2%} | {gg['clean_train_top1']['delta_pp']:+.2f} pp")
    print(f"{'Generalisation Gap':<35} | {gg['gap_pp']['E3']:.2f} pp | {gg['gap_pp']['E6']:.2f} pp | {gg['gap_pp']['delta_pp']:+.2f} pp")
    cal = summary["calibration"]
    print(f"{'10-Bin ECE':<35} | {cal['ece_10bin']['E3']:.4f} | {cal['ece_10bin']['E6']:.4f} | {cal['ece_10bin']['delta']:+.4f}")
    print("=" * 75)


if __name__ == "__main__":
    main()

