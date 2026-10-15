"""Compare E3 (ResNet18 unsmoothed) and E5 (ResNet18 label_smoothing=0.05) on the frozen validation split."""
from __future__ import annotations

import json
import os
from pathlib import Path

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
    e5_dir = Path("outputs/experiment_001/E5_resnet18_label_smoothing")
    fig_dir = Path("outputs/experiment_001/figures")
    fig_dir.mkdir(parents=True, exist_ok=True)

    h3 = pd.read_csv(e3_dir / "history.csv")
    h5 = pd.read_csv(e5_dir / "history.csv")
    s3 = read_json(e3_dir / "summary.json")
    s5 = read_json(e5_dir / "summary.json")
    p3 = pd.read_csv(e3_dir / "validation" / "predictions.csv")
    p5 = pd.read_csv(e5_dir / "validation" / "predictions.csv")

    b3_ep = int(s3["best_epoch"])
    b5_ep = int(s5["best_epoch"])
    b3_row = h3[h3["epoch"] == b3_ep].iloc[0]
    b5_row = h5[h5["epoch"] == b5_ep].iloc[0]

    ece3 = compute_ece(p3["probability"].to_numpy(), p3["correct"].to_numpy(dtype=float))
    ece5 = compute_ece(p5["probability"].to_numpy(), p5["correct"].to_numpy(dtype=float))

    conf3_all = float(p3["probability"].mean())
    conf5_all = float(p5["probability"].mean())
    conf3_corr = float(p3.loc[p3["correct"] == True, "probability"].mean())
    conf5_corr = float(p5.loc[p5["correct"] == True, "probability"].mean())
    conf3_err = float(p3.loc[p3["correct"] == False, "probability"].mean())
    conf5_err = float(p5.loc[p5["correct"] == False, "probability"].mean())

    # 4-panel comparison figure:
    # (a) Unsmoothed Evaluation Cross-Entropy Loss (Clean-Train vs Val)
    # (b) Training Objective Loss (Unsmoothed E3 vs Smoothed E5 train_loss)
    # (c) Clean-Train vs Validation Top-1 Accuracy (%)
    # (d) Generalisation Gap Evolution (Clean-Train Top-1 - Val Top-1, pp)
    fig, axes = plt.subplots(2, 2, figsize=(15, 9.6), layout="constrained")

    runs = [
        ("E3: ResNet18 (label_smoothing=0.0)", h3, s3, b3_row, "#2ca02c"),
        ("E5: ResNet18 (label_smoothing=0.05)", h5, s5, b5_row, "#1f77b4"),
    ]

    for label, h, s, brow, col in runs:
        bep = int(s["best_epoch"])
        short = label.split(":")[0]
        # Panel (a): Standard Unsmoothed Cross-Entropy Loss (Clean-Train vs Val)
        axes[0, 0].plot(
            h["epoch"],
            h["clean_train_loss"],
            color=col,
            linestyle="--",
            alpha=0.7,
            label=f"{short} Clean-Train (Ep {bep}: {brow['clean_train_loss']:.4f})",
        )
        axes[0, 0].plot(
            h["epoch"],
            h["val_loss"],
            color=col,
            linewidth=2.2,
            label=f"{short} Val (Ep {bep}: {brow['val_loss']:.4f})",
        )
        axes[0, 0].scatter([bep], [brow["val_loss"]], color=col, s=65, zorder=5, edgecolor="black")

        # Panel (b): Augmented Training Objective Loss vs Unsmoothed Clean-Train Loss
        axes[0, 1].plot(
            h["epoch"],
            h["train_loss"],
            color=col,
            linewidth=2.2,
            label=f"{short} Augmented Train Objective (Ep {bep}: {brow['train_loss']:.4f})",
        )
        axes[0, 1].plot(
            h["epoch"],
            h["clean_train_loss"],
            color=col,
            linestyle=":",
            alpha=0.8,
            label=f"{short} Unsmoothed Clean-Train CE (Ep {bep}: {brow['clean_train_loss']:.4f})",
        )

        # Panel (c): Clean-Train vs Validation Top-1 Accuracy
        axes[1, 0].plot(
            h["epoch"],
            h["clean_train_top1"] * 100,
            color=col,
            linestyle="--",
            alpha=0.7,
            label=f"{short} Clean-Train ({brow['clean_train_top1']*100:.2f}%)",
        )
        axes[1, 0].plot(
            h["epoch"],
            h["val_top1"] * 100,
            color=col,
            linewidth=2.2,
            label=f"{short} Val Top-1 (Ep {bep}: {brow['val_top1']*100:.2f}%)",
        )
        axes[1, 0].scatter([bep], [brow["val_top1"] * 100], color=col, s=65, zorder=5, edgecolor="black")

        # Panel (d): Generalisation Gap Evolution
        axes[1, 1].plot(
            h["epoch"],
            h["generalisation_gap"] * 100,
            color=col,
            linewidth=2.2,
            label=f"{short} Gap (Ep {bep}: {brow['generalisation_gap']*100:.2f} pp)",
        )
        axes[1, 1].scatter([bep], [brow["generalisation_gap"] * 100], color=col, s=65, zorder=5, edgecolor="black")

    # Mark Stage 1 -> Stage 2 boundary (Epoch 5.5) and minimum theoretical smoothed CE floor
    min_smoothed_ce = -(
        (0.95 + 0.05 / 200) * np.log(0.95 + 0.05 / 200) + 199 * (0.05 / 200) * np.log(0.05 / 200)
    )
    axes[0, 1].axhline(
        min_smoothed_ce,
        color="#1f77b4",
        linestyle="-.",
        alpha=0.6,
        label=f"Theoretical E5 Smoothed CE Floor (~{min_smoothed_ce:.3f})",
    )

    for ax in axes.flat:
        ax.axvline(5.5, color="#555555", linestyle=":", alpha=0.6)
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8, loc="best")

    axes[0, 0].set(
        xlabel="Epoch",
        ylabel="Unsmoothed Cross-Entropy Loss",
        title="(a) Fair Evaluation Loss Comparison: Unsmoothed Clean-Train vs Val CE",
    )
    axes[0, 1].set(
        xlabel="Epoch",
        ylabel="Cross-Entropy Loss",
        title="(b) Training Objective Loss (Smoothed E5 vs Unsmoothed E3) & CE Floor",
    )
    axes[1, 0].set(
        xlabel="Epoch",
        ylabel="Top-1 Accuracy (%)",
        title="(c) Clean-Train vs Validation Top-1 Accuracy (%)",
        ylim=(20, 103),
    )
    axes[1, 1].set(
        xlabel="Epoch",
        ylabel="Generalisation Gap: Clean-Train - Val Top-1 (pp)",
        title="(d) Generalisation Gap Evolution Across Head Warmup & Layer4 Fine-Tuning",
    )

    fig.suptitle(
        "Controlled Transfer-Learning Ablation: E3 ResNet18 (Unsmoothed) vs E5 ResNet18 (Label Smoothing = 0.05)\n"
        "(Post-Test Exploratory Validation-Only Study on Frozen CUB-200 Split, Seed 42, N=959 Validation Images)",
        fontsize=11.5,
        fontweight="bold",
    )
    out_fig = fig_dir / "ablation_e3_vs_e5_label_smoothing.png"
    fig.savefig(out_fig, dpi=220, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)

    summary_out = {
        "study_type": "post_test_exploratory_validation_only",
        "E3_resnet18_unsmoothed": {
            "val_top1": float(s3["validation"]["top1"]),
            "val_avg_acc_per_class": float(s3["validation"]["average_accuracy_per_class"]),
            "val_top5": float(s3["validation"]["top5"]),
            "val_macro_f1": float(s3["validation"]["macro_f1"]),
            "val_unsmoothed_ce_loss": float(s3["validation"]["loss"]),
            "clean_train_top1": float(b3_row["clean_train_top1"]),
            "clean_train_avg_acc_per_class": float(b3_row["clean_train_average_accuracy_per_class"]),
            "clean_train_unsmoothed_ce_loss": float(b3_row["clean_train_loss"]),
            "train_objective_loss": float(b3_row["train_loss"]),
            "generalisation_gap": float(b3_row["generalisation_gap"]),
            "best_epoch": b3_ep,
            "stopping_epoch": int(h3["epoch"].max()),
            "runtime_seconds": round(float(h3["seconds"].sum()), 2),
            "runtime_minutes": round(float(h3["seconds"].sum() / 60.0), 2),
            "checkpoint_sha256": s3["checkpoint_sha256"],
            "calibration_10bin_ece": round(ece3, 5),
            "mean_top1_prob_all": round(conf3_all, 5),
            "mean_top1_prob_correct": round(conf3_corr, 5),
            "mean_top1_prob_incorrect": round(conf3_err, 5),
        },
        "E5_resnet18_label_smoothing_005": {
            "val_top1": float(s5["validation"]["top1"]),
            "val_avg_acc_per_class": float(s5["validation"]["average_accuracy_per_class"]),
            "val_top5": float(s5["validation"]["top5"]),
            "val_macro_f1": float(s5["validation"]["macro_f1"]),
            "val_unsmoothed_ce_loss": float(s5["validation"]["loss"]),
            "clean_train_top1": float(b5_row["clean_train_top1"]),
            "clean_train_avg_acc_per_class": float(b5_row["clean_train_average_accuracy_per_class"]),
            "clean_train_unsmoothed_ce_loss": float(b5_row["clean_train_loss"]),
            "train_objective_loss": float(b5_row["train_loss"]),
            "generalisation_gap": float(b5_row["generalisation_gap"]),
            "best_epoch": b5_ep,
            "stopping_epoch": int(h5["epoch"].max()),
            "runtime_seconds": round(float(h5["seconds"].sum()), 2),
            "runtime_minutes": round(float(h5["seconds"].sum() / 60.0), 2),
            "checkpoint_sha256": s5["checkpoint_sha256"],
            "calibration_10bin_ece": round(ece5, 5),
            "mean_top1_prob_all": round(conf5_all, 5),
            "mean_top1_prob_correct": round(conf5_corr, 5),
            "mean_top1_prob_incorrect": round(conf5_err, 5),
        },
        "figure_path": str(out_fig),
    }
    write_json(e5_dir / "e3_vs_e5_comparison.json", summary_out)
    print(json.dumps(summary_out, indent=2))


if __name__ == "__main__":
    main()

