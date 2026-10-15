"""Generate presentation-ready visual evidence for Sections A, B, C, and D from stored runs."""
from __future__ import annotations

import json
import os
from pathlib import Path

from src.utils import ROOT
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))

import matplotlib
matplotlib.use("Agg")
from matplotlib import colormaps
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
import torch
from torchvision import transforms as T

from src.dataset import image_transform
from src.engine import load_checkpoint
from src.gradcam import gradcam
from src.models import build_model
from src.utils import choose_device, data_config, load_split, read_json, write_json


def compute_top5_and_cam(model, config, class_names, image_path, device):
    """Compute 200-way softmax Top-5 probabilities and layer4 Grad-CAM for a single image."""
    layer = model.layer4 if config["model"] == "resnet18" else model.features
    captured = []

    def hook_fn(_module, _inputs, output):
        output.retain_grad()
        captured.append(output)

    hook = layer.register_forward_hook(hook_fn)
    try:
        with Image.open(image_path) as im:
            original = ImageOps.exif_transpose(im).convert("RGB")
        size = config["image_size"]
        tensor = image_transform(size)(original).unsqueeze(0).to(device).requires_grad_(True)
        model.zero_grad(set_to_none=True)
        logits = model(tensor)
        probs = logits.detach().float().softmax(dim=1)[0].cpu().numpy()
        top5_idx = np.argsort(-probs, kind="stable")[:5]
        top5 = [
            {"label": int(idx), "name": class_names[int(idx)], "prob": float(probs[int(idx)])}
            for idx in top5_idx
        ]
        pred_target = int(top5_idx[0])
        logits[0, pred_target].backward()
        activations = captured[0]
        weights = activations.grad.mean(dim=(2, 3), keepdim=True)
        cam = (weights * activations).sum(1, keepdim=True).relu()
        cam = torch.nn.functional.interpolate(
            cam, size=(size, size), mode="bilinear", align_corners=False
        )[0, 0]
        cam_np = cam.detach().cpu().numpy()
        cam_np = cam_np / max(float(cam_np.max()), 1e-8)

        crop = T.CenterCrop(size)(T.Resize(round(size * 256 / 224))(original))
        crop_np = np.asarray(crop)
        heat_rgb = (colormaps["jet"](cam_np)[..., :3] * 255).astype(np.uint8)
        overlay_np = np.clip(0.55 * crop_np + 0.45 * heat_rgb, 0, 255).astype(np.uint8)
        return crop_np, heat_rgb, overlay_np, top5, probs
    finally:
        hook.remove()


def select_deterministic_examples(predictions_df, confused_pairs_df):
    """Select 3 correct and 3 incorrect samples using transparent deterministic percentile/margin criteria."""
    correct_df = predictions_df[predictions_df["correct"] == True].sort_values(
        ["probability", "name"], ascending=[False, True]
    ).reset_index(drop=True)
    wrong_df = predictions_df[predictions_df["correct"] == False].sort_values(
        ["probability", "name"], ascending=[False, True]
    ).reset_index(drop=True)

    # 3 Correct: 100th percentile (highest confidence), 50th percentile (median), 10th percentile (borderline correct)
    c_high = correct_df.iloc[0]
    c_med = correct_df.iloc[len(correct_df) // 2]
    c_low = correct_df.iloc[int(len(correct_df) * 0.90)]

    # 3 Incorrect:
    # 1. Highest-confidence mistake
    w_high = wrong_df.iloc[0]
    # 2. Representative from #1 confused species pair (different from w_high)
    top_pair_true = confused_pairs_df.iloc[0]["true"]
    top_pair_pred = confused_pairs_df.iloc[0]["predicted"]
    pair_matches = wrong_df[
        (wrong_df["true_name"] == top_pair_true)
        & (wrong_df["predicted_name"] == top_pair_pred)
        & (wrong_df["name"] != w_high["name"])
    ]
    w_pair = pair_matches.iloc[0] if len(pair_matches) > 0 else wrong_df.iloc[1]
    # 3. Median-confidence mistake (representative typical error)
    w_med_candidates = wrong_df[~wrong_df["name"].isin([w_high["name"], w_pair["name"]])].reset_index(drop=True)
    w_med = w_med_candidates.iloc[len(w_med_candidates) // 2]

    return [
        ("Correct #1 (Max Confidence, P100)", c_high),
        ("Correct #2 (Median Confidence, P50)", c_med),
        ("Correct #3 (Low-Margin Correct, P10)", c_low),
        ("Error #1 (High-Confidence Mistake, P100)", w_high),
        (f"Error #2 (Top Confusion Pair: {top_pair_true} -> {top_pair_pred})", w_pair),
        ("Error #3 (Median-Confidence Mistake, P50)", w_med),
    ]


def render_prediction_gallery(examples, records_by_name, image_root, model, config, class_names, device, out_path, title):
    """Render a 6-item gallery (3 correct, 3 incorrect) with image crop and Top-5 horizontal softmax bars."""
    fig, axes = plt.subplots(3, 4, figsize=(16, 10.5), layout="constrained")
    gallery_records = []

    for idx, (criterion_label, row) in enumerate(examples):
        r_idx = idx % 3
        c_offset = 0 if idx < 3 else 2
        img_ax = axes[r_idx, c_offset]
        bar_ax = axes[r_idx, c_offset + 1]

        rec = records_by_name[row["name"]]
        img_path = Path(image_root) / rec["relative_path"]
        crop_np, _, _, top5, probs = compute_top5_and_cam(model, config, class_names, img_path, device)

        true_id = int(row["true_label"])
        pred_id = int(row["predicted_label"])
        true_prob = float(probs[true_id])
        is_correct = true_id == pred_id

        img_ax.imshow(crop_np)
        img_ax.set_xticks([])
        img_ax.set_yticks([])
        status_color = "#1b7837" if is_correct else "#b2182b"
        img_ax.set_title(
            f"{criterion_label}\nFile: {row['name']}\n"
            f"True: [{true_id:03d}] {class_names[true_id]} ({true_prob*100:.1f}%)\n"
            f"Pred: [{pred_id:03d}] {class_names[pred_id]} ({top5[0]['prob']*100:.1f}%)",
            fontsize=8.5,
            fontweight="bold",
            color=status_color,
        )

        labels = [f"[{item['label']:03d}] {item['name']}" for item in reversed(top5)]
        values = [item["prob"] * 100 for item in reversed(top5)]
        colors = [
            "#2ca02c" if item["label"] == true_id else ("#d62728" if item["label"] == pred_id else "#7f7f7f")
            for item in reversed(top5)
        ]
        bars = bar_ax.barh(range(len(top5)), values, color=colors, height=0.62)
        bar_ax.set_yticks(range(len(top5)), labels, fontsize=8)
        bar_ax.set_xlim(0, 108)
        bar_ax.set_xlabel("200-Class Softmax Probability (%)", fontsize=8)
        bar_ax.set_title("Top-5 Predicted Species", fontsize=8.5, fontweight="bold")
        bar_ax.grid(axis="x", alpha=0.25)
        for bar, val in zip(bars, values):
            bar_ax.text(val + 1.2, bar.get_y() + bar.get_height() / 2, f"{val:.1f}%", va="center", fontsize=7.5)

        gallery_records.append({
            "criterion": criterion_label,
            "filename": row["name"],
            "correct": bool(is_correct),
            "true_label": true_id,
            "true_name": class_names[true_id],
            "true_prob": true_prob,
            "predicted_label": pred_id,
            "predicted_name": class_names[pred_id],
            "predicted_prob": top5[0]["prob"],
            "top5": top5,
        })

    fig.suptitle(title, fontsize=12, fontweight="bold")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    write_json(out_path.with_suffix(".json"), gallery_records)
    return gallery_records


def render_gradcam_comparison(examples, records_by_name, image_root, cp_path, model, config, class_names, device, out_path):
    """Render a 6-row x 4-column Grad-CAM interpretability figure comparing 3 correct and 3 incorrect validation examples."""
    fig, axes = plt.subplots(6, 4, figsize=(16, 16.5), layout="constrained")
    gradcam_dir = out_path.parent / "gradcam"
    gradcam_dir.mkdir(parents=True, exist_ok=True)

    for idx, (criterion_label, row) in enumerate(examples):
        rec = records_by_name[row["name"]]
        img_path = Path(image_root) / rec["relative_path"]
        crop_np, heat_rgb, overlay_np, top5, probs = compute_top5_and_cam(model, config, class_names, img_path, device)

        # Also run src.gradcam to save standalone two-panel artifact
        gradcam(cp_path, img_path, gradcam_dir / f"{idx+1:02d}_{Path(row['name']).stem}_gradcam.png")

        true_id = int(row["true_label"])
        pred_id = int(row["predicted_label"])
        is_correct = true_id == pred_id
        status = "CORRECT" if is_correct else "INCORRECT"

        axes[idx, 0].imshow(crop_np)
        axes[idx, 0].set_title(f"{status}: {row['name']}\n224x224 Evaluation Crop", fontsize=7.8, fontweight="bold")
        axes[idx, 1].imshow(heat_rgb)
        axes[idx, 1].set_title(f"layer4 Grad-CAM Heatmap\nTarget: [{pred_id:03d}] {class_names[pred_id]}", fontsize=8)
        axes[idx, 2].imshow(overlay_np)
        axes[idx, 2].set_title(
            f"Grad-CAM Overlay (Pred {top5[0]['prob']*100:.1f}%)\nTrue: [{true_id:03d}] {class_names[true_id]} ({probs[true_id]*100:.1f}%)",
            fontsize=8,
        )
        for col_i in range(3):
            axes[idx, col_i].set_xticks([])
            axes[idx, col_i].set_yticks([])

        bar_ax = axes[idx, 3]
        labels = [f"[{item['label']:03d}] {item['name']}" for item in reversed(top5)]
        values = [item["prob"] * 100 for item in reversed(top5)]
        colors = [
            "#2ca02c" if item["label"] == true_id else ("#d62728" if item["label"] == pred_id else "#7f7f7f")
            for item in reversed(top5)
        ]
        bars = bar_ax.barh(range(len(top5)), values, color=colors, height=0.6)
        bar_ax.set_yticks(range(len(top5)), labels, fontsize=7.5)
        bar_ax.set_xlim(0, 110)
        bar_ax.set_xlabel("Softmax (%)", fontsize=7.5)
        wrapped_crit = criterion_label.replace(": ", ":\n") if len(criterion_label) > 42 else criterion_label
        bar_ax.set_title(wrapped_crit, fontsize=7.8, fontweight="bold")
        bar_ax.grid(axis="x", alpha=0.25)
        for bar, val in zip(bars, values):
            bar_ax.text(val + 1.2, bar.get_y() + bar.get_height() / 2, f"{val:.1f}%", va="center", fontsize=7)

    fig.suptitle(
        "E3 ResNet18 layer4 Grad-CAM Attribution: Correct vs Incorrect Validation Predictions\n"
        "(Caution: Coarse 7x7 spatial attribution indicates salient regions for the predicted logit, not causal reasoning proof)",
        fontsize=11.5,
        fontweight="bold",
    )
    fig.savefig(out_path, dpi=200, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)


def main():
    cfg = data_config("configs/data.local.json")
    split = load_split(Path(cfg["prepared_dir"]) / "split.json")
    class_names = split["class_names"]
    device = choose_device("cuda")

    # =========================================================================
    # SECTION A: Model Comparison & Dropout Ablation Figures
    # =========================================================================
    models_info = [
        ("E1: Baseline CNN", "outputs/baseline_001", "#1f77b4"),
        ("E2: Regularised (Drop 0.4)", "outputs/experiment_001/regularized", "#ff7f0e"),
        ("E4: Ablation (Drop 0.0)", "outputs/experiment_001/E4_regularized_no_dropout", "#d62728"),
        ("E3: ResNet18 Transfer", "outputs/experiment_001/resnet18", "#2ca02c"),
    ]
    fig_dir = Path("outputs/experiment_001/figures")
    fig_dir.mkdir(parents=True, exist_ok=True)

    histories = {}
    summaries = {}
    for label, path, color in models_info:
        histories[label] = (pd.read_csv(Path(path) / "history.csv"), color)
        summaries[label] = read_json(Path(path) / "summary.json")

    # A1. Annotated 3-panel learning curves
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2), layout="constrained")
    for label, (h, col) in histories.items():
        s = summaries[label]
        b_ep = s["best_epoch"]
        b_row = h[h["epoch"] == b_ep].iloc[0]
        short = label.split(":")[0]
        axes[0].plot(h["epoch"], h["clean_train_loss"], color=col, linestyle="--", alpha=0.65, label=f"{short} Clean-Train")
        axes[0].plot(h["epoch"], h["val_loss"], color=col, linewidth=2, label=f"{short} Val (best ep {b_ep})")
        axes[0].scatter([b_ep], [b_row["val_loss"]], color=col, s=60, zorder=5, edgecolor="black")

        axes[1].plot(h["epoch"], h["clean_train_top1"] * 100, color=col, linestyle="--", alpha=0.65, label=f"{short} Clean-Train")
        axes[1].plot(h["epoch"], h["val_top1"] * 100, color=col, linewidth=2, label=f"{short} Val ({b_row['val_top1']*100:.2f}%)")
        axes[1].scatter([b_ep], [b_row["val_top1"] * 100], color=col, s=60, zorder=5, edgecolor="black")

        axes[2].plot(
            h["epoch"],
            h["generalisation_gap"] * 100,
            color=col,
            linewidth=2,
            label=f"{short} (ep {b_ep}: {b_row['generalisation_gap']*100:.2f}pp)",
        )
        axes[2].scatter([b_ep], [b_row["generalisation_gap"] * 100], color=col, s=60, zorder=5, edgecolor="black")

    axes[1].axvline(5.5, color="#2ca02c", linestyle=":", alpha=0.8)
    axes[1].text(6.2, 54, "E3 Stage 2:\nUnfreeze layer4", color="#1b6ca8", fontsize=8, fontweight="bold")
    axes[0].set(xlabel="Epoch", ylabel="Cross-Entropy Loss", title="(a) Clean-Train vs Validation Loss", ylim=(0, 10.8))
    axes[1].set(xlabel="Epoch", ylabel="Top-1 Accuracy (%)", title="(b) Clean-Train vs Validation Top-1 Accuracy", ylim=(0, 103))
    axes[2].set(xlabel="Epoch", ylabel="Generalisation Gap: Train - Val Top-1 (pp)", title="(c) Generalisation Gap Evolution", ylim=(-5, 78))
    for ax in axes:
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=7.5, loc="best")
    fig.suptitle(
        "COS30082 CUB-200 Development Split (3,870 Train / 959 Val): Learning Curves & Annotated Best Checkpoints",
        fontsize=12,
        fontweight="bold",
    )
    fig.savefig(fig_dir / "model_comparison.png", dpi=220, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)

    # A2. Four-model bar chart comparison
    fig, ax = plt.subplots(figsize=(10.5, 5), layout="constrained")
    x = np.arange(len(models_info))
    w = 0.2
    m_labels = [m[0] for m in models_info]
    v_top1 = [summaries[m]["validation"]["top1"] * 100 for m in m_labels]
    v_avg = [summaries[m]["validation"]["average_accuracy_per_class"] * 100 for m in m_labels]
    v_top5 = [summaries[m]["validation"]["top5"] * 100 for m in m_labels]
    gaps = [
        histories[m][0].loc[histories[m][0]["epoch"] == summaries[m]["best_epoch"], "generalisation_gap"].iloc[0] * 100
        for m in m_labels
    ]

    b1 = ax.bar(x - 1.5 * w, v_top1, w, label="Val Top-1 Accuracy (%)", color="#1f77b4")
    b2 = ax.bar(x - 0.5 * w, v_avg, w, label="Val Avg Accuracy / Class (%)", color="#2ca02c")
    b3 = ax.bar(x + 0.5 * w, v_top5, w, label="Val Top-5 Accuracy (%)", color="#9467bd")
    b4 = ax.bar(x + 1.5 * w, gaps, w, label="Best-Epoch Generalisation Gap (pp)", color="#ff7f0e")

    for bars in (b1, b2, b3, b4):
        for bar in bars:
            h_val = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h_val + 1.0,
                f"{h_val:.2f}",
                ha="center",
                va="bottom",
                fontsize=7.5,
                fontweight="bold",
            )

    ax.set_xticks(x, [f"{m}\n(Best Ep {summaries[m]['best_epoch']})" for m in m_labels], fontsize=9)
    ax.set_ylabel("Percentage (%) / Percentage Points (pp)")
    ax.set_ylim(0, 100)
    ax.set_title("Four-Model Validation Comparison on Frozen CUB-200 Split (Seed 42, 959 Validation Images)", fontsize=11.5, fontweight="bold")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="upper left", fontsize=8.5)
    fig.savefig(fig_dir / "four_model_bar_comparison.png", dpi=220, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)

    # A3. Annotated E2 vs E4 Dropout Ablation
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), layout="constrained")
    for label in ["E2: Regularised (Drop 0.4)", "E4: Ablation (Drop 0.0)"]:
        h, col = histories[label]
        s = summaries[label]
        b_ep = s["best_epoch"]
        b_row = h[h["epoch"] == b_ep].iloc[0]
        axes[0].plot(h["epoch"], h["clean_train_loss"], color=col, linestyle="--", alpha=0.7, label=f"{label} Clean-Train")
        axes[0].plot(h["epoch"], h["val_loss"], color=col, linewidth=2, label=f"{label} Val (Ep {b_ep}: {b_row['val_loss']:.2f})")
        axes[0].scatter([b_ep], [b_row["val_loss"]], color=col, s=65, zorder=5, edgecolor="black")

        axes[1].plot(
            h["epoch"],
            h["clean_train_top1"] * 100,
            color=col,
            linestyle="--",
            alpha=0.7,
            label=f"{label} Clean-Train ({b_row['clean_train_top1']*100:.2f}%)",
        )
        axes[1].plot(h["epoch"], h["val_top1"] * 100, color=col, linewidth=2, label=f"{label} Val ({b_row['val_top1']*100:.2f}%)")
        axes[1].scatter([b_ep], [b_row["val_top1"] * 100], color=col, s=65, zorder=5, edgecolor="black")

        axes[2].plot(
            h["epoch"],
            h["generalisation_gap"] * 100,
            color=col,
            linewidth=2,
            label=f"{label} Gap (Ep {b_ep}: {b_row['generalisation_gap']*100:.2f}pp)",
        )
        axes[2].scatter([b_ep], [b_row["generalisation_gap"] * 100], color=col, s=65, zorder=5, edgecolor="black")

    axes[0].set(xlabel="Epoch", ylabel="Cross-Entropy Loss", title="(a) Loss: Removing Dropout Lowers Train & Val Loss")
    axes[1].set(xlabel="Epoch", ylabel="Top-1 Accuracy (%)", title="(b) Top-1: E4 (+3.96pp Val) Relieves E2 Underfitting")
    axes[2].set(xlabel="Epoch", ylabel="Generalisation Gap (pp)", title="(c) Gap: 6.19pp (E2) -> 16.08pp (E4) vs 58.49pp (E1)")
    for ax in axes:
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)
    fig.suptitle("Controlled Dropout Ablation (E2 p=0.4 vs E4 p=0.0): Capacity vs Generalisation Trade-Off", fontsize=11.5, fontweight="bold")
    fig.savefig(fig_dir / "ablation_e2_vs_e4.png", dpi=220, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)

    # =========================================================================
    # SECTION B & C: E3 Validation Prediction Gallery & Grad-CAM Attribution
    # =========================================================================
    e3_cp_path = Path("outputs/experiment_001/resnet18/checkpoints/best.pt")
    e3_cp = load_checkpoint(e3_cp_path)
    e3_model = build_model(e3_cp["config"], pretrained=False).to(device).eval()
    e3_model.load_state_dict(e3_cp["state_dict"])

    val_pred_df = pd.read_csv("outputs/experiment_001/resnet18/validation/predictions.csv")
    val_pairs_df = pd.read_csv("outputs/experiment_001/resnet18/validation/confused_pairs.csv")
    val_records_by_name = {r["name"]: r for r in split["val"]}

    val_examples = select_deterministic_examples(val_pred_df, val_pairs_df)
    render_prediction_gallery(
        val_examples,
        val_records_by_name,
        cfg["train_root"],
        e3_model,
        e3_cp["config"],
        class_names,
        device,
        fig_dir / "val_prediction_gallery.png",
        "E3 ResNet18 Validation Prediction Gallery (3 Deterministic Correct vs 3 Deterministic Incorrect Examples)",
    )
    render_gradcam_comparison(
        val_examples,
        val_records_by_name,
        cfg["train_root"],
        e3_cp_path,
        e3_model,
        e3_cp["config"],
        class_names,
        device,
        fig_dir / "gradcam_val_comparison.png",
    )

    # =========================================================================
    # SECTION D: Final Held-Out Test Visualisations
    # =========================================================================
    test_dir = Path("outputs/final_test_results")
    test_fig_dir = test_dir / "figures"
    test_fig_dir.mkdir(parents=True, exist_ok=True)

    test_comp_df = pd.read_csv(test_dir / "test_comparison.csv")
    test_audit = read_json(test_dir / "test_audit.json")
    test_records_by_name = {r["name"]: r for r in test_audit["records"]}

    # D1. Validation vs Held-Out Test Metrics Summary Bar Chart
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.9), layout="constrained")
    run_labels = ["E1: Baseline CNN", "E2: Regularised (Drop 0.4)", "E3: ResNet18 (Selected)"]
    x3 = np.arange(3)
    w3 = 0.2
    t_top1 = test_comp_df["top1"].values * 100
    t_avg = test_comp_df["average_accuracy_per_class"].values * 100
    t_top5 = test_comp_df["top5"].values * 100
    t_f1 = test_comp_df["macro_f1"].values * 100

    tb1 = axes[0].bar(x3 - 1.5 * w3, t_top1, w3, label="Test Top-1 (%)", color="#1f77b4")
    tb2 = axes[0].bar(x3 - 0.5 * w3, t_avg, w3, label="Test Avg Accuracy / Class (%)", color="#2ca02c")
    tb3 = axes[0].bar(x3 + 0.5 * w3, t_top5, w3, label="Test Top-5 (%)", color="#9467bd")
    tb4 = axes[0].bar(x3 + 1.5 * w3, t_f1, w3, label="Test Macro-F1 (%)", color="#ff7f0e")
    for bars in (tb1, tb2, tb3, tb4):
        for bar in bars:
            v = bar.get_height()
            axes[0].text(bar.get_x() + bar.get_width() / 2, v + 1.0, f"{v:.2f}", ha="center", va="bottom", fontsize=7.5, fontweight="bold")
    axes[0].set_xticks(x3, run_labels, fontsize=9)
    axes[0].set_ylim(0, 100)
    axes[0].set_ylabel("Accuracy / Score (%)")
    axes[0].set_title("(a) Official Held-Out Test Set Performance (1,204 Images, 200 Classes)", fontsize=10, fontweight="bold")
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].legend(fontsize=8, loc="upper left")

    # Subplot 2: Validation vs Test Consistency for Selected E3 ResNet18
    e3_val = summaries["E3: ResNet18 Transfer"]["validation"]
    e3_test = read_json(test_dir / "resnet18" / "metrics.json")
    metric_keys = [("top1", "Top-1 Accuracy"), ("average_accuracy_per_class", "Avg Acc / Class"), ("top5", "Top-5 Accuracy"), ("macro_f1", "Macro-F1")]
    xm = np.arange(len(metric_keys))
    wm = 0.32
    val_vals = [e3_val[k] * 100 for k, _ in metric_keys]
    test_vals = [e3_test[k] * 100 for k, _ in metric_keys]
    vb = axes[1].bar(xm - wm / 2, val_vals, wm, label="Development Validation (N=959)", color="#2b8cbe")
    teb = axes[1].bar(xm + wm / 2, test_vals, wm, label="Official Held-Out Test (N=1,204)", color="#2ca02c")
    for bars in (vb, teb):
        for bar in bars:
            v = bar.get_height()
            axes[1].text(bar.get_x() + bar.get_width() / 2, v + 1.0, f"{v:.2f}%", ha="center", va="bottom", fontsize=8, fontweight="bold")
    axes[1].set_xticks(xm, [lbl for _, lbl in metric_keys], fontsize=9)
    axes[1].set_ylim(0, 100)
    axes[1].set_ylabel("Score (%)")
    axes[1].set_title("(b) Selected E3 ResNet18: Validation vs Held-Out Test Parity", fontsize=10, fontweight="bold")
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].legend(fontsize=8.5, loc="lower right")
    fig.suptitle("One-Time Held-Out Test Evaluation Summary (Pre-Committed Validation Winner: E3 ResNet18)", fontsize=12, fontweight="bold")
    fig.savefig(test_fig_dir / "test_metrics_summary.png", dpi=220, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)

    # D2. Per-Class Accuracy Distribution, Train-Count Relationship, and Top Confusions on Test Set
    test_per_class = pd.read_csv(test_dir / "resnet18" / "per_class.csv")
    class_counts_df = pd.read_csv(Path(cfg["prepared_dir"]) / "class_counts.csv")
    merged_cls = pd.merge(test_per_class, class_counts_df, on="label")
    test_pairs = pd.read_csv(test_dir / "resnet18" / "confused_pairs.csv")

    fig, axes = plt.subplots(1, 3, figsize=(17, 5), layout="constrained")
    acc_pct = merged_cls["accuracy"] * 100
    axes[0].hist(acc_pct, bins=10, range=(0, 100), color="#2ca02c", edgecolor="black", alpha=0.8)
    axes[0].axvline(acc_pct.mean(), color="#d62728", linestyle="--", linewidth=2, label=f"Mean (Avg/Class): {acc_pct.mean():.2f}%")
    axes[0].axvline(acc_pct.median(), color="#1f77b4", linestyle=":", linewidth=2, label=f"Median: {acc_pct.median():.2f}%")
    axes[0].set(
        xlabel="Per-Class Test Accuracy (%)",
        ylabel="Number of Bird Species (out of 200)",
        title=f"(a) Per-Class Test Accuracy Distribution\n(100% acc: {(acc_pct == 100).sum()} classes; <=25% acc: {(acc_pct <= 25).sum()} classes)",
    )
    axes[0].grid(alpha=0.25)
    axes[0].legend(fontsize=8)

    # Scatter of train support vs test accuracy
    r_corr = np.corrcoef(merged_cls["train"], acc_pct)[0, 1]
    axes[1].scatter(merged_cls["train"], acc_pct, alpha=0.55, color="#1f77b4", edgecolors="black", s=35)
    m_slope, b_int = np.polyfit(merged_cls["train"], acc_pct, 1)
    xs = np.linspace(merged_cls["train"].min(), merged_cls["train"].max(), 50)
    axes[1].plot(xs, m_slope * xs + b_int, color="#d62728", linewidth=2, label=f"Linear trend (Pearson r = {r_corr:.3f})")
    axes[1].set(
        xlabel="Development Training Images per Class (80% Split)",
        ylabel="Held-Out Test Accuracy (%)",
        title="(b) Per-Class Test Accuracy vs Training Sample Count\n(4-8 test images/class discretizes per-class recall)",
        ylim=(-5, 105),
    )
    axes[1].grid(alpha=0.25)
    axes[1].legend(fontsize=8, loc="lower right")

    # Top confusion pairs bar chart
    top_p = test_pairs.head(12).iloc[::-1]
    pair_labels = [f"{r['true']} -> {r['predicted']}" for _, r in top_p.iterrows()]
    axes[2].barh(range(len(top_p)), top_p["count"], color="#d95f02", edgecolor="black")
    axes[2].set_yticks(range(len(top_p)), pair_labels, fontsize=7.5)
    axes[2].set_xlabel("Misclassified Test Images (Count)")
    axes[2].set_title("(c) Top Held-Out Test Confusion Pairs (E3 ResNet18)", fontsize=10, fontweight="bold")
    axes[2].grid(axis="x", alpha=0.25)

    fig.suptitle("Held-Out Test Diagnostic Analysis: Class Distribution, Sample-Count Sensitivity & Fine-Grained Confusions", fontsize=11.5, fontweight="bold")
    fig.savefig(test_fig_dir / "test_per_class_and_confusions.png", dpi=220, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)

    # D3. Held-Out Test Prediction Gallery (3 Deterministic Correct + 3 Deterministic Incorrect)
    test_pred_df = pd.read_csv(test_dir / "resnet18" / "predictions.csv")
    test_examples = select_deterministic_examples(test_pred_df, test_pairs)
    render_prediction_gallery(
        test_examples,
        test_records_by_name,
        cfg["test_root"],
        e3_model,
        e3_cp["config"],
        class_names,
        device,
        test_fig_dir / "test_prediction_gallery.png",
        "E3 ResNet18 Held-Out Test Prediction Gallery (3 Deterministic Correct vs 3 Deterministic Incorrect Examples)",
    )
    print("All Section A, B, C, and D visual evidence generated successfully.")


if __name__ == "__main__":
    main()

