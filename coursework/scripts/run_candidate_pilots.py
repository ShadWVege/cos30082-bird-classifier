"""Short frozen-backbone/head-training pilots for ConvNeXt-Tiny and EfficientNetV2-S.

Learning Objective & Design Choice:
Before committing to full 35-epoch fine-tuning on our RTX 3060 Laptop GPU (6 GB VRAM),
we execute a controlled 3-epoch head warmup pilot for both candidate architectures.
This establishes empirical baselines for:
1. Exact CUDA VRAM headroom under automatic mixed precision (AMP).
2. Throughput (seconds per epoch) on real training and validation sets.
3. Early feature representation quality: how linearly separable are the pretrained
   features on CUB-200 species before any backbone weights are altered?
"""
import json
import os
import platform
import sys
import time
from pathlib import Path

# Ensure project root is on sys.path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from src.data import read_manifest, infer_class_names
from src.dataset import BirdDataset
from src.engine import make_loader, run_epoch, rng_state
from src.models import build_model, set_stage, training_mode
from src.utils import ROOT, choose_device, data_config, load_split, read_json, seed_everything, write_json


def run_model_pilot(model_name, epochs=3, batch_size=32, image_size=224, lr=1e-3, amp=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_everything(42)

    data = data_config("configs/data.local.json")
    split = load_split(Path(data["prepared_dir"]) / "split.json")
    class_names = read_json(Path(data["prepared_dir"]) / "class_names.json")

    out_dir = ROOT / "outputs" / "experiment_001" / "pilots" / model_name
    out_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "model": model_name,
        "num_classes": len(class_names),
        "image_size": image_size,
        "batch_size": batch_size,
        "lr": lr,
        "weight_decay": 1e-4,
        "amp": amp,
        "num_workers": 0,
        "seed": 42,
        "augment": True,
    }

    # Reset CUDA memory stats
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    print(f"\n{'='*70}\nStarting Pilot for: {model_name} (Resolution: {image_size}x{image_size}, Batch: {batch_size}, AMP: {amp})\n{'='*70}")

    model = build_model(config, pretrained=True).to(device)
    set_stage(model, model_name, "head")
    training_mode(model, model_name, "head")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params

    print(f"Total Parameters:     {total_params:,}")
    print(f"Trainable (Head):     {trainable_params:,}")
    print(f"Frozen (Backbone):    {frozen_params:,}")

    loaders = {
        "train": make_loader(data["train_root"], split["train"], config, augment=True, shuffle=True),
        "clean_train": make_loader(data["train_root"], split["train"], config, augment=False, shuffle=False),
        "val": make_loader(data["train_root"], split["val"], config, augment=False, shuffle=False),
    }

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
        weight_decay=1e-4,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=amp and device.type == "cuda")

    history = []
    t_start = time.monotonic()

    for epoch in range(1, epochs + 1):
        t_epoch_start = time.monotonic()
        training_mode(model, model_name, "head")

        train_res = run_epoch(
            model, loaders["train"], device, config["num_classes"],
            optimizer=optimizer, scaler=scaler, amp=amp, label_smoothing=0.0
        )

        model.eval()
        clean_res = run_epoch(
            model, loaders["clean_train"], device, config["num_classes"],
            optimizer=None, scaler=None, amp=amp, label_smoothing=0.0
        )
        val_res = run_epoch(
            model, loaders["val"], device, config["num_classes"],
            optimizer=None, scaler=None, amp=amp, label_smoothing=0.0
        )

        epoch_time = time.monotonic() - t_epoch_start
        peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 ** 2) if device.type == "cuda" else 0.0

        row = {
            "epoch": epoch,
            "seconds": round(epoch_time, 2),
            "peak_vram_mb": round(peak_vram_mb, 1),
            "train_loss": round(train_res["metrics"]["loss"], 4),
            "train_top1": round(train_res["metrics"]["top1"], 4),
            "clean_train_loss": round(clean_res["metrics"]["loss"], 4),
            "clean_train_top1": round(clean_res["metrics"]["top1"], 4),
            "val_loss": round(val_res["metrics"]["loss"], 4),
            "val_top1": round(val_res["metrics"]["top1"], 4),
            "val_average_accuracy_per_class": round(val_res["metrics"]["average_accuracy_per_class"], 4),
            "val_top5": round(val_res["metrics"]["top5"], 4),
            "val_macro_f1": round(val_res["metrics"]["macro_f1"], 4),
        }
        history.append(row)

        print(
            f"Epoch {epoch}/{epochs} ({epoch_time:.1f}s, Peak VRAM: {peak_vram_mb:.0f} MB): "
            f"Train Top-1: {row['train_top1']*100:.2f}% | "
            f"Clean Train Top-1: {row['clean_train_top1']*100:.2f}% | "
            f"Val Top-1: {row['val_top1']*100:.2f}% | "
            f"Val Avg/Class: {row['val_average_accuracy_per_class']*100:.2f}% | "
            f"Val Top-5: {row['val_top5']*100:.2f}% | "
            f"Val Loss: {row['val_loss']:.4f}"
        )

    total_time = time.monotonic() - t_start

    # Save checkpoint
    ckpt_path = out_dir / "head_best.pt"
    torch.save({
        "model": model_name,
        "config": config,
        "state_dict": model.state_dict(),
        "history": history,
        "split_fingerprint": split["fingerprint"],
        "class_names": class_names,
    }, ckpt_path)

    summary = {
        "model": model_name,
        "image_size": image_size,
        "batch_size": batch_size,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "frozen_params": frozen_params,
        "checkpoint_size_mb": round(ckpt_path.stat().st_size / (1024 ** 2), 2),
        "total_pilot_seconds": round(total_time, 2),
        "mean_seconds_per_epoch": round(sum(r["seconds"] for r in history) / len(history), 2),
        "peak_vram_mb": round(max(r["peak_vram_mb"] for r in history), 1),
        "best_epoch": int(np.argmax([r["val_average_accuracy_per_class"] for r in history]) + 1),
        "best_val_top1": max(r["val_top1"] for r in history),
        "best_val_average_accuracy_per_class": max(r["val_average_accuracy_per_class"] for r in history),
        "best_val_top5": max(r["val_top5"] for r in history),
        "final_epoch_metrics": history[-1],
        "history": history,
    }

    write_json(out_dir / "pilot_summary.json", summary)
    return summary


def main():
    print("Executing Phase D Candidate Architecture Pilots on RTX 3060 Laptop GPU...")

    results = {}

    # Pilot 1: ConvNeXt-Tiny (224x224)
    results["convnext_tiny"] = run_model_pilot(
        "convnext_tiny", epochs=3, batch_size=32, image_size=224, lr=1e-3, amp=True
    )

    # Pilot 2: EfficientNetV2-S (224x224)
    results["efficientnet_v2_s"] = run_model_pilot(
        "efficientnet_v2_s", epochs=3, batch_size=32, image_size=224, lr=1e-3, amp=True
    )

    # Save overall comparison
    comp_file = ROOT / "outputs" / "experiment_001" / "pilots" / "candidate_pilot_comparison.json"
    write_json(comp_file, results)
    print(f"\nAll candidate pilots finished. Comprehensive results saved to {comp_file}")


if __name__ == "__main__":
    main()
