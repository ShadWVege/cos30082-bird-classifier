"""Evaluate Test-Time Augmentation (TTA) on the frozen validation split.

Learning objective:
Test-Time Augmentation (TTA) is an inference-only technique that averages predictions
across multiple transformed views of the same input image (e.g., standard center-crop
plus horizontally-flipped center-crop). Because birds exhibit bilateral symmetry,
horizontal flipping provides an authentic semantic invariant that can reduce variance
and margin errors without retraining or changing model weights.
"""
import argparse
import json
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(".").resolve()))

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
from torchvision import transforms as T

from src.dataset import MEAN, STD
from src.engine import load_checkpoint
from src.metrics import classification_metrics
from src.models import build_model
from src.utils import choose_device, data_config, load_split, read_json, seed_everything


def build_transforms(size=224):
    """Deterministic transforms for standard view and horizontal flip."""
    base = T.Compose([
        T.Resize(round(size * 256 / 224)),
        T.CenterCrop(size),
    ])
    to_tensor_norm = T.Compose([
        T.ToTensor(),
        T.Normalize(MEAN, STD),
    ])
    return base, to_tensor_norm


def evaluate_with_tta(model, records, root, device, num_classes, use_tta=False):
    """Run inference over records with optional horizontal-flip TTA."""
    model.eval()
    base_crop, to_tensor_norm = build_transforms(224)
    targets, probabilities = [], []
    latencies = []

    with torch.no_grad():
        for row in records:
            img_path = Path(root) / row["relative_path"]
            with Image.open(img_path) as im:
                image = ImageOps.exif_transpose(im).convert("RGB")

            t0 = time.perf_counter()
            cropped = base_crop(image)
            t_orig = to_tensor_norm(cropped).unsqueeze(0).to(device)

            if use_tta:
                # 2-view TTA: original center-crop + horizontal flip
                flipped = cropped.transpose(Image.FLIP_LEFT_RIGHT)
                t_flip = to_tensor_norm(flipped).unsqueeze(0).to(device)
                batch = torch.cat([t_orig, t_flip], dim=0)
                logits = model(batch)
                probs = F.softmax(logits, dim=-1).mean(dim=0, keepdim=True)
            else:
                logits = model(t_orig)
                probs = F.softmax(logits, dim=-1)

            dt = time.perf_counter() - t0
            latencies.append(dt)

            targets.append(row["label"])
            probabilities.append(probs.cpu().numpy()[0])

    probs_arr = np.array(probabilities)
    scores, per_class, cm = classification_metrics(targets, probs_arr, num_classes)
    scores["mean_latency_ms"] = float(np.mean(latencies) * 1000.0)
    scores["p95_latency_ms"] = float(np.percentile(latencies, 95) * 1000.0)
    scores["total_time_sec"] = float(np.sum(latencies))
    return scores, per_class


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint .pt")
    parser.add_argument("--data", default="configs/data.local.json")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out", default=None, help="Output JSON path")
    args = parser.parse_args()

    seed_everything(42)
    device = choose_device(args.device)
    data = data_config(args.data)
    split = load_split(Path(data["prepared_dir"]) / "split.json")
    ckpt = load_checkpoint(args.checkpoint)

    config = ckpt["config"]
    model = build_model(config, pretrained=False).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    num_classes = config["num_classes"]
    val_records = split["val"]

    print(f"Evaluating {args.checkpoint} on {len(val_records)} validation images...")
    print("1. Running standard single-view inference...")
    scores_single, _ = evaluate_with_tta(model, val_records, data["train_root"], device, num_classes, use_tta=False)
    print(f"   Single-view: Top-1 = {scores_single['top1']:.4%}, Avg/Class = {scores_single['average_accuracy_per_class']:.4%}, Latency = {scores_single['mean_latency_ms']:.2f} ms/image")

    print("2. Running 2-view Horizontal-Flip TTA inference...")
    scores_tta, _ = evaluate_with_tta(model, val_records, data["train_root"], device, num_classes, use_tta=True)
    print(f"   2-View TTA:  Top-1 = {scores_tta['top1']:.4%}, Avg/Class = {scores_tta['average_accuracy_per_class']:.4%}, Latency = {scores_tta['mean_latency_ms']:.2f} ms/image")

    diff_top1 = scores_tta["top1"] - scores_single["top1"]
    diff_avg = scores_tta["average_accuracy_per_class"] - scores_single["average_accuracy_per_class"]
    print(f"\nDelta with 2-View TTA: Top-1 = {diff_top1:+.4%} ({diff_top1 * 100:+.2f} pp), Avg/Class = {diff_avg:+.4%} ({diff_avg * 100:+.2f} pp)")

    result = {
        "checkpoint": str(args.checkpoint),
        "device": str(device),
        "validation_samples": len(val_records),
        "single_view": scores_single,
        "two_view_tta": scores_tta,
        "delta": {
            "top1_pp": (scores_tta["top1"] - scores_single["top1"]) * 100,
            "average_accuracy_per_class_pp": (scores_tta["average_accuracy_per_class"] - scores_single["average_accuracy_per_class"]) * 100,
            "top5_pp": (scores_tta["top5"] - scores_single["top5"]) * 100,
            "macro_f1_pp": (scores_tta["macro_f1"] - scores_single["macro_f1"]) * 100,
            "latency_multiplier": scores_tta["mean_latency_ms"] / scores_single["mean_latency_ms"]
        }
    }

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"Saved comparison to {out_path}")

    return result


if __name__ == "__main__":
    main()

