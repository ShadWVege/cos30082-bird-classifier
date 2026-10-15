"""Official one-time held-out test evaluation of the frozen E7 ConvNeXt-Tiny model.

Learning objective:
Evaluates the frozen E7 checkpoint on the 1,204 official CUB-200 test images.
Discloses post-test exploratory status (E3 test diagnostics were observed earlier).
Computes single-view and bilateral horizontal-flip TTA metrics under the exact
frozen evaluation transform protocol.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.data import audit_images, read_manifest, verify_files
from src.dataset import BirdDataset
from src.engine import load_checkpoint
from src.metrics import classification_metrics
from src.models import build_model
from src.utils import choose_device, data_config, digest, load_split, read_json, seed_worker, write_csv, write_json


EXPECTED_E7_HASH = "18cfc800908652c7b055e7a13e4369f995ec4a91b4ae4b1b7f134f4722d015b2"
EXPECTED_SPLIT_FINGERPRINT = "55f05f6ac27722f9ca3de2b60a6311117b1bcb0b679220aa739c8ff11c9c6033"


def evaluate_e7(data_config_path="configs/data.local.json",
                checkpoint_path="outputs/experiment_001/E7_convnext_tiny/checkpoints/best.pt",
                out_dir="outputs/final_test_results/E7_convnext_tiny",
                device_str="auto"):
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print("=== PHASE A: FREEZE AND VERIFY E7 ===")
    ckpt_file = Path(checkpoint_path)
    if not ckpt_file.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_file}")
    
    ckpt_bytes = ckpt_file.read_bytes()
    ckpt_hash = hashlib.sha256(ckpt_bytes).hexdigest()
    print(f"Checkpoint SHA-256: {ckpt_hash}")
    if ckpt_hash != EXPECTED_E7_HASH:
        raise ValueError(f"Checkpoint hash mismatch! Expected {EXPECTED_E7_HASH}, got {ckpt_hash}")
    print("Checkpoint SHA-256 verification: PASSED")

    checkpoint = load_checkpoint(str(ckpt_file))
    print(f"Checkpoint Epoch: {checkpoint.get('epoch')}")
    print(f"Checkpoint Stage: {checkpoint.get('stage')}")
    print(f"Number of classes in checkpoint: {len(checkpoint.get('class_names', []))}")
    if len(checkpoint.get("class_names", [])) != 200:
        raise ValueError("Expected exactly 200 classes in checkpoint")

    cfg = data_config(data_config_path)
    split = load_split(Path(cfg["prepared_dir"]) / "split.json")
    if split["fingerprint"] != EXPECTED_SPLIT_FINGERPRINT:
        raise ValueError("Split fingerprint mismatch!")
    if checkpoint.get("split_fingerprint") != EXPECTED_SPLIT_FINGERPRINT:
        raise ValueError("Checkpoint split fingerprint mismatch!")
    print("Split fingerprint verification: PASSED")

    # Read test manifest
    records = read_manifest(cfg["test_manifest"], cfg["num_classes"])
    print(f"Test manifest records read: {len(records)}")
    if len(records) != 1204:
        raise ValueError(f"Expected exactly 1204 test records, got {len(records)}")

    # Audit test images
    audit = audit_images(cfg["test_root"], records, cfg["num_classes"])
    if audit["errors"]:
        raise ValueError(f"Test audit found errors: {audit['errors']}")
    print(f"Test image audit: 1204/1204 images verified successfully.")

    # Build model and load state_dict
    device = choose_device(device_str)
    print(f"Evaluation device: {device}")
    model = build_model(checkpoint["config"], pretrained=False).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    # Create evaluation dataset and dataloader using standard shared evaluation transform
    test_dataset = BirdDataset(cfg["test_root"], audit["records"], checkpoint["config"]["image_size"], augment=False)
    generator = torch.Generator().manual_seed(checkpoint["config"]["seed"])
    test_loader = DataLoader(
        test_dataset,
        batch_size=checkpoint["config"]["batch_size"],
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
        worker_init_fn=seed_worker,
        generator=generator
    )

    print("\n=== PHASE B: OFFICIAL TEST EVALUATION (1,204 IMAGES) ===")
    single_loss_sum = 0.0
    tta_loss_sum = 0.0
    total_samples = 0

    all_targets = []
    all_names = []
    all_single_logits = []
    all_single_probs = []
    all_tta_probs = []

    t0_eval = time.perf_counter()

    with torch.no_grad():
        for batch_idx, (images, labels, batch_names) in enumerate(test_loader):
            images = images.to(device)
            labels = labels.to(device)
            batch_sz = len(labels)

            # 1. Single-view inference
            logits_sv = model(images).float()
            probs_sv = F.softmax(logits_sv, dim=-1)
            loss_sv = F.cross_entropy(logits_sv, labels, reduction="sum")
            single_loss_sum += loss_sv.item()

            # 2. Horizontal-flip TTA inference
            images_flip = torch.flip(images, dims=[-1])
            logits_flip = model(images_flip).float()
            probs_flip = F.softmax(logits_flip, dim=-1)
            probs_tta = 0.5 * (probs_sv + probs_flip)
            
            # Loss for TTA using log-probabilities
            log_probs_tta = torch.log(probs_tta.clamp(min=1e-12))
            loss_tta = F.nll_loss(log_probs_tta, labels, reduction="sum")
            tta_loss_sum += loss_tta.item()

            total_samples += batch_sz
            all_targets.extend(labels.cpu().tolist())
            all_names.extend(batch_names)
            all_single_logits.append(logits_sv.cpu().numpy())
            all_single_probs.append(probs_sv.cpu().numpy())
            all_tta_probs.append(probs_tta.cpu().numpy())

    eval_duration = time.perf_counter() - t0_eval
    print(f"Evaluated {total_samples} test images in {eval_duration:.2f} seconds.")

    targets_arr = np.array(all_targets, dtype=int)
    single_logits_arr = np.concatenate(all_single_logits, axis=0)
    single_probs_arr = np.concatenate(all_single_probs, axis=0)
    tta_probs_arr = np.concatenate(all_tta_probs, axis=0)

    # Compute Single-View Metrics
    sv_scores, sv_per_class, sv_confusion = classification_metrics(targets_arr, single_probs_arr, cfg["num_classes"])
    sv_scores["loss"] = single_loss_sum / total_samples

    # Compute TTA Metrics
    tta_scores, tta_per_class, tta_confusion = classification_metrics(targets_arr, tta_probs_arr, cfg["num_classes"])
    tta_scores["loss"] = tta_loss_sum / total_samples

    print("\n--- OFFICIAL E7 SINGLE-VIEW TEST RESULTS ---")
    print(f"Top-1 Accuracy:                 {sv_scores['top1'] * 100.0:.2f}% ({sv_scores['top1']:.6f})")
    print(f"Average Accuracy/Class:         {sv_scores['average_accuracy_per_class'] * 100.0:.2f}% ({sv_scores['average_accuracy_per_class']:.6f})")
    print(f"Top-5 Accuracy:                 {sv_scores['top5'] * 100.0:.2f}% ({sv_scores['top5']:.6f})")
    print(f"Macro-F1:                       {sv_scores['macro_f1'] * 100.0:.2f}% ({sv_scores['macro_f1']:.6f})")
    print(f"Cross-Entropy Loss:             {sv_scores['loss']:.6f}")

    print("\n--- OFFICIAL E7 TWO-VIEW TTA TEST RESULTS ---")
    print(f"Top-1 Accuracy (TTA):           {tta_scores['top1'] * 100.0:.2f}% ({tta_scores['top1']:.6f})")
    print(f"Average Accuracy/Class (TTA):   {tta_scores['average_accuracy_per_class'] * 100.0:.2f}% ({tta_scores['average_accuracy_per_class']:.6f})")
    print(f"Top-5 Accuracy (TTA):           {tta_scores['top5'] * 100.0:.2f}% ({tta_scores['top5']:.6f})")
    print(f"Macro-F1 (TTA):                 {tta_scores['macro_f1'] * 100.0:.2f}% ({tta_scores['macro_f1']:.6f})")
    print(f"Cross-Entropy Loss (TTA):       {tta_scores['loss']:.6f}")

    # Save detailed JSON metrics
    write_json(out_path / "test_metrics_single_view.json", sv_scores)
    write_json(out_path / "test_metrics_tta.json", tta_scores)

    # Save predictions CSVs
    class_names = checkpoint["class_names"]
    pred_sv_records = []
    top5_sv = np.argsort(-single_probs_arr, axis=1)[:, :5]
    for i, name in enumerate(all_names):
        true_lbl = targets_arr[i]
        pred_lbl = int(np.argmax(single_probs_arr[i]))
        prob_val = float(single_probs_arr[i, pred_lbl])
        top5_classes = [class_names[idx] for idx in top5_sv[i]]
        pred_sv_records.append({
            "image_name": name,
            "true_label": true_lbl,
            "true_name": class_names[true_lbl],
            "pred_label": pred_lbl,
            "pred_name": class_names[pred_lbl],
            "confidence": prob_val,
            "correct": int(true_lbl == pred_lbl),
            "top5_predicted": "; ".join(top5_classes)
        })
    write_csv(out_path / "predictions_single_view.csv", pred_sv_records)

    pred_tta_records = []
    top5_tta = np.argsort(-tta_probs_arr, axis=1)[:, :5]
    for i, name in enumerate(all_names):
        true_lbl = targets_arr[i]
        pred_lbl = int(np.argmax(tta_probs_arr[i]))
        prob_val = float(tta_probs_arr[i, pred_lbl])
        top5_classes = [class_names[idx] for idx in top5_tta[i]]
        pred_tta_records.append({
            "image_name": name,
            "true_label": true_lbl,
            "true_name": class_names[true_lbl],
            "pred_label": pred_lbl,
            "pred_name": class_names[pred_lbl],
            "confidence": prob_val,
            "correct": int(true_lbl == pred_lbl),
            "top5_predicted": "; ".join(top5_classes)
        })
    write_csv(out_path / "predictions_tta.csv", pred_tta_records)

    # Save per-class metrics
    per_class_table = []
    for row_sv, row_tta in zip(sv_per_class, tta_per_class):
        lbl = row_sv["label"]
        per_class_table.append({
            "label": lbl,
            "class_name": class_names[lbl],
            "support": row_sv["support"],
            "sv_correct": row_sv["correct"],
            "sv_accuracy": row_sv["accuracy"],
            "sv_precision": row_sv["precision"],
            "sv_f1": row_sv["f1"],
            "tta_correct": row_tta["correct"],
            "tta_accuracy": row_tta["accuracy"],
            "tta_precision": row_tta["precision"],
            "tta_f1": row_tta["f1"],
        })
    write_csv(out_path / "per_class_metrics.csv", per_class_table)

    # Save confusion matrices
    np.save(out_path / "confusion_matrix_single_view.npy", sv_confusion)
    np.save(out_path / "confusion_matrix_tta.npy", tta_confusion)

    # Provenance record
    provenance = {
        "model": "convnext_tiny",
        "experiment_id": "E7_convnext_tiny",
        "checkpoint_sha256": ckpt_hash,
        "checkpoint_epoch": checkpoint.get("epoch"),
        "evaluation_timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "test_manifest_sha256": digest(cfg["test_manifest"]),
        "split_fingerprint": split["fingerprint"],
        "num_test_samples": total_samples,
        "evaluation_duration_sec": round(eval_duration, 2),
        "post_test_exploratory_status": True,
        "notes": "Post-test exploratory evaluation; E3 official test diagnostics were examined earlier during development. No tuning was performed."
    }
    write_json(out_path / "provenance.json", provenance)

    # Save consolidated test comparison table
    comparison_table = [
        {"run": "baseline_001", "model": "custom_cnn", "top1": 0.210133, "average_accuracy_per_class": 0.206256, "top5": 0.420266, "macro_f1": 0.199072, "loss": 6.731989},
        {"run": "regularized", "model": "custom_cnn_reg", "top1": 0.089701, "average_accuracy_per_class": 0.087958, "top5": 0.253322, "macro_f1": 0.070001, "loss": 4.412919},
        {"run": "resnet18_e3", "model": "resnet18", "top1": 0.704319, "average_accuracy_per_class": 0.701179, "top5": 0.891196, "macro_f1": 0.698970, "loss": 1.200356},
        {"run": "convnext_tiny_e7", "model": "convnext_tiny", "top1": sv_scores["top1"], "average_accuracy_per_class": sv_scores["average_accuracy_per_class"], "top5": sv_scores["top5"], "macro_f1": sv_scores["macro_f1"], "loss": sv_scores["loss"]},
        {"run": "convnext_tiny_e7_tta", "model": "convnext_tiny_tta", "top1": tta_scores["top1"], "average_accuracy_per_class": tta_scores["average_accuracy_per_class"], "top5": tta_scores["top5"], "macro_f1": tta_scores["macro_f1"], "loss": tta_scores["loss"]},
    ]
    write_csv(out_path / "test_comparison.csv", comparison_table)
    print(f"\nEvaluation completed cleanly. Results saved to {out_path.resolve()}")

    return sv_scores, tta_scores


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", default="configs/data.local.json")
    p.add_argument("--checkpoint", default="outputs/experiment_001/E7_convnext_tiny/checkpoints/best.pt")
    p.add_argument("--out", default="outputs/final_test_results/E7_convnext_tiny")
    p.add_argument("--device", default="auto")
    args = p.parse_args()
    evaluate_e7(args.data, args.checkpoint, args.out, args.device)
