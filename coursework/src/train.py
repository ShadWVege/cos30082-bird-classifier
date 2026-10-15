"""Training orchestrator enforcing frozen split protocol and staged transfer learning.

Learning Objectives & Strict Methodological Protocol:
1. Frozen Development Split: Training and model selection use ONLY the pre-committed
   training (3,870 images) and validation (959 images) partitions. The official held-out
   test set (data/test.txt) is NEVER accessed during training, tuning, or checkpoint selection.
2. Three-DataLoader Protocol:
   - 'train': Training images with stochastic augmentations (crop, flip, rotation, jitter)
     and seeded random batch shuffling.
   - 'clean_train': Identical training images evaluated under DETERMINISTIC evaluation
     transforms without augmentations or shuffling. This isolates true training fit from
     augmentation noise, allowing exact calculation of the generalization gap:
     generalization_gap = clean_train_top1 - val_top1.
   - 'val': Held-out validation images under deterministic evaluation transforms.
3. Staged Transfer Learning:
   - Stage 1 ('head'): Freezes backbone representations; trains only the newly initialized
     200-class classification head for `head_epochs` (e.g., 5 epochs) to avoid gradient shock.
   - Stage Transition: Reloads the best Stage 1 checkpoint before commencing Stage 2 fine-tuning.
   - Stage 2 ('finetune'): Unfreezes terminal convolutional blocks (layer4 in ResNet18,
     features.7 in ConvNeXt) using discriminative learning rates.
4. Model Selection Contract:
   - Primary selection metric: Validation Average Accuracy per Class (macro recall over 200 classes).
   - Secondary tie-breaker: Validation cross-entropy loss.
   - Early stopping: Halts fine-tuning after `patience` (8) consecutive non-improving epochs.
"""
import argparse
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import torch
import torchvision

from .data import verify_files
from .engine import make_loader, run_epoch, rng_state, restore_rng, save_checkpoint, load_checkpoint
from .models import build_model, set_stage, training_mode
from .plots import learning_curves, save_analysis
from .utils import ROOT, choose_device, data_config, digest, load_split, read_json, seed_everything, write_csv, write_json


def train(args):
    """Execute complete staged training, validation monitoring, and checkpoint selection."""
    config = read_json(args.config)
    data = data_config(args.data)
    split = load_split(Path(data["prepared_dir"]) / "split.json")
    if config["num_classes"] != split["num_classes"]:
        raise ValueError("Model output size and split class count differ")
    if digest(data["train_manifest"]) != split["manifest_sha256"]:
        raise ValueError("Training manifest changed since split preparation")
    if args.epochs is not None:
        config["epochs"] = args.epochs
    if args.batch_size is not None:
        config["batch_size"] = args.batch_size
    if config["epochs"] < 1 or config["image_size"] < 32:
        raise ValueError("Need at least one epoch and image_size >= 32")
    if args.max_batches is not None and args.max_batches < 1:
        raise ValueError("max-batches must be positive")
    is_staged = config["model"] in ("resnet18", "convnext_tiny", "convnext_small", "efficientnet_v2_s")
    if is_staged and not args.max_batches and config["epochs"] <= config.get("head_epochs", 0) and config.get("head_epochs", 0) > 0:
        raise ValueError("Full transfer run must include the fine-tuning stage")
    config["smoke_test"] = args.max_batches is not None
    config["max_batches"] = args.max_batches
    out = Path(args.out)
    if out.exists() and any(out.iterdir()) and not args.resume:
        raise FileExistsError(f"Use a new run directory or --resume: {out}")
    out.mkdir(parents=True, exist_ok=True)
    checkpoints = out / "checkpoints"
    checkpoints.mkdir(exist_ok=True)
    previous = load_checkpoint(checkpoints / "last.pt") if args.resume else None
    if previous and (previous["config"] != config or previous["split_fingerprint"] != split["fingerprint"]):
        raise ValueError("Resume needs the same configuration and split")
    if previous and previous.get("complete") and (out / "summary.json").exists():
        raise ValueError("This run has already completed; start a new run for new experiments")
    seed_everything(config["seed"])
    device = choose_device(args.device)
    verify_files(data["train_root"], split["train"] + split["val"])
    # Cache any ImageNet download in this project, not a hidden global folder.
    os.environ.setdefault("TORCH_HOME", str(ROOT / ".cache" / "torch"))
    model = build_model(config, pretrained=previous is None).to(device)
    (out / "architecture.txt").write_text(str(model) + f"\n\nTotal parameters: {sum(p.numel() for p in model.parameters()):,}\n", encoding="utf-8")
    # Three-DataLoader split: augmented training, clean deterministic training, deterministic validation
    loaders = {"train": make_loader(data["train_root"], split["train"], config, config["augment"], True),
               "clean_train": make_loader(data["train_root"], split["train"], config),
               "val": make_loader(data["train_root"], split["val"], config)}
    scaler = torch.amp.GradScaler("cuda", enabled=config["amp"] and device.type == "cuda")
    history, best_score, best_loss, bad_epochs = [], -1.0, float("inf"), 0
    stage_best_score, stage_best_loss = -1.0, float("inf")
    start_epoch, current_stage = 0, None
    if previous:
        model.load_state_dict(previous["state_dict"])
        history, best_score, best_loss = previous["history"], previous["best_score"], previous["best_loss"]
        bad_epochs, start_epoch = previous["bad_epochs"], previous["epoch"]
        stage_best_score, stage_best_loss = previous["stage_best_score"], previous["stage_best_loss"]
        scaler.load_state_dict(previous["scaler"])
        restore_rng(previous["rng"], loaders["train"])
    write_json(out / "config.json", config)
    write_json(out / "environment.json", {"python": sys.version, "platform": platform.platform(),
               "torch": str(torch.__version__), "torchvision": str(torchvision.__version__),
               "device": str(device), "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
               "split_fingerprint": split["fingerprint"],
               "source_sha256": {p.name: digest(p) for p in (ROOT / "src").glob("*.py")}})
    freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True)
    (out / "requirements-frozen.txt").write_text(freeze.stdout, encoding="utf-8")
    for epoch in range(start_epoch, config["epochs"]):
        started = time.monotonic()
        stage = ("head" if epoch < config.get("head_epochs", 0) else "finetune") if is_staged else "full"
        finetune_blocks = tuple(config.get("finetune_blocks", ["layer4"]))
        freeze_bn = bool(config.get("freeze_bn", False))
        if stage != current_stage:
            # Stage transition gate: Begin fine-tuning from the best warmup checkpoint,
            # rather than an arbitrary noisy state at the final warmup epoch.
            if current_stage == "head" and stage == "finetune":
                model.load_state_dict(load_checkpoint(checkpoints / "best.pt")["state_dict"])
            set_stage(model, config["model"], stage, finetune_blocks)
            lr = config["finetune_lr"] if stage == "finetune" else config["lr"]
            optimizer_type = torch.optim.Adam if config["model"] == "baseline" else torch.optim.AdamW
            if stage == "finetune" and "discriminative_lr" in config:
                # Discriminative Learning Rates: Deeper backbone blocks extract task-general
                # representations and require lower learning rates (e.g. 1e-5 or 5e-5), while
                # the newly attached classifier requires higher learning rates (e.g. 2e-4).
                disc_cfg = config["discriminative_lr"]
                param_groups = []
                assigned_params = set()
                for block_name, block_lr in disc_cfg.items():
                    prefix = f"{block_name}."
                    params = [p for n, p in model.named_parameters() if n.startswith(prefix) and p.requires_grad]
                    if params:
                        param_groups.append({"params": params, "lr": block_lr, "weight_decay": config["weight_decay"]})
                        assigned_params.update(params)
                other_params = [p for p in model.parameters() if p.requires_grad and p not in assigned_params]
                if other_params:
                    param_groups.append({"params": other_params, "lr": config["finetune_lr"], "weight_decay": config["weight_decay"]})
                optimizer = optimizer_type(param_groups)
            else:
                optimizer = optimizer_type([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=config["weight_decay"])

            # Learning Rate Scheduler: Reduce learning rate by factor (0.5) when validation
            # loss fails to improve for `patience` (3) epochs.
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                patience=config.get("scheduler_patience", 3),
                factor=config.get("scheduler_factor", 0.5)
            ) if config["scheduler"] else None
            if previous and epoch == start_epoch and previous["stage"] == stage:
                optimizer.load_state_dict(previous["optimizer"])
                if scheduler:
                    scheduler.load_state_dict(previous["scheduler"])
            else:
                bad_epochs = 0
                stage_best_score, stage_best_loss = -1.0, float("inf")
                if previous and epoch == start_epoch and previous["stage"] == "head" and stage == "finetune":
                    model.load_state_dict(load_checkpoint(checkpoints / "best.pt")["state_dict"])
            current_stage = stage
        training_mode(model, config["model"], stage, finetune_blocks, freeze_bn)
        used_lr = optimizer.param_groups[-1]["lr"]
        train_result = run_epoch(model, loaders["train"], device, config["num_classes"], optimizer, scaler,
                                 config["amp"], args.max_batches, config.get("label_smoothing", 0.0))
        model.eval()
        clean_result = run_epoch(model, loaders["clean_train"], device, config["num_classes"], max_batches=args.max_batches)
        val_result = run_epoch(model, loaders["val"], device, config["num_classes"], max_batches=args.max_batches)

        # Model Selection Rule: Choose checkpoints based on validation Average Accuracy per Class
        # (macro recall), breaking ties with validation cross-entropy loss.
        score, loss = val_result["metrics"]["average_accuracy_per_class"], val_result["metrics"]["loss"]
        improved = score > best_score or (score == best_score and loss < best_loss)
        if improved:
            best_score, best_loss = score, loss
        stage_improved = score > stage_best_score or (score == stage_best_score and loss < stage_best_loss)
        if stage_improved:
            stage_best_score, stage_best_loss, bad_epochs = score, loss, 0
        else:
            bad_epochs += 1
        if scheduler:
            scheduler.step(loss)
        peak_vram_mb = round(torch.cuda.max_memory_allocated() / (1024 ** 2), 1) if device.type == "cuda" else 0.0
        row = {"epoch": epoch + 1, "stage": stage, "lr": used_lr, "seconds": round(time.monotonic() - started, 2), "peak_vram_mb": peak_vram_mb}
        for prefix, result in (("train", train_result), ("clean_train", clean_result), ("val", val_result)):
            for k in ("loss", "top1", "average_accuracy_per_class", "top5", "macro_f1"):
                if k in result["metrics"]:
                    row[f"{prefix}_{k}"] = result["metrics"][k]

        # Generalization Gap: Mathematically isolated gap between deterministic clean training
        # and held-out validation accuracy, free from stochastic data augmentation artifacts.
        row["generalisation_gap"] = row["clean_train_top1"] - row["val_top1"]
        history.append(row)
        stop = bool(config["patience"] and bad_epochs >= config["patience"] and stage != "head")
        checkpoint = {"state_dict": model.state_dict(), "config": config, "epoch": epoch + 1, "stage": stage,
                      "class_names": split["class_names"], "split_fingerprint": split["fingerprint"],
                      "validation_metrics": val_result["metrics"], "best_score": best_score, "best_loss": best_loss,
                      "stage_best_score": stage_best_score, "stage_best_loss": stage_best_loss,
                      "bad_epochs": bad_epochs, "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict() if scheduler else None,
                      "scaler": scaler.state_dict(), "rng": rng_state(loaders["train"]), "history": history,
                      "complete": stop or epoch + 1 == config["epochs"]}
        if improved:
            save_checkpoint(checkpoints / "best.pt", checkpoint)
        save_checkpoint(checkpoints / "last.pt", checkpoint)
        write_csv(out / "history.csv", history)
        learning_curves(history, out / "figures" / "learning_curves.png", smoke=config["smoke_test"])
        print(f"Epoch {epoch + 1:02d} [{stage}]: val Top-1={row['val_top1']:.3%}, Avg/class={score:.3%}, gap={row['generalisation_gap']:.3%}, {row['seconds']:.1f}s", flush=True)
        if stop:
            break
    best = load_checkpoint(checkpoints / "best.pt")
    model.load_state_dict(best["state_dict"])
    model.eval()
    validation = run_epoch(model, loaders["val"], device, config["num_classes"], max_batches=args.max_batches)
    save_analysis(validation, split["class_names"], data["train_root"], split["val"], out / "validation")
    write_json(out / "summary.json", {"best_epoch": best["epoch"], "best_stage": best["stage"],
               "validation": validation["metrics"], "smoke_test": config["smoke_test"],
               "split_fingerprint": split["fingerprint"], "checkpoint_sha256": digest(checkpoints / "best.pt")})
    return out


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--data", default="configs/data.local.json")
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="auto")
    p.add_argument("--epochs", type=int)
    p.add_argument("--batch-size", type=int)
    p.add_argument("--max-batches", type=int, help="Software smoke test only; cannot be selected for final evaluation")
    p.add_argument("--resume", action="store_true")
    return p


if __name__ == "__main__":
    train(parser().parse_args())
