"""Training/evaluation mechanics shared by command-line entry points."""
import random
import numpy as np
import torch
from torch.utils.data import DataLoader

from .dataset import BirdDataset
from .metrics import classification_metrics
from .utils import seed_worker


def make_loader(root, records, config, augment=False, shuffle=False):
    generator = torch.Generator().manual_seed(config["seed"])
    return DataLoader(BirdDataset(root, records, config["image_size"], augment),
                      batch_size=config["batch_size"], shuffle=shuffle,
                      num_workers=config["num_workers"], pin_memory=torch.cuda.is_available(),
                      worker_init_fn=seed_worker, generator=generator)


def run_epoch(model, loader, device, classes, optimizer=None, scaler=None, amp=False, max_batches=None, label_smoothing=0.0):
    """Run one training or evaluation epoch.

    Learning objective: When `training=True` (`optimizer is not None`), optional label smoothing
    (`CrossEntropyLoss(label_smoothing=...)`) softens one-hot targets to discourage extreme logit
    overconfidence on small fine-grained classes. When `training=False` (`clean_train`, `val`, `test`),
    `label_smoothing` is forced to `0.0` so evaluation cross-entropy remains on an identical,
    unsmoothed scale across all experiments.
    """
    loss_sum, count, targets, predictions, names = 0.0, 0, [], [], []
    training = optimizer is not None
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=float(label_smoothing) if training else 0.0)
    for batch, (images, labels, batch_names) in enumerate(loader):
        if max_batches is not None and batch >= max_batches:
            break
        images, labels = images.to(device), labels.to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            with torch.autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
                logits = model(images)
                loss = criterion(logits, labels)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite loss; inspect images, precision and learning rate")
            if training:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                scaler.step(optimizer)
                scaler.update()
        count += len(labels)
        loss_sum += loss.item() * len(labels)
        targets.extend(labels.cpu().tolist())
        predictions.append(logits.detach().float().cpu().numpy())
        names.extend(batch_names)
    if not count:
        raise ValueError("No batches were processed")
    scores, per_class, confusion = classification_metrics(targets, np.concatenate(predictions), classes,
                                                          require_all=max_batches is None)
    scores["loss"] = loss_sum / count
    return {"metrics": scores, "per_class": per_class, "confusion": confusion,
            "targets": targets, "logits": np.concatenate(predictions), "names": names}


def rng_state(loader):
    state = np.random.get_state()
    return {"python": random.getstate(), "numpy": [state[0], state[1].tolist(), state[2], state[3], state[4]],
            "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            "loader": loader.generator.get_state()}


def restore_rng(state, loader):
    random.setstate(state["python"])
    n = state["numpy"]
    np.random.set_state((n[0], np.array(n[1], dtype=np.uint32), n[2], n[3], n[4]))
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])
    loader.generator.set_state(state["loader"].cpu())


def save_checkpoint(path, value):
    from pathlib import Path
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def load_checkpoint(path, mmap=False):
    # Safe tensor/primitive-only format; never need arbitrary pickle execution.
    return torch.load(str(path), map_location="cpu", weights_only=True, mmap=bool(mmap))
