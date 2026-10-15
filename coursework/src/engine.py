"""Training and evaluation execution engine for CUB-200 bird classification.

Learning Objectives & Design Rationale:
- Deterministic DataLoader: Seeds batch shuffling and worker processes via PyTorch
  Generator and worker_init_fn, guaranteeing reproducible data presentation order.
- Mixed-Precision & Gradient Scaling: Uses torch.amp.autocast and GradScaler on CUDA
  devices to accelerate training while preventing FP16 gradient underflow.
- Gradient Clipping: Clips parameter gradient L2-norm to 5.0, protecting deep backbone
  weights from explosive updates during early fine-tuning.
- Atomic Checkpointing: Writes weights to a temporary file before renaming, preventing
  corrupted partial saves upon unexpected termination.
- Safe Loading: Enforces weights_only=True to prevent arbitrary code execution via pickle.
"""
import random
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader

from .dataset import BirdDataset
from .metrics import classification_metrics
from .utils import seed_worker


def make_loader(root, records, config, augment=False, shuffle=False):
    """Construct a seeded PyTorch DataLoader with deterministic batching.

    Args:
        root (Path | str): Image root directory.
        records (list[dict]): Dataset records for train/val/test split.
        config (dict): Run configuration specifying batch_size, seed, num_workers.
        augment (bool): Whether to apply stochastic training augmentations.
        shuffle (bool): Whether to randomize batch sample ordering.

    Design Rationale:
    - Generator Seeding: Binding an explicit `torch.Generator` seeded with config['seed']
      ensures that batch shuffling sequences are strictly identical across identical runs.
    - Windows DataLoader Workers: Defaulting to num_workers=0 on Windows avoids Python
      multiprocessing spawn overhead, memory duplication, and shared memory lockups.
    - Pinned Memory: Enables fast asynchronous host-to-device memory copies via DMA
      when CUDA is available.
    """
    generator = torch.Generator().manual_seed(config["seed"])
    return DataLoader(
        BirdDataset(root, records, config["image_size"], augment),
        batch_size=config["batch_size"],
        shuffle=shuffle,
        num_workers=config["num_workers"],
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=seed_worker,
        generator=generator,
    )


def run_epoch(
    model,
    loader,
    device,
    classes,
    optimizer=None,
    scaler=None,
    amp=False,
    max_batches=None,
    label_smoothing=0.0,
):
    """Execute one training or evaluation epoch over a DataLoader.

    Args:
        model (nn.Module): CNN architecture under training or evaluation.
        loader (DataLoader): DataLoader supplying batches of (images, labels, names).
        device (torch.device): Target compute device (CPU or CUDA).
        classes (int): Number of target classes (200 for CUB-200).
        optimizer (Optimizer | None): If provided, executes training backpropagation.
            If None, executes pure evaluation under torch.no_grad().
        scaler (GradScaler | None): Gradient scaler for Automatic Mixed Precision.
        amp (bool): Whether CUDA Automatic Mixed Precision is active.
        max_batches (int | None): Limits processed batches (for quick smoke tests).
        label_smoothing (float): Label smoothing epsilon (0.0 to 0.05).

    Learning Objective & Evaluation Contract:
    - Training Mode: When optimizer is present, CrossEntropyLoss applies optional label
      smoothing (e.g. 0.05 in E5) to regularize logit overconfidence on 200 fine-grained classes.
    - Evaluation Mode: When optimizer is None (clean_train, val, test), label_smoothing
      is strictly forced to 0.0 so cross-entropy loss remains on an identical, unsmoothed
      scale across all baseline and exploratory experiments.
    """
    loss_sum, count, targets, predictions, names = 0.0, 0, [], [], []
    training = optimizer is not None
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=float(label_smoothing) if training else 0.0)

    for batch, (images, labels, batch_names) in enumerate(loader):
        if max_batches is not None and batch >= max_batches:
            break
        images, labels = images.to(device), labels.to(device)
        if training:
            # set_to_none=True deallocates gradient buffers rather than zeroing them,
            # reducing memory bandwidth and resident tensor allocations.
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            with torch.autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
                logits = model(images)
                loss = criterion(logits, labels)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite loss; inspect images, precision and learning rate")
            if training:
                # Scale loss before backward to prevent FP16 gradients from underflowing to zero
                scaler.scale(loss).backward()
                # Unscale gradients prior to clipping so the L2-norm is mathematically accurate
                scaler.unscale_(optimizer)
                # Clip gradient norm to 5.0 to prevent gradient explosions during backbone fine-tuning
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

    # Compute official Top-1, Average Accuracy per Class, Top-5, and Macro-F1 metrics
    scores, per_class, confusion = classification_metrics(
        targets, np.concatenate(predictions), classes, require_all=max_batches is None
    )
    scores["loss"] = loss_sum / count
    return {
        "metrics": scores,
        "per_class": per_class,
        "confusion": confusion,
        "targets": targets,
        "logits": np.concatenate(predictions),
        "names": names,
    }


def rng_state(loader):
    """Capture full random number generator state across Python, NumPy, PyTorch, and DataLoader."""
    state = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": [state[0], state[1].tolist(), state[2], state[3], state[4]],
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "loader": loader.generator.get_state(),
    }


def restore_rng(state, loader):
    """Restore saved RNG state to resume training identically from an interrupted checkpoint."""
    random.setstate(state["python"])
    n = state["numpy"]
    np.random.set_state((n[0], np.array(n[1], dtype=np.uint32), n[2], n[3], n[4]))
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])
    loader.generator.set_state(state["loader"].cpu())


def save_checkpoint(path, value):
    """Atomically write checkpoint to disk via temporary file rename.

    Writing first to `.tmp` and executing an atomic OS rename prevents leaving corrupted
    or truncated checkpoint files on disk if the process is terminated mid-write.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def load_checkpoint(path, mmap=False):
    """Safely load model weights and configuration from a saved checkpoint file.

    Design Rationale:
    - weights_only=True: Enforces PyTorch safe deserialization, rejecting arbitrary
      unpickling of executable Python objects (CWE-502 vulnerability mitigation).
    - mmap=True: On Linux, memory-maps tensor storages directly from disk, backing
      weights via the kernel page cache (RssFile) instead of private anonymous heap
      (RssAnon), significantly reducing startup memory spikes in 512 MiB cloud containers.
    """
    return torch.load(str(path), map_location="cpu", weights_only=True, mmap=bool(mmap))

