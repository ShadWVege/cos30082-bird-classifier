"""Small shared helpers. Data preparation intentionally needs no PyTorch."""
from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path, value):
    """Replace atomically so interruption does not leave half a JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def write_csv(path, rows, fields=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def object_digest(value):
    """Hash a canonical JSON representation for split reproducibility checks."""
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def load_split(path):
    """Read the precomputed split and reject accidental changes to its hash."""
    split = read_json(path)
    payload = {k: v for k, v in split.items() if k != "fingerprint"}
    if object_digest(payload) != split["fingerprint"]:
        raise ValueError("Prepared split was edited after creation; use the original or create a documented new split")
    return split


def project_path(value):
    path = Path(os.path.expandvars(str(value))).expanduser()
    return path if path.is_absolute() else ROOT / path


def data_config(path):
    cfg = read_json(path)
    for key in ("train_root", "test_root", "train_manifest", "test_manifest", "prepared_dir"):
        if key in cfg:
            cfg[key] = str(project_path(cfg[key]))
    return cfg


def seed_everything(seed):
    """Deterministic kernels where supported; log environment for reproducibility."""
    import random
    import numpy as np
    import torch
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def seed_worker(_worker_id):
    import random
    import numpy as np
    import torch
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def choose_device(name="auto"):
    """Select CPU/CUDA explicitly and fail fast on unavailable CUDA devices."""
    import torch
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable. Install a GPU PyTorch build or use --device cpu.")
    return torch.device(name)
