"""Strict annotation parsing, safe file lookup, image audit and grouped splitting.

Learning Objectives:
- Robust File Parsing: Manifests use space-separated `<path> <zero_based_label>` lines.
  Path traversal attacks (`..`, absolute paths) and non-numeric labels are rejected.
- Species Vocabulary Recovery: Recovers 200 readable English bird names from training
  filenames via regex, guaranteeing 1-to-1 bijection across all 200 class indices.
- Exact-Pixel Duplicate Detection: Decodes raw RGB image arrays and hashes pixels via
  SHA-256, finding metadata-only duplicate images and resolving conflicting annotations.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath
import hashlib
import random
import re

from PIL import Image, ImageOps
from .utils import digest


def read_manifest(path, num_classes=200):
    """Parse authoritative CUB-200 manifest files (data/train.txt, data/test.txt).

    Design Rationale:
    - Authoritative Format: Lines contain `<filename> <class_index>`, where class_index
      is a zero-based integer from 0 to 199.
    - Path Traversal Security: Rejects absolute paths, parent directory sequences ('..'),
      and drive letter colons to guarantee all references stay within dataset roots.
    - Duplicate Entry Detection: Ensures no filename appears multiple times in the manifest.
    """
    records, seen = [], set()
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        try:
            name, raw_label = line.rsplit(maxsplit=1)
            label = int(raw_label)
        except ValueError as error:
            raise ValueError(f"{path}:{line_number}: expected filename and integer label") from error
        name = name.replace("\\", "/")
        rel = PurePosixPath(name)
        if rel.is_absolute() or ".." in rel.parts or ":" in name or not name:
            raise ValueError(f"Unsafe image path: {name}")
        if not 0 <= label < num_classes:
            raise ValueError(f"Label {label} outside 0..{num_classes - 1}")
        if name.casefold() in seen:
            raise ValueError(f"Repeated filename in manifest: {name}")
        seen.add(name.casefold())
        records.append({"name": name, "label": label})
    if not records:
        raise ValueError(f"Empty manifest: {path}")
    return records


def infer_class_names(records, num_classes):
    """Infer readable bird species names from structured training filenames.

    Design Rationale:
    Training filenames follow the naming convention `<Species_Name>_<bird_id>_<photo_id>.jpg`
    (e.g., 'Red_headed_Woodpecker_0012_2578571305.jpg'). We extract the species prefix,
    replace underscores with spaces, and verify that every class index maps to exactly
    one unique species name across all 200 classes. Numeric test filenames (e.g. '0001.jpg')
    contain no species metadata, so class names must be learned strictly from training.
    """
    candidates = defaultdict(set)
    for item in records:
        match = re.match(r"^(.+?)_\d+_\d+$", Path(item["name"]).stem)
        if match:
            candidates[item["label"]].add(match.group(1).replace("_", " "))
    names = []
    for label in range(num_classes):
        if len(candidates[label]) != 1:
            raise ValueError(f"Cannot uniquely infer species name for label {label}; supply class_names.json.")
        names.append(next(iter(candidates[label])))
    if len(set(names)) != len(names):
        raise ValueError("Species names are not unique")
    return names


def resolve_files(root, records):
    """Support both flat ZIP extractions and nested folders; reject ambiguity."""
    root = Path(root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Image folder not found: {root}. Edit configs/data.local.json.")
    index = defaultdict(list)
    for path in root.rglob("*"):
        if path.is_file():
            index[path.name.casefold()].append(path)
    paths = []
    for item in records:
        direct = root / item["name"]
        matches = [direct] if direct.is_file() else index[Path(item["name"]).name.casefold()]
        if len(matches) > 1:
            raise ValueError(f"Ambiguous filename {item['name']}: {matches}")
        path = matches[0].resolve() if matches else direct.resolve()
        if not path.is_relative_to(root):
            raise ValueError(f"Image resolves outside configured root: {path}")
        paths.append(path)
    return paths


def audit_images(root, records, num_classes):
    """Hash both file bytes and decoded RGB pixels (finds metadata-only copies)."""
    paths = resolve_files(root, records)
    rows, errors = [], []
    for item, path in zip(records, paths):
        if not path.is_file():
            errors.append({"name": item["name"], "error": "missing"})
            continue
        try:
            with Image.open(path) as im:
                im.verify()
            with Image.open(path) as im:
                im.load()
                mode = im.mode
                rgb = ImageOps.exif_transpose(im).convert("RGB")
                pixel_hash = hashlib.sha256(str(rgb.size).encode() + rgb.tobytes()).hexdigest()
                width, height = rgb.size
            rows.append({**item, "relative_path": path.relative_to(Path(root).resolve()).as_posix(),
                         "width": width, "height": height, "mode": mode,
                         "file_sha256": digest(path), "pixel_sha256": pixel_hash})
        except (OSError, ValueError, Image.DecompressionBombError) as error:
            errors.append({"name": item["name"], "error": str(error)})
    groups = defaultdict(list)
    for row in rows:
        groups[row["pixel_sha256"]].append(row)
    duplicates = [[r["name"] for r in group] for group in groups.values() if len(group) > 1]
    conflicts = [[r["name"] for r in group] for group in groups.values()
                 if len({r["label"] for r in group}) > 1]
    counts = Counter(r["label"] for r in records)
    return {"records": rows, "errors": errors, "duplicates": duplicates,
            "conflicting_duplicates": conflicts, "class_counts": [counts[i] for i in range(num_classes)],
            "total_listed": len(records), "total_readable": len(rows),
            "width_range": [min((r["width"] for r in rows), default=0), max((r["width"] for r in rows), default=0)],
            "height_range": [min((r["height"] for r in rows), default=0), max((r["height"] for r in rows), default=0)],
            "modes": dict(Counter(r["mode"] for r in rows)),
            "duplicate_scope": "Exact decoded RGB matches; near-duplicates are not detected."}


def stratified_group_split(rows, fraction=0.2, seed=42, conflict_policy="error"):
    """Perform a duplicate-grouped, class-stratified split into train and validation sets.

    Learning Objectives & Data Integrity Guarantees:
    1. Group Integrity (No Data Leakage): All pixel-identical images (sharing the exact same
       SHA-256 hash of decoded RGB pixel arrays) are grouped together. Every image in a
       group is assigned to the SAME partition (either train or val). This guarantees that
       no validation sample has an exact copy in the training set.
    2. Conflicting Duplicate Pinning (conflict_policy='train_only'): The source metadata
       contains three duplicate pairs with contradictory labels (e.g., identical photo
       labelled both Crested Auklet and Parakeet Auklet). Pinning all members of conflicting
       groups to the training set eliminates label contradiction in validation, keeping
       the validation benchmark clean while transparently disclosing 0.16% training label noise.
    3. Deterministic Subset-Sum Search: Because duplicate groups have varying sample counts
       (1 to 3 images), standard random splitting cannot hit an exact 20% holdout per class.
       We formulate validation selection per class as an exact subset-sum search to find the
       combination of distinct image groups whose total count is closest to `total * fraction`.

    Args:
        rows (list[dict]): Audited image record dictionaries.
        fraction (float): Target validation fraction (default: 0.2, i.e., 20%).
        seed (int): RNG seed for reproducible group shuffling (default: 42).
        conflict_policy (str): Action for conflicting duplicate hashes ('error' or 'train_only').

    Returns:
        tuple[list[dict], list[dict]]: (train_records, val_records) sorted deterministically by name.
    """
    if not 0 < fraction < 1:
        raise ValueError("Validation fraction must lie strictly between zero and one")
    rng = random.Random(seed)
    by_class, hash_labels = defaultdict(lambda: defaultdict(list)), defaultdict(set)
    for row in rows:
        by_class[row["label"]][row["pixel_sha256"]].append(row)
        hash_labels[row["pixel_sha256"]].add(row["label"])
    conflicts = {h for h, labels in hash_labels.items() if len(labels) > 1}
    if conflicts and conflict_policy != "train_only":
        raise ValueError("Identical images have conflicting labels; resolve before splitting")
    train, val = [], []
    for label in sorted(by_class):
        # Pin any conflicting-label duplicate groups directly to the training partition
        pinned = [r for h, group in by_class[label].items() if h in conflicts for r in group]
        train.extend(pinned)
        groups = [group for h, group in by_class[label].items() if h not in conflicts]
        rng.shuffle(groups)
        if len(groups) < 2:
            raise ValueError(f"Class {label} needs at least two distinct image groups")
        total = sum(map(len, groups))
        # Subset-sum dynamic programming to find group combination closest to target holdout
        possible = {0: []}
        for index, group in enumerate(groups):
            for size, selected in list(possible.items()):
                possible.setdefault(size + len(group), selected + [index])
        size = min((s for s in possible if 0 < s < total), key=lambda s: (abs(s - (total + len(pinned)) * fraction), s))
        selected = set(possible[size])
        for index, group in enumerate(groups):
            (val if index in selected else train).extend(group)
    return sorted(train, key=lambda r: r["name"]), sorted(val, key=lambda r: r["name"])


def verify_files(root, rows):
    """Detect dataset mutation since preparation instead of training on a stale split."""
    root = Path(root).resolve()
    for row in rows:
        path = (root / row["relative_path"]).resolve()
        if not path.is_relative_to(root) or not path.is_file() or digest(path) != row["file_sha256"]:
            raise ValueError(f"Image changed/missing since preparation: {row['name']}. Audit again in a new prepared directory.")
