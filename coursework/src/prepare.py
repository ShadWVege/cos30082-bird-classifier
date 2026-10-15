"""Audit TRAIN only and persist the common split: python -m src.prepare ..."""
import argparse
from pathlib import Path

from .data import audit_images, infer_class_names, read_manifest, stratified_group_split
from .utils import data_config, digest, object_digest, read_json, write_csv, write_json


def prepare(config_path):
    """Audit training images and save an immutable grouped stratified split.

    This function intentionally never opens the official test image set.
    """
    cfg = data_config(config_path)
    out = Path(cfg["prepared_dir"])
    if (out / "split.json").exists():
        raise FileExistsError(f"Split already exists in {out}; reuse it or choose a new prepared_dir.")
    records = read_manifest(cfg["train_manifest"], cfg["num_classes"])
    expected = cfg.get("expected_train_count")
    if expected is not None and len(records) != expected:
        raise ValueError(f"Expected {expected} training records, found {len(records)}")
    if set(r["label"] for r in records) != set(range(cfg["num_classes"])):
        raise ValueError("Training manifest does not cover every configured class")
    report = audit_images(cfg["train_root"], records, cfg["num_classes"])
    write_json(out / "train_audit.json", report)
    policy = cfg.get("conflict_policy", "error")
    if policy not in ("error", "train_only"):
        raise ValueError("conflict_policy must be error or train_only")
    if report["errors"] or (report["conflicting_duplicates"] and policy == "error"):
        raise ValueError(f"Audit failed. See {out / 'train_audit.json'}; no split was created.")
    if cfg.get("class_names"):
        from .utils import project_path
        names = read_json(project_path(cfg["class_names"]))
    else:
        names = infer_class_names(records, cfg["num_classes"])
    if len(names) != cfg["num_classes"] or len(set(names)) != len(names):
        raise ValueError("Class names must be a unique ordered list, indexed by zero-based labels")
    train, val = stratified_group_split(report["records"], cfg["val_fraction"], cfg["seed"], policy)
    split = {"seed": cfg["seed"], "val_fraction": cfg["val_fraction"], "num_classes": cfg["num_classes"],
             "manifest_sha256": digest(cfg["train_manifest"]), "class_names": names, "train": train, "val": val,
             "conflict_policy": policy, "conflicting_duplicates": report["conflicting_duplicates"]}
    split["fingerprint"] = object_digest(split)
    write_json(out / "split.json", split)
    write_json(out / "class_names.json", names)
    write_csv(out / "class_counts.csv", [{"label": i, "name": names[i],
              "train": sum(r["label"] == i for r in train), "validation": sum(r["label"] == i for r in val)}
              for i in range(cfg["num_classes"])])
    print(f"Prepared {len(train)} training / {len(val)} validation images; {len(names)} classes. Test set not opened.")
    return split


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="configs/data.local.json")
    prepare(parser.parse_args().data)
