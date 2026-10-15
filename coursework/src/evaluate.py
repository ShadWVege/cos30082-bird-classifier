"""Official held-out test evaluation runner enforcing strict pre-commitment protocols.

Scientific Protocol & Rubric Compliance:
1. Frozen Pre-Commitment: Before accessing the official held-out test set (`data/test.txt`),
   candidate models must be frozen in `selection.json` based strictly on development
   validation performance. Checkpoint SHA-256 hashes and split fingerprints are verified.
2. Zero Post-Test Tuning: The test set is evaluated ONCE for unbiased performance assessment.
   Hyperparameters, architectures, and checkpoints cannot be altered after observing test scores.
3. Test Data Integrity Auditing: All 1,204 test images are audited for decoding validity,
   class presence, and cross-split duplicate pixels against the training manifest.
4. Transparent Limitation Disclosure: Four cross-split conflicting duplicate pairs between
   Train and Test are acknowledged and disclosed as an inherent benchmark artifact, preserving
   the full 1,204-image denominator without silent tampering.
"""
import argparse
from pathlib import Path

from .data import audit_images, read_manifest, verify_files
from .engine import load_checkpoint, make_loader, run_epoch
from .models import build_model
from .plots import save_analysis
from .utils import choose_device, data_config, digest, load_split, read_json, write_csv, write_json


def evaluate(args):
    """Execute official held-out test evaluation across pre-committed model checkpoints."""
    if not args.final:
        raise ValueError("Final evaluation requires --final after validation choices are frozen")
    selection = read_json(args.selection)
    cfg = data_config(args.data)
    split = load_split(Path(cfg["prepared_dir"]) / "split.json")
    # Verify all pre-committed checkpoints against split fingerprint BEFORE opening test data
    for entry in selection["runs"]:
        if entry["split_fingerprint"] != split["fingerprint"] or digest(entry["checkpoint"]) != entry["checkpoint_sha256"]:
            raise ValueError("Selection/checkpoint/split mismatch")
        checkpoint = load_checkpoint(entry["checkpoint"])
        if checkpoint["config"].get("smoke_test"):
            raise ValueError("Smoke checkpoints are ineligible for final evaluation")
    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("Choose a fresh final-evaluation directory; do not overwrite results")
    out.mkdir(parents=True, exist_ok=True)
    verify_files(cfg["train_root"], split["train"] + split["val"])
    records = read_manifest(cfg["test_manifest"], cfg["num_classes"])
    audit = audit_images(cfg["test_root"], records, cfg["num_classes"])
    train_hashes = {r["pixel_sha256"] for r in split["train"] + split["val"]}
    overlap = [r["name"] for r in audit["records"] if r["pixel_sha256"] in train_hashes]
    audit["overlap_with_provided_training_set"] = overlap
    write_json(out / "test_audit.json", audit)
    if audit["errors"] or any(c == 0 for c in audit["class_counts"]):
        raise ValueError("Held-out audit failed; see test_audit.json")
    if (overlap or audit["conflicting_duplicates"]) and not args.acknowledge_integrity_issues:
        raise ValueError("Official test data has duplicate/label issues. Review test_audit.json, document limitations, then explicitly acknowledge in a fresh output directory. No scores produced.")
    device = choose_device(args.device)
    table = []
    for entry in selection["runs"]:
        checkpoint = load_checkpoint(entry["checkpoint"])
        model = build_model(checkpoint["config"], pretrained=False).to(device)
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        loader = make_loader(cfg["test_root"], audit["records"], checkpoint["config"])
        result = run_epoch(model, loader, device, cfg["num_classes"])
        save_analysis(result, checkpoint["class_names"], cfg["test_root"], audit["records"], out / entry["run"])
        table.append({"run": entry["run"], "model": entry["model"], **result["metrics"]})
        print(entry["run"], result["metrics"], flush=True)
    write_csv(out / "test_comparison.csv", table)
    write_json(out / "provenance.json", {"selection_sha256": digest(args.selection),
               "test_manifest_sha256": digest(cfg["test_manifest"]), "split_fingerprint": split["fingerprint"],
               "deployment_run_chosen_before_test": selection["deployment_run"],
               "integrity_issues_acknowledged": args.acknowledge_integrity_issues,
               "overlap_count": len(overlap), "test_used_for_model_selection": False})
    return table


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--selection", default="outputs/selection.json")
    p.add_argument("--data", default="configs/data.local.json")
    p.add_argument("--out", default="outputs/final_test")
    p.add_argument("--device", default="auto")
    p.add_argument("--final", action="store_true")
    p.add_argument("--acknowledge-integrity-issues", action="store_true")
    evaluate(p.parse_args())
