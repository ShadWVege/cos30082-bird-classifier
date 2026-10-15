"""Compare validation results and freeze checkpoint choices before using test."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
from .engine import load_checkpoint
from .utils import digest, read_json, write_csv, write_json


def select_runs(run_paths, out):
    """Select the highest macro-recall checkpoint from compatible validation runs.

    Refuse smoke runs, changed checkpoints, or mixed splits. The selection
    record is write-once to preserve the test-set firewall.
    """
    out = Path(out)
    if out.exists():
        raise FileExistsError("Selection already exists; preserve this experiment record")
    entries = []
    for path in run_paths:
        run = Path(path).resolve()
        summary = read_json(run / "summary.json")
        checkpoint = run / "checkpoints" / "best.pt"
        if summary["smoke_test"]:
            raise ValueError("Smoke tests are not eligible for final selection")
        if digest(checkpoint) != summary["checkpoint_sha256"]:
            raise ValueError(f"Checkpoint changed after validation: {checkpoint}")
        state = load_checkpoint(checkpoint)
        entries.append({"run": run.name, "checkpoint": str(checkpoint),
                        "checkpoint_sha256": summary["checkpoint_sha256"], "model": state["config"]["model"],
                        "split_fingerprint": summary["split_fingerprint"], "best_epoch": summary["best_epoch"],
                        "validation": summary["validation"]})
    if len({e["split_fingerprint"] for e in entries}) != 1:
        raise ValueError("Runs used different splits and cannot be compared fairly")
    if len({e["run"] for e in entries}) != len(entries):
        raise ValueError("Run names must be unique")
    best = max(entries, key=lambda e: (e["validation"]["average_accuracy_per_class"], -e["validation"]["loss"]))
    result = {"created_utc": datetime.now(timezone.utc).isoformat(),
              "criterion": "validation average accuracy per class; tie-break lowest validation cross-entropy",
              "deployment_run": best["run"], "runs": entries}
    write_json(out, result)
    write_csv(out.with_suffix(".csv"), [{"run": e["run"], "model": e["model"], "best_epoch": e["best_epoch"], **e["validation"]} for e in entries])
    print(f"Frozen {len(entries)} checkpoints; validation-selected deployment model: {best['run']}")
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--out", default="outputs/selection.json")
    a = p.parse_args()
    select_runs(a.runs, a.out)
