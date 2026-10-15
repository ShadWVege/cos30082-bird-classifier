"""Run the three planned models sequentially; freeze validation-based selection."""
import argparse
from pathlib import Path
import subprocess
import sys
from .compare import select_runs
from .utils import ROOT


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", default="configs/data.local.json")
    p.add_argument("--out", default="outputs/experiment_001")
    p.add_argument("--device", default="auto")
    p.add_argument("--resume", action="store_true")
    a = p.parse_args()
    base = Path(a.out).resolve()
    runs = []
    for model in ("baseline", "regularized", "resnet18"):
        run = base / model
        runs.append(run)
        if a.resume and (run / "summary.json").is_file():
            continue
        cmd = [sys.executable, "-m", "src.train", "--config", str(ROOT / "configs" / f"{model}.json"),
               "--data", str(Path(a.data).resolve()), "--out", str(run), "--device", a.device]
        if a.resume and (run / "checkpoints" / "last.pt").is_file():
            cmd.append("--resume")
        subprocess.run(cmd, check=True, cwd=ROOT)
    if not (base / "selection.json").exists():
        select_runs(runs, base / "selection.json")
    print("Experiments complete. Inspect validation results before explicitly running final test evaluation.")
