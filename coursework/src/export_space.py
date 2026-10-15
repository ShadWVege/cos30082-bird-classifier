"""Build a standalone Hugging Face Space directory from the validation-selected checkpoint."""
import argparse
import importlib.metadata
from pathlib import Path
import shutil

from .engine import load_checkpoint, save_checkpoint
from .utils import ROOT, digest, read_json, write_json


def export_space(selection_path, out, overwrite=False):
    """Export the winning checkpoint, inference source modules, and static chart assets into `out`."""
    selection = read_json(selection_path)
    entry = next(e for e in selection["runs"] if e["run"] == selection["deployment_run"])
    if digest(entry["checkpoint"]) != entry["checkpoint_sha256"]:
        raise ValueError("Selected checkpoint was modified")
    state = load_checkpoint(entry["checkpoint"])
    if state["config"].get("smoke_test"):
        raise ValueError("Cannot deploy a smoke-test checkpoint")
    out = Path(out)
    if out.exists() and any(out.iterdir()):
        if not overwrite:
            raise FileExistsError("Export destination must be new/empty")
        for child in out.iterdir():
            if child.name == ".git":
                continue
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
    (out / "src").mkdir(parents=True, exist_ok=True)
    for name in (
        "__init__.py",
        "models.py",
        "dataset.py",
        "inference.py",
        "engine.py",
        "utils.py",
        "metrics.py",
    ):
        shutil.copy2(ROOT / "src" / name, out / "src" / name)
    shutil.copy2(ROOT / "app.py", out / "app.py")

    # Copy static learning-curve and summary figures (if present) so Section 3 (Model Insights) works standalone.
    asset_sources = [
        ROOT / "outputs" / "experiment_001" / "figures" / "model_comparison.png",
        ROOT / "outputs" / "experiment_001" / "figures" / "ablation_e2_vs_e4.png",
        ROOT / "outputs" / "final_test_results" / "figures" / "test_metrics_summary.png",
    ]
    existing_assets = [p for p in asset_sources if p.is_file()]
    if existing_assets:
        (out / "assets").mkdir(parents=True, exist_ok=True)
        for asset in existing_assets:
            shutil.copy2(asset, out / "assets" / asset.name)

    save_checkpoint(
        out / "model.pt",
        {key: state[key] for key in ("state_dict", "config", "class_names")},
    )
    write_json(out / "class_names.json", state["class_names"])
    write_json(
        out / "model_info.json",
        {
            "architecture": state["config"]["model"],
            "epoch": state["epoch"],
            "validation_metrics": state["validation_metrics"],
            "source_checkpoint_sha256": entry["checkpoint_sha256"],
            "split_fingerprint": state["split_fingerprint"],
            "selection": selection["criterion"],
        },
    )
    packages = ("torch", "torchvision", "numpy", "Pillow", "gradio")
    # Direct Linux cloud builders (Render Free Web Service / HF CPU Spaces) to install the
    # lightweight CPU-only PyTorch wheels (~181 MiB) instead of the 2.5 GB CUDA bundle.
    req_lines = ["--extra-index-url https://download.pytorch.org/whl/cpu"]
    req_lines.extend(f"{p}=={importlib.metadata.version(p).split('+')[0]}" for p in packages)
    (out / "requirements.txt").write_text("\n".join(req_lines) + "\n", encoding="utf-8")

    # Pin Python 3.12 and Render Free Web Service configuration (0.0.0.0:$PORT, single worker, 512 MiB safe env)
    (out / ".python-version").write_text("3.12.10\n", encoding="utf-8")
    (out / "render.yaml").write_text(
        "services:\n"
        "  - type: web\n"
        "    name: cos30082-bird-classifier\n"
        "    runtime: python\n"
        "    plan: free\n"
        "    buildCommand: pip install --no-cache-dir -r requirements.txt\n"
        "    startCommand: python app.py\n"
        "    envVars:\n"
        "      - key: PYTHON_VERSION\n"
        "        value: \"3.12.10\"\n"
        "      - key: GRADIO_SERVER_NAME\n"
        "        value: \"0.0.0.0\"\n"
        "      - key: MALLOC_ARENA_MAX\n"
        "        value: \"2\"\n"
        "      - key: OMP_NUM_THREADS\n"
        "        value: \"1\"\n"
        "      - key: MKL_NUM_THREADS\n"
        "        value: \"1\"\n",
        encoding="utf-8",
    )

    version = importlib.metadata.version("gradio")
    (out / "README.md").write_text(
        f"---\n"
        f"title: Bird Species Explorer\n"
        f"emoji: 🐦\n"
        f"colorFrom: green\n"
        f"colorTo: blue\n"
        f"sdk: gradio\n"
        f"sdk_version: {version}\n"
        f"app_file: app.py\n"
        f"python_version: '3.12'\n"
        f"pinned: false\n"
        f"---\n\n"
        f"# COS30082 Bird Species Classification (CUB-200)\n\n"
        f"200-class North American bird species classifier (`000–199`) powered by our validation-selected "
        f"**Staged ResNet18 Transfer Learning (`E3`)** model.\n\n"
        f"- **Validation Performance (959 images):** `69.66%` Top-1 Accuracy | `69.10%` Average Accuracy per Class | `89.57%` Top-5 Accuracy.\n"
        f"- **Official Held-Out Test Performance (1,204 images):** `70.43%` Top-1 Accuracy | `70.12%` Average Accuracy per Class | `89.12%` Top-5 Accuracy | `69.90%` Macro-F1.\n\n"
        f"See `model_info.json` for checkpoint provenance and SHA-256 verification. Softmax scores sum to 100% across "
        f"the 200 closed-set classes and are not calibrated confidence. Compatible with both **Render Free Web Service** "
        f"(`render.yaml`, 512 MiB RAM CPU profile) and **Hugging Face Spaces** (`app.py`). No dataset archives or raw "
        f"training/test photographs are distributed in this package.\n",
        encoding="utf-8",
    )
    (out / ".gitignore").write_text("__pycache__/\n*.py[cod]\n", encoding="utf-8")
    print(f"Standalone deployment package ready: {out.resolve()}")
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--selection", default="outputs/selection.json")
    p.add_argument("--out", default="outputs/huggingface_space")
    p.add_argument("--overwrite", action="store_true")
    a = p.parse_args()
    export_space(a.selection, a.out, overwrite=a.overwrite)
