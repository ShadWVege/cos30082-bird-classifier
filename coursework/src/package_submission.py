"""Create a source ZIP without images, caches, secrets, runs or environment files."""
import argparse
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
from .utils import ROOT


def package(out, force=False):
    """Build a source-only assignment ZIP while excluding checkpoints.

    Keep the authoring source, tests, metadata, and selected report figures;
    exclude local machine settings, cached intermediates and dataset photos.
    """
    out = Path(out)
    if out.exists():
        if force:
            out.unlink()
        else:
            raise FileExistsError("Use a new ZIP filename or pass --force to overwrite the prior package")
    out.parent.mkdir(parents=True, exist_ok=True)
    roots = [ROOT / folder for folder in ("src", "tests", "docs", "configs", "scripts", "references", "presentation")]
    files = [p for folder in roots for p in folder.rglob("*") if p.is_file()
             and "__pycache__" not in p.parts and p.name != "data.local.json"]
    # Include report source markup (typst) and referenced publication figures
    report_typ = ROOT / "reports" / "report.typ"
    if report_typ.is_file():
        files.append(report_typ)
    report_figures = [
        ROOT / "outputs/experiment_001/figures/model_comparison.png",
        ROOT / "outputs/experiment_001/figures/ablation_e2_vs_e4.png",
        ROOT / "outputs/experiment_001/figures/ablation_e3_vs_e7.png",
        ROOT / "outputs/experiment_001/figures/gradcam_focused.png",
        ROOT / "outputs/final_test_results/figures/test_per_class_and_confusions.png",
        ROOT / "outputs/experiment_001/figures/live_render_tta_view.png",
    ]
    for rf in report_figures:
        if rf.is_file():
            files.append(rf)
    # Include deploy code and assets while strictly excluding 112 MB model weights and zip archives
    deploy_root = ROOT / "deploy"
    if deploy_root.is_dir():
        files += [p for p in deploy_root.rglob("*") if p.is_file()
                  and "__pycache__" not in p.parts
                  and p.suffix not in (".pt", ".bin")
                  and not p.name.endswith((".zip", ".part1", ".part2"))]
    files += [p for p in ROOT.iterdir() if p.is_file() and (p.suffix in (".md", ".txt", ".toml", ".py", ".code-workspace") or p.name == ".gitignore")]
    files += [ROOT / "data" / name for name in ("train.txt", "test.txt")]
    prepared = ROOT / "data" / "prepared"
    if prepared.is_dir():
        files += [p for p in prepared.iterdir() if p.suffix in (".json", ".csv")]
    with ZipFile(out, "w", ZIP_DEFLATED) as archive:
        for path in sorted(set(files)):
            archive.write(path, Path("COS30082_BirdClassification") / path.relative_to(ROOT))
    print(f"Source package: {out.resolve()} ({len(files)} files)")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default="outputs/source_submission.zip")
    p.add_argument("--force", "-f", action="store_true", help="Overwrite existing output ZIP archive")
    args = p.parse_args()
    package(args.out, force=args.force)
