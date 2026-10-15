from pathlib import Path
import pytest
from PIL import Image
from src.data import read_manifest, audit_images, stratified_group_split, resolve_files


def test_manifest_rejects_invalid_labels_duplicates_and_traversal(tmp_path):
    manifest = tmp_path / "train.txt"
    for text in ("a.jpg 2", "../a.jpg 0", "a.jpg 0\nA.jpg 1", "oops"):
        manifest.write_text(text)
        with pytest.raises(ValueError):
            read_manifest(manifest, 2)
    manifest.write_text("a photo.jpg 0\nb.jpg 1\n")
    assert read_manifest(manifest, 2)[0]["name"] == "a photo.jpg"


def test_audit_detects_missing_corrupt_duplicates_and_nested_layout(tmp_path):
    nested = tmp_path / "Train"
    nested.mkdir()
    Image.new("RGB", (12, 20), "red").save(nested / "a.png")
    Image.new("RGB", (12, 20), "red").save(nested / "b.png")
    (nested / "broken.png").write_text("not an image")
    rows = [{"name": n, "label": i % 2} for i, n in enumerate(("a.png", "b.png", "broken.png", "absent.png"))]
    audit = audit_images(tmp_path, rows, 2)
    assert len(audit["errors"]) == 2
    assert audit["conflicting_duplicates"] == [["a.png", "b.png"]]
    assert audit["records"][0]["relative_path"] == "Train/a.png"
    (tmp_path / "other").mkdir()
    Image.new("RGB", (3, 3)).save(tmp_path / "other" / "a.png")
    with pytest.raises(ValueError, match="Ambiguous"):
        resolve_files(tmp_path, rows)


def test_stratification_group_isolation_and_conflict_policy():
    rows = [{"name": f"c{c}-{i}", "label": c, "pixel_sha256": f"{c}-{i // 2}"}
            for c in range(3) for i in range(10)]
    first = stratified_group_split(rows)
    assert first == stratified_group_split(rows)
    train, val = first
    assert set(r["pixel_sha256"] for r in train).isdisjoint(r["pixel_sha256"] for r in val)
    assert len(val) == 6
    rows[0]["pixel_sha256"] = rows[10]["pixel_sha256"] = "conflict"
    with pytest.raises(ValueError, match="conflicting"):
        stratified_group_split(rows)
    train, val = stratified_group_split(rows, conflict_policy="train_only")
    assert sum(r["pixel_sha256"] == "conflict" for r in train) == 2
    assert not any(r["pixel_sha256"] == "conflict" for r in val)
    assert {r["label"] for r in val} == {0, 1, 2}
