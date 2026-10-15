"""Integration tests use invented temporary images, NEVER assignment data."""
from argparse import Namespace
from pathlib import Path
import numpy as np
import pytest
import torch
from PIL import Image
from src.utils import read_json, write_json
from src.prepare import prepare
from src.train import train
from src.compare import select_runs
from src.evaluate import evaluate
from src.export_space import export_space
from src.inference import Predictor
from src.gradcam import gradcam


@pytest.fixture
def synthetic(tmp_path):
    torch.set_num_threads(2)
    rng = np.random.default_rng(10)
    train_root, test_root = tmp_path / "train", tmp_path / "test"
    train_root.mkdir()
    test_root.mkdir()
    for root, count in ((train_root, 8), (test_root, 2)):
        lines = []
        for label in range(3):
            for i in range(count):
                name = f"Species_{chr(65 + label)}_{i:04d}_{label * 100 + i}.png"
                Image.fromarray(rng.integers(0, 256, (40, 45, 3), dtype=np.uint8)).save(root / name)
                lines.append(f"{name} {label}")
        (tmp_path / f"{root.name}.txt").write_text("\n".join(lines))
    data = {"train_root": str(train_root), "test_root": str(test_root),
            "train_manifest": str(tmp_path / "train.txt"), "test_manifest": str(tmp_path / "test.txt"),
            "prepared_dir": str(tmp_path / "prepared"), "num_classes": 3, "seed": 42, "val_fraction": 0.25}
    data_path = tmp_path / "data.json"
    write_json(data_path, data)
    prepare(data_path)
    return tmp_path, data_path, data


def arguments(root, data_path, kind="baseline", smoke=False):
    config = {"model": kind, "num_classes": 3, "seed": 42, "image_size": 32, "batch_size": 6,
              "num_workers": 0, "epochs": 2, "head_epochs": 1, "lr": 0.001, "finetune_lr": 0.0001,
              "weight_decay": 0.0001, "augment": kind != "baseline", "dropout": 0.4,
              "patience": 3, "scheduler": True, "amp": False}
    path = root / f"{kind}.json"
    write_json(path, config)
    return Namespace(config=str(path), data=str(data_path), epochs=None, batch_size=None,
                     out=str(root / f"run_{kind}"), resume=False, device="cpu", max_batches=1 if smoke else None)


def test_full_pipeline_from_synthetic_images(synthetic, monkeypatch):
    root, data_path, data = synthetic
    args = arguments(root, data_path)
    # Even an unusable test path cannot affect train.py.
    original = dict(data)
    data["test_root"] = "NEVER_OPEN_THIS_DURING_TRAINING"
    write_json(data_path, data)
    run = train(args)
    write_json(data_path, original)
    selection_path = root / "selection.json"
    selection = select_runs([run], selection_path)
    assert selection["deployment_run"] == "run_baseline"
    eval_args = Namespace(selection=str(selection_path), data=str(data_path), out=str(root / "final"),
                          final=False, device="cpu", acknowledge_integrity_issues=False)
    with pytest.raises(ValueError, match="--final"):
        evaluate(eval_args)
    eval_args.final = True
    scores = evaluate(eval_args)
    assert scores[0]["samples"] == 6
    export = export_space(selection_path, root / "space")
    predictor = Predictor(export / "model.pt")
    image = next(Path(original["test_root"]).glob("*.png"))
    with Image.open(image) as im:
        result = predictor.predict(im)
    assert set(result) == set(read_json(export / "class_names.json"))
    assert sum(result.values()) == pytest.approx(1.0)
    gradcam(export / "model.pt", image, root / "cam.png")
    assert (root / "cam.png").is_file()
    import app
    monkeypatch.setattr(app, "MODEL_PATH", export / "model.pt")
    app.predictor.cache_clear()
    with Image.open(image) as im:
        assert len(app.classify(im)[0]) == 3
    assert app.build_app() is not None


def test_smoke_checkpoint_cannot_enter_final_comparison(synthetic):
    root, data_path, _ = synthetic
    run = train(arguments(root, data_path, "regularized", smoke=True))
    with pytest.raises(ValueError, match="Smoke"):
        select_runs([run], root / "selection.json")


def test_resnet_stage_transition_without_download(synthetic, monkeypatch):
    root, data_path, _ = synthetic
    from src.models import build_model
    monkeypatch.setattr("src.train.build_model", lambda config, pretrained=True: build_model(config, pretrained=False))
    args = arguments(root, data_path, "resnet18")
    run = train(args)
    import csv
    with (run / "history.csv").open() as f:
        assert [r["stage"] for r in csv.DictReader(f)] == ["head", "finetune"]


def test_interrupted_resume_matches_uninterrupted_training(synthetic, monkeypatch):
    root, data_path, _ = synthetic
    from src.engine import run_epoch, load_checkpoint
    args = arguments(root, data_path, "regularized")
    args.out = str(root / "uninterrupted")
    full = train(args)
    counter = [0]
    def interrupt(*a, **kw):
        counter[0] += 1
        if counter[0] == 4:
            raise RuntimeError("simulated interruption")
        return run_epoch(*a, **kw)
    monkeypatch.setattr("src.train.run_epoch", interrupt)
    args.out = str(root / "interrupted")
    with pytest.raises(RuntimeError, match="simulated"):
        train(args)
    monkeypatch.setattr("src.train.run_epoch", run_epoch)
    args.resume = True
    resumed = train(args)
    expected = load_checkpoint(full / "checkpoints" / "last.pt")
    actual = load_checkpoint(resumed / "checkpoints" / "last.pt")
    for name in expected["state_dict"]:
        torch.testing.assert_close(expected["state_dict"][name], actual["state_dict"][name], rtol=0, atol=0)


def test_modified_split_is_rejected(synthetic):
    root, data_path, data = synthetic
    path = Path(data["prepared_dir"]) / "split.json"
    split = read_json(path)
    split["train"][0]["label"] = 2
    write_json(path, split)
    with pytest.raises(ValueError, match="edited"):
        train(arguments(root, data_path))


def test_gradcam_toggle_and_input_validation_guards(synthetic, monkeypatch):
    """Verify Grad-CAM toggle behavior, individual explainability views, and invalid/tiny upload guards."""
    root, data_path, data = synthetic
    run = train(arguments(root, data_path))
    selection_path = root / "selection.json"
    select_runs([run], selection_path)
    export = export_space(selection_path, root / "space_cam")
    # Force a clean TORCH_HOME to prove Predictor never downloads pretrained weights
    monkeypatch.setenv("TORCH_HOME", str(root / "empty_torch_cache"))
    pred = Predictor(export / "model.pt")
    image_path = next(Path(data["test_root"]).glob("*.png"))
    with Image.open(image_path) as im:
        res_on = pred.predict_with_gradcam(im, include_gradcam=True)
        res_off = pred.predict_with_gradcam(im, include_gradcam=False)
    assert res_on["scores"] == res_off["scores"]
    assert res_on["crop"] is not None and res_on["heatmap"] is not None and res_on["overlay"] is not None
    assert res_off["crop"] is None and res_off["heatmap"] is None and res_off["overlay"] is None
    with pytest.raises(ValueError, match="too small"):
        pred.predict(Image.new("RGB", (8, 8)))
    with pytest.raises(ValueError, match="Unsupported"):
        pred.predict("not_an_image")


def test_label_smoothing_applied_in_training_only(synthetic):
    """Verify label_smoothing alters training loss while evaluation loss remains unsmoothed."""
    root, data_path, _ = synthetic
    from src.engine import make_loader, run_epoch
    from src.models import build_model
    from src.utils import load_split, seed_everything

    cfg = {
        "model": "baseline",
        "num_classes": 3,
        "seed": 42,
        "image_size": 32,
        "batch_size": 6,
        "num_workers": 0,
        "dropout": 0.0,
    }
    split = load_split(root / "prepared" / "split.json")
    loader = make_loader(str(root / "train"), split["train"], cfg, augment=False, shuffle=False)

    seed_everything(42)
    m1 = build_model(cfg, pretrained=False).eval()
    eval_unsmoothed = run_epoch(m1, loader, torch.device("cpu"), 3, optimizer=None, label_smoothing=0.0)
    eval_with_ls_arg = run_epoch(m1, loader, torch.device("cpu"), 3, optimizer=None, label_smoothing=0.05)
    # Evaluation must ignore label_smoothing and always return standard unsmoothed cross-entropy
    assert eval_unsmoothed["metrics"]["loss"] == pytest.approx(eval_with_ls_arg["metrics"]["loss"], rel=0, abs=0)

    # Training pass with label_smoothing=0.05 must use CrossEntropyLoss(label_smoothing=0.05)
    seed_everything(42)
    m_train = build_model(cfg, pretrained=False)
    opt = torch.optim.SGD(m_train.parameters(), lr=0.0)  # zero lr to compare exact forward loss on same weights
    scaler = torch.amp.GradScaler("cpu", enabled=False)
    train_ls = run_epoch(m_train, loader, torch.device("cpu"), 3, optimizer=opt, scaler=scaler, amp=False, label_smoothing=0.05)
    assert train_ls["metrics"]["loss"] != pytest.approx(eval_unsmoothed["metrics"]["loss"], abs=1e-6)


