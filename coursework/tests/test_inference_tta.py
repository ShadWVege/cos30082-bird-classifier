"""Unit and integration tests for Test-Time Augmentation (TTA) in inference and Grad-CAM."""
from pathlib import Path
import pytest
import torch
from PIL import Image

import app
from src.inference import Predictor
from src.models import build_model
from src.engine import save_checkpoint


@pytest.fixture
def dummy_predictor(tmp_path):
    config = {"model": "resnet18", "num_classes": 5, "image_size": 32}
    model = build_model(config, pretrained=False).eval()
    ckpt_path = tmp_path / "model.pt"
    class_names = ["Species A", "Species B", "Species C", "Species D", "Species E"]
    save_checkpoint(ckpt_path, {"state_dict": model.state_dict(), "config": config, "class_names": class_names})
    return Predictor(ckpt_path)


def test_tta_single_view_parity(dummy_predictor):
    image = Image.new("RGB", (64, 64), color="blue")
    res_default = dummy_predictor.predict(image)
    res_explicit = dummy_predictor.predict(image, use_tta=False)
    assert res_default == res_explicit


def test_tta_two_view_aggregation(dummy_predictor):
    image = Image.new("RGB", (64, 64), color="green")
    res_tta = dummy_predictor.predict(image, use_tta=True)
    assert len(res_tta) == 5
    assert sum(res_tta.values()) == pytest.approx(1.0, abs=1e-5)
    for p in res_tta.values():
        assert 0.0 <= p <= 1.0


def test_tta_gradcam_attribution(dummy_predictor):
    image = Image.new("RGB", (64, 64), color="yellow")
    # Single-view Grad-CAM
    cam_single = dummy_predictor.predict_with_gradcam(image, include_gradcam=True, use_tta=False)
    assert cam_single["tta_enabled"] is False
    assert cam_single["heatmap"] is not None
    assert cam_single["target_layer"] == "layer4"

    # 2-view TTA Grad-CAM
    cam_tta = dummy_predictor.predict_with_gradcam(image, include_gradcam=True, use_tta=True)
    assert cam_tta["tta_enabled"] is True
    assert cam_tta["heatmap"] is not None
    assert cam_tta["panel"].size == (3 * 32, 32)
    assert cam_tta["target_class_name"] in dummy_predictor.names


def test_app_classify_tta_toggle(dummy_predictor, monkeypatch):
    monkeypatch.setattr(app, "MODEL_PATH", dummy_predictor.model_path)
    app.predictor.cache_clear()
    image = Image.new("RGB", (64, 64), color="red")

    # Single-view default
    scores, text, crop, heat, over, panel = app.classify(image, show_gradcam=True, use_tta=False)
    assert "Standard Single-View" in text
    assert len(scores) == 5

    # Enhanced TTA mode
    scores_tta, text_tta, crop_tta, heat_tta, over_tta, panel_tta = app.classify(image, show_gradcam=True, use_tta=True)
    assert "Enhanced 2-View TTA" in text_tta
    assert len(scores_tta) == 5


def test_input_validation_guards(dummy_predictor):
    """Verify robust rejection of invalid, null, non-image, and undersized inputs."""
    with pytest.raises(ValueError, match="Choose an image first"):
        dummy_predictor.predict(None)

    with pytest.raises(ValueError, match="Unsupported input format"):
        dummy_predictor.predict("not_a_valid_image")

    tiny = Image.new("RGB", (8, 8))
    with pytest.raises(ValueError, match="Image is too small"):
        dummy_predictor.predict(tiny)


def test_gradcam_panel_dimensions_and_channels(dummy_predictor):
    """Verify that Grad-CAM produces exact 3-panel side-by-side RGB visualization."""
    image = Image.new("RGB", (64, 64), color="cyan")
    res = dummy_predictor.predict_with_gradcam(image, include_gradcam=True)
    size = dummy_predictor.config["image_size"]
    assert res["panel"].size == (3 * size, size)
    assert res["panel"].mode == "RGB"
    assert res["crop"].size == (size, size)
    assert res["heatmap"].size == (size, size)
    assert res["overlay"].size == (size, size)


def test_class_mappings_and_name_to_id(dummy_predictor):
    """Verify bijective mapping between zero-based class IDs and species names."""
    assert len(dummy_predictor.names) == 5
    for idx, name in enumerate(dummy_predictor.names):
        assert dummy_predictor.name_to_id[name] == idx


