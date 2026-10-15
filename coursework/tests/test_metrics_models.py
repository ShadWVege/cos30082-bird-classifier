import numpy as np
import pytest
import torch
from src.metrics import classification_metrics
from src.models import build_model, set_stage, training_mode
from src.dataset import image_transform
from PIL import Image


def test_metrics_macro_recall_not_overall_accuracy():
    scores, rows, cm = classification_metrics([0, 0, 0, 1], [[2, 0], [2, 0], [0, 2], [0, 2]], 2)
    assert scores["top1"] == 0.75
    assert scores["average_accuracy_per_class"] == pytest.approx(5 / 6)
    assert cm.tolist() == [[2, 1], [0, 1]]
    assert rows[0]["support"] == 3
    with pytest.raises(ValueError, match="Missing class"):
        classification_metrics([0], [[1, 0]], 2)
    scores, _, _ = classification_metrics([5], [[6, 5, 4, 3, 2, 1]], 6, require_all=False)
    assert scores["top5"] == 0


@pytest.mark.parametrize("kind", ["baseline", "regularized", "resnet18"])
def test_models_output_logits_and_checkpoint_roundtrip(kind, tmp_path):
    from src.engine import load_checkpoint, save_checkpoint
    torch.set_num_threads(2)
    config = {"model": kind, "num_classes": 200}
    model = build_model(config, pretrained=False).eval()
    image = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        expected = model(image)
    assert expected.shape == (2, 200)
    path = tmp_path / "model.pt"
    save_checkpoint(path, {"state_dict": model.state_dict(), "config": config})
    restored = build_model(config, pretrained=False).eval()
    restored.load_state_dict(load_checkpoint(path)["state_dict"])
    with torch.no_grad():
        torch.testing.assert_close(expected, restored(image))


def test_transfer_freezes_backbone_parameters_and_bn_statistics():
    model = build_model({"model": "resnet18", "num_classes": 3}, pretrained=False)
    set_stage(model, "resnet18", "head")
    training_mode(model, "resnet18", "head")
    before = model.bn1.running_mean.clone()
    assert all(n.startswith("fc.") for n, p in model.named_parameters() if p.requires_grad)
    model(torch.randn(2, 3, 32, 32)).sum().backward()
    torch.testing.assert_close(before, model.bn1.running_mean)
    set_stage(model, "resnet18", "finetune")
    training_mode(model, "resnet18", "finetune")
    assert model.layer4.training and not model.layer3.training
    assert model.layer4[0].conv1.weight.requires_grad
    assert not model.layer3[0].conv1.weight.requires_grad


def test_transfer_layer3_layer4_finetuning():
    model = build_model({"model": "resnet18", "num_classes": 3}, pretrained=False)
    set_stage(model, "resnet18", "finetune", finetune_blocks=("layer3", "layer4"))
    training_mode(model, "resnet18", "finetune", finetune_blocks=("layer3", "layer4"), freeze_bn=True)
    assert model.layer4[0].conv1.weight.requires_grad
    assert model.layer3[0].conv1.weight.requires_grad
    assert not model.layer2[0].conv1.weight.requires_grad
    assert not model.layer1[0].conv1.weight.requires_grad
    # Verify BN running stats are preserved in eval mode when freeze_bn=True
    assert not model.layer3[0].bn1.training
    assert not model.layer4[0].bn1.training


def test_inference_preprocessing_is_deterministic():
    im = Image.new("RGB", (121, 300), "green")
    transform = image_transform(32)
    assert transform(im).shape == (3, 32, 32)
    torch.testing.assert_close(transform(im), transform(im))


def test_metrics_perfect_prediction_and_macro_f1():
    """Verify that perfect diagonal predictions yield 1.0 across Top-1, Avg/Class, Top-5, and F1."""
    targets = [0, 1, 2, 3]
    logits = [[10.0, 0.0, 0.0, 0.0],
              [0.0, 10.0, 0.0, 0.0],
              [0.0, 0.0, 10.0, 0.0],
              [0.0, 0.0, 0.0, 10.0]]
    scores, per_class, cm = classification_metrics(targets, logits, 4)
    assert scores["top1"] == 1.0
    assert scores["average_accuracy_per_class"] == 1.0
    assert scores["top5"] == 1.0
    assert scores["macro_f1"] == 1.0
    assert np.all(cm == np.eye(4, dtype=int))


def test_convnext_tiny_model_construction():
    """Verify that build_model correctly surgery-replaces classifier head on ConvNeXt-Tiny."""
    config = {"model": "convnext_tiny", "num_classes": 200}
    model = build_model(config, pretrained=False).eval()
    assert model.classifier[2].out_features == 200
    dummy = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        out = model(dummy)
    assert out.shape == (1, 200)


def test_custom_cnn_parameter_counts_and_checkpoint_compatibility():
    """Regression test: verify exact parameter counts and strict checkpoint loading compatibility.

    Verifies:
    1. Baseline E1 has exactly 439,816 parameters (387,936 conv + 480 conv bias + 51,400 fc).
    2. Regularized E2 has exactly 440,296 parameters (387,936 conv + 0 bias + 960 BN + 51,400 fc).
    3. No-Dropout E4 has exactly 440,296 parameters.
    4. Reconciles the 480 parameter difference (conv bias omission under BatchNorm).
    5. Verifies strict load_state_dict on actual historical checkpoints if present.
    """
    from pathlib import Path
    from src.models import CustomCNN

    # 1. Parameter counts
    e1_model = build_model({"model": "baseline", "num_classes": 200}, pretrained=False)
    e1_params = sum(p.numel() for p in e1_model.parameters())
    assert e1_params == 439816, f"Expected 439,816, got {e1_params}"

    e2_model = build_model({"model": "regularized", "num_classes": 200, "dropout": 0.4}, pretrained=False)
    e2_params = sum(p.numel() for p in e2_model.parameters())
    assert e2_params == 440296, f"Expected 440,296, got {e2_params}"

    e4_model = build_model({"model": "regularized", "num_classes": 200, "dropout": 0.0}, pretrained=False)
    e4_params = sum(p.numel() for p in e4_model.parameters())
    assert e4_params == 440296, f"Expected 440,296, got {e4_params}"

    # Verify difference is exactly 480 (960 BN parameters - 480 omitted conv biases)
    assert e2_params - e1_params == 480

    # 2. Strict checkpoint loading verification
    checkpoints = {
        "baseline": Path("outputs/baseline_001/checkpoints/best.pt"),
        "regularized": Path("outputs/experiment_001/regularized/checkpoints/best.pt"),
        "no_dropout": Path("outputs/experiment_001/E4_regularized_no_dropout/checkpoints/best.pt"),
    }
    for model_type, ckpt_path in checkpoints.items():
        if ckpt_path.is_file():
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            sd = ckpt["state_dict"]
            target_model = e1_model if model_type == "baseline" else e2_model
            # strict=True guarantees zero missing keys and zero unexpected keys
            load_result = target_model.load_state_dict(sd, strict=True)
            assert len(load_result.missing_keys) == 0
            assert len(load_result.unexpected_keys) == 0


