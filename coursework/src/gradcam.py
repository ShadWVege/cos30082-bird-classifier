"""Command-line utility for Gradient-weighted Class Activation Mapping (Grad-CAM).

Learning Objectives & Interpretability Protocol:
1. Visual Explainability: Grad-CAM generates spatial localization maps highlighting
   which regions in the input image influenced the CNN's classification score for a target class.
2. Target Layer: Hooks into the terminal convolutional block (`layer4` in ResNet-18 or
   `features` in ConvNeXt), capturing high-level semantic representations before global pooling.
3. 1:1 Pixel Alignment: The attribution heatmap is overlaid strictly on the deterministic
   CenterCrop (224x224), matching exactly what the model saw during inference.
4. Scientific Limitation Disclosure: Heatmap highlights show correlational feature presence;
   they do not constitute proof of causal biological reasoning.
"""
import argparse
from pathlib import Path
import os
from .utils import ROOT
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))
import numpy as np
import torch
from PIL import Image, ImageOps
from torchvision import transforms as T
import matplotlib
matplotlib.use("Agg")
from matplotlib import colormaps

from .dataset import image_transform
from .engine import load_checkpoint
from .models import build_model


def gradcam(checkpoint_path, image_path, out, target=None):
    """Compute and save Grad-CAM spatial heatmap panel for a target bird photograph."""
    state = load_checkpoint(checkpoint_path)
    config = state["config"]
    model = build_model(config, pretrained=False).eval()
    model.load_state_dict(state["state_dict"])
    layer = model.layer4 if config["model"] == "resnet18" else model.features
    captured = []
    def capture(_module, _inputs, output):
        output.retain_grad()
        captured.append(output)
    hook = layer.register_forward_hook(capture)
    try:
        with Image.open(image_path) as im:
            original = ImageOps.exif_transpose(im).convert("RGB")
        tensor = image_transform(config["image_size"])(original).unsqueeze(0).requires_grad_(True)
        logits = model(tensor)
        target = int(logits.argmax(1)) if target is None else target
        if not 0 <= target < config["num_classes"]:
            raise ValueError("Target class outside model output range")
        logits[0, target].backward()
        activations = captured[0]
        weights = activations.grad.mean(dim=(2, 3), keepdim=True)
        cam = (weights * activations).sum(1, keepdim=True).relu()
        cam = torch.nn.functional.interpolate(cam, size=(config["image_size"], config["image_size"]), mode="bilinear", align_corners=False)[0, 0]
        cam = cam.detach().numpy()
        cam = cam / max(float(cam.max()), 1e-8)
        size = config["image_size"]
        crop = T.CenterCrop(size)(T.Resize(round(size * 256 / 224))(original))
        colour = colormaps["jet"](cam)[..., :3]
        overlay = np.clip(0.55 * np.asarray(crop) + 0.45 * colour * 255, 0, 255).astype(np.uint8)
        panel = Image.new("RGB", (2 * size, size))
        panel.paste(crop, (0, 0))
        panel.paste(Image.fromarray(overlay), (size, 0))
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        panel.save(out)
        print(f"Grad-CAM target: {target} ({state['class_names'][target]}). Left: model crop; right: overlay. Attribution is not proof of causal reasoning.")
    finally:
        hook.remove()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--image", required=True)
    p.add_argument("--out", default="figures/gradcam.png")
    p.add_argument("--target", type=int)
    a = p.parse_args()
    gradcam(a.checkpoint, a.image, a.out, a.target)
