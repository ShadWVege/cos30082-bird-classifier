"""Portable CPU/GPU inference engine with deterministic preprocessing and optional layer4 Grad-CAM."""
import ctypes
import gc
import os
from pathlib import Path
import sys
import threading
import numpy as np
import torch
from PIL import Image, ImageOps
from torchvision import transforms as T
from .dataset import image_transform
from .engine import load_checkpoint
from .models import build_model

# Guard against decompression-bomb uploads on 512 MiB cloud instances while allowing standard DSLR photos.
Image.MAX_IMAGE_PIXELS = 25_000_000


def _release_unused_memory():
    """Release unreachable Python objects and return freed glibc heap pages to the Linux kernel."""
    gc.collect()
    if sys.platform.startswith("linux"):
        try:
            ctypes.CDLL("libc.so.6").malloc_trim(0)
        except OSError:
            pass


def _jet_colormap(cam):
    """Pure-NumPy jet colormap mapping normalized [0, 1] activation maps to uint8 RGB (H, W, 3).

    Design choice: Implementing the piecewise linear jet mapping directly in NumPy avoids
    requiring matplotlib inside the lightweight cloud container while matching
    standard heat-map visual conventions (blue = low activation, red = high activation).
    """
    r = np.clip(1.5 - np.abs(4.0 * cam - 3.0), 0.0, 1.0)
    g = np.clip(1.5 - np.abs(4.0 * cam - 2.0), 0.0, 1.0)
    b = np.clip(1.5 - np.abs(4.0 * cam - 1.0), 0.0, 1.0)
    return (np.stack([r, g, b], axis=-1) * 255.0).astype(np.uint8)


class Predictor:
    """Encapsulates the validation-selected checkpoint, class vocabulary, and Grad-CAM hooks.

    Learning objective: Guarantee strict train/validation/test/deployment parity by loading
    both the architecture configuration (`config`) and the exact 200-species class order
    (`class_names`) directly from the serialized checkpoint dictionary, and applying the
    shared deterministic `image_transform` from `src/dataset.py`.
    """

    def __init__(self, model_path, device="cpu"):
        if not Path(model_path).is_file():
            raise FileNotFoundError(
                "No trained model exported yet. Train, compare, then run python -m src.export_space."
            )
        self.model_path = Path(model_path)
        self.device = torch.device(device)
        self._lock = threading.RLock()
        if self.device.type == "cpu":
            # Limit intra-op CPU threads on constrained containers (e.g. Render 512 MiB / 0.1 CPU)
            # so PyTorch does not spawn per-core OpenMP thread arenas on multi-socket cloud hosts.
            torch.set_num_threads(max(1, int(os.environ.get("OMP_NUM_THREADS", "1"))))

        # On Linux CPU containers, memory-map model.pt (mmap=True) so the 43.10 MiB weights stay
        # backed by reclaimable file cache (RssFile) rather than anonymous heap (RssAnon).
        use_mmap = self.device.type == "cpu" and sys.platform.startswith("linux")
        checkpoint = load_checkpoint(model_path, mmap=use_mmap)
        self.names = checkpoint["class_names"]
        self.config = checkpoint["config"]
        self.name_to_id = {name: idx for idx, name in enumerate(self.names)}

        # Build architecture with pretrained=False because trained weights come from state_dict.
        # On CPU, instantiate on the zero-allocation 'meta' device and assign state_dict tensors
        # directly (assign=True) so weights are never duplicated in RAM during startup.
        if self.device.type == "cpu":
            with torch.device("meta"):
                self.model = build_model(self.config, pretrained=False)
            self.model.load_state_dict(checkpoint["state_dict"], assign=True)
            self.model = self.model.eval()
        else:
            self.model = build_model(self.config, pretrained=False).to(self.device).eval()
            self.model.load_state_dict(checkpoint["state_dict"])

        # Inference-only model: disable parameter gradients across all 11.28M weights to prevent
        # allocating 43.03 MiB of persistent .grad tensors during Grad-CAM attribution.
        self.model.requires_grad_(False)
        self.transform = image_transform(self.config["image_size"])
        del checkpoint
        _release_unused_memory()

    def _validate_image(self, image):
        """Validate uploaded image type and dimensions, applying EXIF orientation correction."""
        if image is None:
            raise ValueError("Choose an image first")
        if not isinstance(image, Image.Image):
            raise ValueError(
                "Unsupported input format: please upload a valid RGB bird photograph (JPEG, PNG, or WebP)."
            )
        w, h = image.size
        if w < 16 or h < 16:
            raise ValueError(
                f"Image is too small ({w}x{h}px); please upload a clearer photograph (>=16x16px)."
            )
        if w * h > 25_000_000:
            raise ValueError(
                f"Image resolution ({w}x{h}px) exceeds the 25 MP safety limit; please upload a standard photograph."
            )
        image = ImageOps.exif_transpose(image).convert("RGB")
        w, h = image.size
        # Safely bound oversized multi-megapixel uploads (>2048px longest side) before preprocessing
        # to prevent transient RGB buffer spikes on 512 MiB cloud containers. All CUB-200 benchmark
        # images (<=500px) are well below this cap and remain 100% untouched.
        if max(w, h) > 2048:
            scale = 2048.0 / float(max(w, h))
            image = image.resize(
                (max(16, round(w * scale)), max(16, round(h * scale))),
                Image.Resampling.BILINEAR,
            )
        return image

    @torch.inference_mode()
    def predict(self, image, use_tta=False):
        """Compute top-5 softmax probabilities using inference_mode (no gradient graph allocated).

        When use_tta=True, runs sequential inference on original and horizontally flipped images
        and averages the resulting 200-class softmax probabilities. Sequential execution
        preserves peak memory (batch size 1).
        """
        with self._lock:
            image = self._validate_image(image)
            tensor_orig = self.transform(image).unsqueeze(0).to(self.device)
            probs_orig = self.model(tensor_orig).softmax(dim=1)[0]
            del tensor_orig

            if use_tta:
                flipped = image.transpose(Image.FLIP_LEFT_RIGHT)
                tensor_flip = self.transform(flipped).unsqueeze(0).to(self.device)
                probs_flip = self.model(tensor_flip).softmax(dim=1)[0]
                del tensor_flip
                probabilities = 0.5 * (probs_orig + probs_flip)
            else:
                probabilities = probs_orig

            top = probabilities.topk(min(5, len(self.names)))
            # Probabilities use ALL 200 classes in the softmax denominator, not a renormalized top-5 subset.
            return {self.names[int(i)]: float(p) for p, i in zip(top.values, top.indices)}

    def predict_with_gradcam(self, image, include_gradcam=True, use_tta=False):
        """Run prediction and optionally compute layer4 Grad-CAM components.

        Design choice: When `include_gradcam=False`, this method delegates to `predict()` under
        `@torch.inference_mode()`, completely skipping forward hooks and the backward pass to save
        compute and latency. When `include_gradcam=True`, it attaches gradient tracking exclusively
        to the final convolutional block output (`layer4` in ResNet18, $7\\times 7$ spatial grid).

        When `use_tta=True`, it evaluates the horizontally flipped view sequentially under `no_grad()`,
        averages softmax probabilities to determine the top-1 predicted class, and computes the
        original-view `layer4` Grad-CAM attribution for that selected target class.
        """
        with self._lock:
            image = self._validate_image(image)
            size = self.config["image_size"]
            if not include_gradcam:
                scores = self.predict(image, use_tta=use_tta)
                records = [
                    {"rank": r, "label": self.name_to_id[k], "name": k, "prob": v}
                    for r, (k, v) in enumerate(scores.items(), 1)
                ]
                return {
                    "scores": scores,
                    "records": records,
                    "crop": None,
                    "heatmap": None,
                    "overlay": None,
                    "panel": None,
                    "tta_enabled": bool(use_tta),
                }

            probs_flip = None
            if use_tta:
                flipped = image.transpose(Image.FLIP_LEFT_RIGHT)
                with torch.no_grad():
                    tensor_flip = self.transform(flipped).unsqueeze(0).to(self.device)
                    probs_flip = self.model(tensor_flip).softmax(dim=1)[0].cpu()
                    del tensor_flip

            layer = self.model.layer4 if self.config["model"] == "resnet18" else self.model.features
            captured = []

            def capture(_mod, _inp, out):
                out = out.detach().requires_grad_(True)
                out.retain_grad()
                captured.append(out)
                return out

            hook = layer.register_forward_hook(capture)
            try:
                tensor = self.transform(image).unsqueeze(0).to(self.device)
                logits = self.model(tensor)
                probs_orig = logits.detach().float().softmax(dim=1)[0].cpu()
                if probs_flip is not None:
                    probabilities = 0.5 * (probs_orig + probs_flip)
                else:
                    probabilities = probs_orig

                top = probabilities.topk(min(5, len(self.names)))
                scores = {self.names[int(i)]: float(p) for p, i in zip(top.values, top.indices)}
                records = [
                    {"rank": r, "label": int(i), "name": self.names[int(i)], "prob": float(p)}
                    for r, (p, i) in enumerate(zip(top.values, top.indices), 1)
                ]

                # Backpropagate the unnormalized logit of the top-1 predicted class through avgpool+fc to layer4
                target = int(top.indices[0])
                logits[0, target].backward()
                activations = captured[0]
                weights = activations.grad.mean(dim=(2, 3), keepdim=True)
                cam = (weights * activations).sum(1, keepdim=True).relu()
                cam = (
                    torch.nn.functional.interpolate(
                        cam, size=(size, size), mode="bilinear", align_corners=False
                    )[0, 0]
                    .detach()
                    .cpu()
                    .numpy()
                )
                cam = cam / max(float(cam.max()), 1e-8)

                # Deterministic evaluation crop matching image_transform(size) before tensor normalization
                crop_img = T.CenterCrop(size)(T.Resize(round(size * 256 / 224))(image))
                heat_rgb = _jet_colormap(cam)
                overlay_rgb = np.clip(0.55 * np.asarray(crop_img) + 0.45 * heat_rgb, 0, 255).astype(
                    np.uint8
                )
                heatmap_img = Image.fromarray(heat_rgb)
                overlay_img = Image.fromarray(overlay_rgb)

                panel = Image.new("RGB", (3 * size, size))
                panel.paste(crop_img, (0, 0))
                panel.paste(heatmap_img, (size, 0))
                panel.paste(overlay_img, (2 * size, 0))
                return {
                    "scores": scores,
                    "records": records,
                    "crop": crop_img,
                    "heatmap": heatmap_img,
                    "overlay": overlay_img,
                    "panel": panel,
                    "target_class_id": target,
                    "target_class_name": self.names[target],
                    "target_layer": "layer4" if self.config["model"] == "resnet18" else "features",
                    "tta_enabled": bool(use_tta),
                }
            finally:
                hook.remove()
                captured.clear()
                _release_unused_memory()
