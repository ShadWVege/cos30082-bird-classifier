"""Generate a high-legibility 2-case Grad-CAM figure comparing E3 ResNet18 vs E7 ConvNeXt-Tiny."""
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms as T
from torchvision.models import convnext_tiny, resnet18

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def jet_colormap(cam):
    r = np.clip(1.5 - np.abs(4.0 * cam - 3.0), 0.0, 1.0)
    g = np.clip(1.5 - np.abs(4.0 * cam - 2.0), 0.0, 1.0)
    b = np.clip(1.5 - np.abs(4.0 * cam - 1.0), 0.0, 1.0)
    return (np.stack([r, g, b], axis=-1) * 255.0).astype(np.uint8)


def get_cam(model, target_layer, tensor, target_class=None):
    captured = []
    def hook(_m, _i, out):
        out = out.detach().requires_grad_(True)
        out.retain_grad()
        captured.append(out)
        return out

    h = target_layer.register_forward_hook(hook)
    try:
        logits = model(tensor)
        probs = F.softmax(logits, dim=-1)[0]
        if target_class is None:
            target_class = int(probs.argmax())
        score = logits[0, target_class]
        score.backward()
        acts = captured[0]
        grads = acts.grad
        weights = grads.mean(dim=(2, 3), keepdim=True)
        cam = torch.clamp((weights * acts).sum(dim=1, keepdim=True), min=0)
        cam = F.interpolate(cam, size=(224, 224), mode="bilinear", align_corners=False)
        cam_np = cam[0, 0].detach().cpu().numpy()
        cam_max = cam_np.max()
        cam_norm = cam_np / cam_max if cam_max > 0 else cam_np
        return target_class, probs, cam_norm
    finally:
        h.remove()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load E3
    e3_ckpt = torch.load("outputs/experiment_001/resnet18/checkpoints/best.pt", map_location=device)
    class_names = e3_ckpt["class_names"]
    e3_model = resnet18(weights=None)
    e3_model.fc = nn.Linear(512, len(class_names))
    e3_model.load_state_dict(e3_ckpt["state_dict"])
    e3_model.to(device).eval().requires_grad_(False)

    # Load E7
    e7_ckpt = torch.load("outputs/experiment_001/E7_convnext_tiny/checkpoints/best.pt", map_location=device)
    e7_model = convnext_tiny(weights=None)
    e7_model.classifier[2] = nn.Linear(e7_model.classifier[2].in_features, len(class_names))
    e7_model.load_state_dict(e7_ckpt["state_dict"])
    e7_model.to(device).eval().requires_grad_(False)

    transform = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(MEAN, STD),
    ])
    crop_eval = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
    ])

    # Resolve train image root dynamically
    train_root = Path("D:/COS30082_Datasets/Train")
    cfg_path = Path("configs/data.local.json")
    if cfg_path.exists():
        try:
            import json
            cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
            if "train_dir" in cfg and Path(cfg["train_dir"]).exists():
                train_root = Path(cfg["train_dir"])
        except Exception:
            pass

    # Two test cases:
    # 1. Red-headed Woodpecker (Deployment Demo Image)
    # 2. American Goldfinch (Species with yellow plumage and black cap)
    test_cases = [
        {
            "title": "Case 1: Red-headed Woodpecker (Live Deployment Test Image)",
            "path": train_root / "Red_headed_Woodpecker_0012_2578571305.jpg",
            "true_name": "Red headed Woodpecker",
            "true_id": 190,
        },
        {
            "title": "Case 2: American Goldfinch (Diagnostic Plumage Landmark Focus)",
            "path": train_root / "American_Goldfinch_0013_146872871.jpg",
            "true_name": "American Goldfinch",
            "true_id": 46,
        },
    ]

    fig, axes = plt.subplots(2, 4, figsize=(15, 6.8), layout="constrained")

    for row_idx, tc in enumerate(test_cases):
        img_raw = Image.open(tc["path"]).convert("RGB")
        crop_img = crop_eval(img_raw)
        crop_np = np.asarray(crop_img)
        tensor = transform(img_raw).unsqueeze(0).to(device)

        # E3 CAM
        e3_tc, e3_probs, e3_cam = get_cam(e3_model, e3_model.layer4, tensor)
        e3_heat = jet_colormap(e3_cam)
        e3_overlay = np.clip(0.55 * crop_np + 0.45 * e3_heat, 0, 255).astype(np.uint8)

        # E7 CAM
        e7_tc, e7_probs, e7_cam = get_cam(e7_model, e7_model.features[-1], tensor)
        e7_heat = jet_colormap(e7_cam)
        e7_overlay = np.clip(0.55 * crop_np + 0.45 * e7_heat, 0, 255).astype(np.uint8)

        # Plot Col 0: Crop
        ax0 = axes[row_idx, 0]
        ax0.imshow(crop_np)
        ax0.set_title(f"True: [{tc['true_id']:03d}] {tc['true_name']}\n224x224 Deterministic Crop", fontsize=9.5, fontweight="bold")
        ax0.axis("off")

        # Plot Col 1: E3 Overlay
        ax1 = axes[row_idx, 1]
        ax1.imshow(e3_overlay)
        e3_conf = float(e3_probs[e3_tc]) * 100.0
        ax1.set_title(f"E3 ResNet18 (layer4)\nPred: [{e3_tc:03d}] {class_names[e3_tc]} ({e3_conf:.1f}%)", fontsize=9.5, fontweight="bold", color="#0f3460")
        ax1.axis("off")

        # Plot Col 2: E7 Overlay
        ax2 = axes[row_idx, 2]
        ax2.imshow(e7_overlay)
        e7_conf = float(e7_probs[e7_tc]) * 100.0
        ax2.set_title(f"E7 ConvNeXt-Tiny (features.7)\nPred: [{e7_tc:03d}] {class_names[e7_tc]} ({e7_conf:.1f}%)", fontsize=9.5, fontweight="bold", color="#1b5e20")
        ax2.axis("off")

        # Plot Col 3: Softmax Top-3 comparison bar chart
        ax3 = axes[row_idx, 3]
        top3_e3 = e3_probs.topk(3)
        top3_e7 = e7_probs.topk(3)

        # Common union of top classes
        union_classes = list(dict.fromkeys([int(i) for i in top3_e7.indices] + [int(i) for i in top3_e3.indices]))[:3]
        y_pos = np.arange(len(union_classes))
        w = 0.35

        vals_e3 = [float(e3_probs[c]) * 100.0 for c in union_classes]
        vals_e7 = [float(e7_probs[c]) * 100.0 for c in union_classes]
        names = [f"[{c:03d}] {class_names[c][:18]}" for c in union_classes]

        ax3.barh(y_pos - w/2, vals_e3, height=w, label="E3 ResNet18", color="#3b82f6", alpha=0.9)
        ax3.barh(y_pos + w/2, vals_e7, height=w, label="E7 ConvNeXt", color="#10b981", alpha=0.9)
        ax3.set_yticks(y_pos)
        ax3.set_yticklabels(names, fontsize=8.5)
        ax3.invert_yaxis()
        ax3.set_xlim(0, 115)
        ax3.set_xlabel("Probability (%)", fontsize=8.5)
        ax3.set_title(f"{tc['title'].split(':')[0]} Softmax Comparison", fontsize=9.0, fontweight="bold")
        ax3.legend(loc="lower right", fontsize=8.0)
        ax3.grid(axis="x", alpha=0.3)

        for i, (v3, v7) in enumerate(zip(vals_e3, vals_e7)):
            ax3.text(v3 + 1.5, i - w/2, f"{v3:.1f}%", va="center", fontsize=7.5, color="#1e40af")
            ax3.text(v7 + 1.5, i + w/2, f"{v7:.1f}%", va="center", fontsize=7.5, color="#065f46")

    fig.suptitle(
        "Grad-CAM Feature Attribution Comparison: E3 ResNet18 (layer4) vs E7 ConvNeXt-Tiny (features.7)\n"
        "(Saliency maps indicate localized regions contributing to target logits; they do not prove causal reasoning)",
        fontsize=11.5,
        fontweight="bold",
    )
    out_path = Path("outputs/experiment_001/figures/gradcam_focused.png")
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved focused Grad-CAM comparison figure to {out_path}!")


if __name__ == "__main__":
    main()

