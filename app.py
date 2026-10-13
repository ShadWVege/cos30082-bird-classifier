"""Local Gradio application and Render / Hugging Face Spaces entry point; no random model fallback."""
from functools import lru_cache
import json
import os
from pathlib import Path

# Constrain glibc heap arenas and OpenMP thread pools before importing PyTorch so 512 MiB
# cloud containers (e.g. Render Free Web Service) do not allocate per-core thread arenas.
os.environ.setdefault("MALLOC_ARENA_MAX", "2")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import gradio as gr
from src.inference import Predictor

# Optional Hugging Face ZeroGPU decorator support; falls back to a no-op decorator on CPU/local runs.
try:
    import spaces  # type: ignore

    gpu_decorator = spaces.GPU(duration=30)
except ImportError:

    def gpu_decorator(fn):
        return fn


ROOT_DIR = Path(__file__).parent
MODEL_PATH = Path(os.environ.get("BIRD_MODEL_PATH", str(ROOT_DIR / "model.pt")))

DEFAULT_NOTE = (
    "Upload a clear bird photograph and click **Identify Bird Species** to display the "
    "predicted species name, class code (`000–199`), and 200-class softmax scores."
)


@lru_cache(maxsize=1)
def predictor():
    """Lazily load and cache the validation-selected checkpoint once per process."""
    if MODEL_PATH.is_file():
        return Predictor(MODEL_PATH)
    fallback = ROOT_DIR / "outputs" / "huggingface_space" / "model.pt"
    if fallback.is_file():
        return Predictor(fallback)
    return Predictor(MODEL_PATH)


def _model_ready():
    return MODEL_PATH.is_file() or (ROOT_DIR / "outputs" / "huggingface_space" / "model.pt").is_file()


def _resolve_asset(filename):
    """Locate static chart assets in either standalone Space (`assets/`) or repo (`outputs/...`)."""
    candidates = [
        ROOT_DIR / "assets" / filename,
        ROOT_DIR / "outputs" / "experiment_001" / "figures" / filename,
        ROOT_DIR / "outputs" / "final_test_results" / "figures" / filename,
    ]
    for path in candidates:
        if path.is_file():
            return str(path)
    return None


def _discover_examples():
    """Load packaged example images if present, or dynamically read local validation config in dev mode."""
    examples_dir = ROOT_DIR / "examples"
    if examples_dir.is_dir():
        found = sorted(
            str(p)
            for p in examples_dir.iterdir()
            if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp")
        )
        if found:
            return found[:4]
    local_cfg = ROOT_DIR / "configs" / "data.local.json"
    split_cfg = ROOT_DIR / "data" / "prepared" / "split.json"
    if local_cfg.is_file() and split_cfg.is_file():
        try:
            train_root = Path(json.loads(local_cfg.read_text(encoding="utf-8")).get("train_root", ""))
            val_records = json.loads(split_cfg.read_text(encoding="utf-8")).get("val", [])
            preferred = [
                "Red_headed_Woodpecker_0012_2578571305.jpg",
                "Scarlet_Tanager_0022_2227619316.jpg",
                "Mourning_Warbler_0018_392495285.jpg",
            ]
            by_name = {r["name"]: train_root / r["relative_path"] for r in val_records}
            found = [str(by_name[n]) for n in preferred if n in by_name and by_name[n].is_file()]
            if found:
                return found
        except OSError:
            pass
    return []


@gpu_decorator
def classify(image, show_gradcam=True):
    """Run deterministic CUB-200 inference and optional layer4 Grad-CAM attribution.

    Returns a 6-tuple whose first element (`[0]`) is the top-k `{species_name: probability}` dict
    for backward compatibility with automated unit tests (`tests/test_workflow.py`).
    """
    if image is None:
        raise gr.Error("Please upload a bird photograph first (supported formats: JPEG, PNG, WebP).")
    try:
        result = predictor().predict_with_gradcam(image, include_gradcam=bool(show_gradcam))
    except (FileNotFoundError, ValueError, OSError, TypeError) as error:
        raise gr.Error(str(error)) from error

    scores = result["scores"]
    records = result["records"]
    top1 = records[0]
    best = top1["prob"]
    caution = (
        "Low top-1 margin (<50% softmax mass) — several visually similar species share probability mass; "
        "inspect the Top-5 candidates and plumage markings carefully."
        if best < 0.5
        else "High top-1 concentration — compare key field marks (bill shape, wing bars, crown plumage) against the Top-5 candidates below."
    )
    gradcam_status = (
        f"Computed `layer4` (7×7) Grad-CAM for predicted target **`[{top1['label']:03d}] {top1['name']}`**."
        if show_gradcam
        else "Grad-CAM computation was skipped (`Compute layer4 Grad-CAM Explainability` unchecked) — ran fast forward-only `@torch.inference_mode()` pass."
    )
    table_lines = [
        f"### Predicted Species: **{top1['name']}** &nbsp;|&nbsp; Zero-Based Class ID: **`{top1['label']:03d}`** (`{top1['label']}` in `0–199`) &nbsp;|&nbsp; Top-1 Softmax: **`{best * 100:.2f}%`**",
        "",
        "| Rank | Predicted Bird Species (Common Name) | Zero-Based Class ID (`0–199`) | 200-Class Softmax Score |",
        "|:---:|---|:---:|---:|",
    ]
    for item in records:
        table_lines.append(
            f"| **#{item['rank']}** | **{item['name']}** | `{item['label']:03d}` (`{item['label']}`) | `{item['prob'] * 100:.2f}%` |"
        )
    table_lines.extend(
        [
            "",
            f"- **Prediction Guidance:** {caution}",
            f"- **Explainability Status:** {gradcam_status}",
            "- **Closed-Set Scope (200 Trained Species Only):** This classifier recognises **only the 200 North American bird species (`000–199`)** from the CUB-200 benchmark. Softmax scores sum to 100% across those 200 classes and **are not calibrated confidence**; out-of-distribution species or non-bird images will still be assigned to the closest learned class.",
        ]
    )
    return (
        scores,
        "\n".join(table_lines),
        result["crop"],
        result["heatmap"],
        result["overlay"],
        result["panel"],
    )


def reset_ui():
    """Reset image upload, predictions, and Grad-CAM panels back to clean initial state."""
    return None, None, DEFAULT_NOTE, None, None, None, None


def build_app():
    """Construct the three-section responsive Gradio interface."""
    learning_curves_path = _resolve_asset("model_comparison.png")
    ablation_curves_path = _resolve_asset("ablation_e2_vs_e4.png")
    test_summary_path = _resolve_asset("test_metrics_summary.png")

    with gr.Blocks(title="Bird Species Explorer — COS30082") as demo:
        gr.Markdown(
            "# Bird Species Explorer — COS30082 Assignment 1\n"
            "Fine-grained visual classification across **200 bird species (`000–199`)** using our "
            "validation-selected **Staged ResNet18 Transfer Learning (E3)** model (**69.66% Val Top-1** / "
            "**70.43% Official Held-Out Test Top-1**)."
        )
        if not _model_ready():
            gr.Markdown(
                "**Setup pending:** a trained model must be exported (`python -m src.export_space`) before predictions are available."
            )

        with gr.Tabs():
            with gr.TabItem("1. Bird Identification & Grad-CAM Explainability"):
                # =============================================================
                # SECTION 1 — Bird Identification
                # =============================================================
                gr.Markdown("## Section 1 — Bird Identification")
                with gr.Row():
                    with gr.Column(scale=5, min_width=300):
                        photo = gr.Image(
                            type="pil",
                            label="Upload Bird Photograph (JPEG / PNG / WebP)",
                            sources=["upload"],
                        )
                        show_cam = gr.Checkbox(
                            value=True,
                            label="Compute layer4 Grad-CAM Explainability (uncheck for faster forward-only inference)",
                        )
                        with gr.Row():
                            submit = gr.Button(
                                "Identify Bird Species",
                                variant="primary",
                                interactive=_model_ready(),
                            )
                            clear = gr.Button(value="Clear / Reset")
                        gr.Markdown(
                            "**Closed-Set Notice:** This model recognises **only the 200 trained CUB-200 bird species (`000–199`)**. "
                            "Uploading non-bird images or species outside CUB-200 will still yield a 200-way softmax distribution."
                        )
                        example_paths = _discover_examples()
                        if example_paths:
                            gr.Examples(
                                examples=[[p] for p in example_paths],
                                inputs=[photo],
                                label="Development Validation Examples (Click to load)",
                            )

                    with gr.Column(scale=6, min_width=320):
                        prediction = gr.Label(
                            num_top_classes=5,
                            label="Top-5 Predicted Species (200-Class Softmax Percentage Bars)",
                        )
                        note = gr.Markdown(DEFAULT_NOTE)

                # =============================================================
                # SECTION 2 — Grad-CAM Explainability
                # =============================================================
                gr.Markdown(
                    "---\n"
                    "## Section 2 — Grad-CAM Explainability (`layer4` Convolutional Attribution)\n"
                    "When **Compute layer4 Grad-CAM Explainability** is enabled above, the panels below display "
                    "the exact **224×224 deterministic evaluation crop**, the **7×7 `layer4` "
                    "activation heatmap** upsampled to 224×224, and the **spatial overlay** for the #1 predicted class."
                )
                with gr.Row():
                    crop_out = gr.Image(
                        type="pil",
                        label="1. Deterministic Evaluation Crop (224x224 RGB)",
                    )
                    heatmap_out = gr.Image(
                        type="pil",
                        label="2. ResNet18 layer4 Grad-CAM Heatmap (7x7 -> 224x224)",
                    )
                    overlay_out = gr.Image(
                        type="pil",
                        label="3. Grad-CAM Heatmap Overlay on Evaluation Crop",
                    )
                with gr.Accordion("Combined 3-Panel Grad-CAM Strip & Interpretability Limitations", open=False):
                    cam_panel_out = gr.Image(
                        type="pil",
                        label="Combined 3-Panel Strip (Crop | layer4 Heatmap | Overlay)",
                    )
                    gr.Markdown(
                        "### What Grad-CAM Does and Does Not Prove\n"
                        "- **What it shows:** Grad-CAM weights the 512 channels of ResNet18's final convolutional block (`layer4`, spatial resolution 7×7) by the global-average-pooled gradients of the top-1 predicted class logit, highlighting coarse image regions that positively contributed to that logit.\n"
                        "- **What it does NOT prove:** Grad-CAM is a **post-hoc spatial correlation map, not causal proof of taxonomic reasoning**. Because `layer4` has a coarse 7×7 receptive field (32×32 pixel stride), heatmaps naturally spill onto adjacent perches, foliage, or water and cannot isolate fine anatomical concepts (e.g., bill culmen vs tarsus color) on their own."
                    )

            with gr.TabItem("2. Model Insights, Ablation & Benchmark Results"):
                # =============================================================
                # SECTION 3 — Model Insights
                # =============================================================
                gr.Markdown(
                    "## Section 3 — Model Insights & Experimental Comparison\n"
                    "All four candidate models were trained and compared strictly on the **frozen grouped stratified "
                    "development split** (`3,870` training / `959` validation images across `200` classes, seed `42`) "
                    "before performing the **one-time official held-out test evaluation** (`1,204` test images).\n\n"
                    "### E1 / E2 / E4 / E3 Experimental Summary\n"
                    "| Experiment | Architecture & Key Mechanism | Params | Best Ep | Clean-Train Top-1 | Val Top-1 | Val Avg/Class | Val Top-5 | Best-Ep Gap | Official Test Top-1 | Official Test Avg/Class |\n"
                    "|---|---|---:|:---:|---:|---:|---:|---:|---:|---:|---:|\n"
                    "| **E1 (`baseline_001`)** | 4-Block CustomCNN from scratch, no augmentation, no dropout, Adam `lr=1e-3` | 439,816 | 31 | 80.18% | 21.69% | 21.29% | 42.75% | +58.49 pp | 21.01% | 20.63% |\n"
                    "| **E2 (`regularized`)** | CustomCNN + BatchNorm + **Dropout(0.4)** + Augmentation + AdamW + Plateau LR | 440,776 | 21 | 16.61% | 10.43% | 9.63% | 27.11% | +6.19 pp | 8.97% | 8.80% |\n"
                    "| **E4 (`no_dropout`)** | Controlled Ablation: identical to E2 with **Dropout(0.0)** | 440,776 | 38 | 30.47% | 14.39% | 13.93% | 34.83% | +16.08 pp | *Not run* | *Not run* |\n"
                    "| **E3 (`resnet18`) [Selected]** | Pretrained ImageNet-1K ResNet18 (`Linear(512, 200)`): 5-ep head warmup + 30-ep `layer4+fc` fine-tune | 11,279,112 | 34 | 99.90% | **69.66%** | **69.10%** | **89.57%** | +30.24 pp | **70.43%** | **70.12%** |\n\n"
                    "> **Official Held-Out Test Benchmark (E3 ResNet18 on 1,204 Test Images):** "
                    "**Top-1 Accuracy = `70.43%`** (`848 / 1,204`) &nbsp;|&nbsp; "
                    "**Average Accuracy per Class = `70.12%`** &nbsp;|&nbsp; "
                    "**Top-5 Accuracy = `89.12%`** (`1,073 / 1,204`) &nbsp;|&nbsp; "
                    "**Macro-F1 = `69.90%`**."
                )

                with gr.Row():
                    if learning_curves_path:
                        gr.Image(
                            value=learning_curves_path,
                            label="Actual Training & Validation Loss, Top-1 Accuracy, and Generalisation Gap Curves (E1, E2, E4, E3)",
                            interactive=False,
                        )
                    if ablation_curves_path:
                        gr.Image(
                            value=ablation_curves_path,
                            label="Controlled Dropout Ablation Curves: E2 (Dropout p=0.4) vs E4 (Dropout p=0.0)",
                            interactive=False,
                        )
                if test_summary_path:
                    gr.Image(
                        value=test_summary_path,
                        label="Official Held-Out Test Evaluation Summary & Validation-to-Test Parity",
                        interactive=False,
                    )

                gr.Markdown(
                    "### Key Scientific Takeaways\n"
                    "1. **High-Variance Memorisation in E1 (Overfitting):** Training a 4-block CNN (`439,816` parameters) from random initialization without augmentation on ~19.35 images/species allowed the network to memorise training textures (`80.18%` clean-train Top-1 at epoch 31, `92.64%` at epoch 40) while validation accuracy plateaued at `21.69%` (a `58.49 pp` best-epoch generalisation gap and `71.99 pp` final-epoch gap).\n"
                    "2. **Capacity Bottleneck & Compound Regularisation in E2 (Underfitting):** Adding `BatchNorm2d`, `Dropout2d(0.10)`, `Dropout(0.40)`, weight decay (`1e-4`), and spatial/photometric augmentation shrank the generalisation gap tenfold to `6.19 pp`, but constrained effective representational capacity so severely that clean-train Top-1 reached only `16.61%` and validation Top-1 dropped to `10.43%`.\n"
                    "3. **Causal Isolation via Controlled Dropout Ablation (E2 vs E4):** Holding architecture, seed, split, augmentation, and optimizer identical while setting `dropout = 0.0` (E4) raised clean-train Top-1 from `16.61%` to `30.47%` and validation Top-1 from `10.43%` to `14.39%` (`+3.96 pp`, `+4.30 pp` Avg/Class). This proves `Dropout(0.4)` was a primary contributor to E2's underfitting, while even without dropout a shallow 4-block CNN trained from scratch lacks mid-level part detectors for 200 fine-grained bird species.\n"
                    "4. **Why Staged Transfer Learning (E3 ResNet18) Succeeds:** Initialising with ImageNet-1K pretrained filters provides reusable edge, texture, and feather-pattern detectors. Warming up the replacement `Linear(512, 200)` head for 5 epochs (`lr=1e-3`, reaching `53.28%` val Top-1) before selectively unfreezing `layer4 + fc` (`lr=1e-4`, with frozen backbone BatchNorm running statistics) lifts validation Top-1 to **`69.66%`** and official held-out test Top-1 to **`70.43%`** (`70.12%` Average Accuracy per Class). Notice that E3 still exhibits a `30.24 pp` train-to-validation gap (`99.90%` clean-train vs `69.66%` val), whereas the close validation-to-test agreement (`69.66%` val vs `70.43%` test) confirms that our duplicate-grouped stratified split is an unbiased estimator of held-out generalisation."
                )

        clear.click(
            reset_ui,
            inputs=[],
            outputs=[photo, prediction, note, crop_out, heatmap_out, overlay_out, cam_panel_out],
        )
        submit.click(
            classify,
            inputs=[photo, show_cam],
            outputs=[prediction, note, crop_out, heatmap_out, overlay_out, cam_panel_out],
            concurrency_limit=1,
        )
    demo.queue(default_concurrency_limit=1, max_size=8)
    return demo


if __name__ == "__main__":
    default_host = "0.0.0.0" if (os.environ.get("PORT") or os.environ.get("RENDER")) else "127.0.0.1"
    build_app().launch(
        server_name=os.environ.get("GRADIO_SERVER_NAME", default_host),
        server_port=int(os.environ.get("PORT", "7860")),
    )
