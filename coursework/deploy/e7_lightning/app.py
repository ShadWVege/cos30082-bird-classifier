"""Standalone Gradio application for E7 ConvNeXt-Tiny fine-grained bird classification.

Designed for lightweight CPU hosting on Lightning AI Free Studio, local CPU, or Docker.
"""
from functools import lru_cache
import json
import os
from pathlib import Path
import sys

# Constrain glibc heap arenas and OpenMP thread pools before importing PyTorch
# to keep memory usage lightweight on shared cloud host containers.
os.environ.setdefault("MALLOC_ARENA_MAX", "2")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import gradio as gr
from src.inference import Predictor

# Optional Hugging Face ZeroGPU decorator support; falls back to no-op on CPU/local.
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
    "predicted species name, zero-based class code (`000–199`), and 200-class softmax scores."
)


@lru_cache(maxsize=1)
def predictor():
    """Lazily load and cache the ConvNeXt-Tiny E7 inference engine once per process."""
    if MODEL_PATH.is_file():
        return Predictor(MODEL_PATH)
    fallback = ROOT_DIR.parent.parent / "outputs" / "experiment_001" / "E7_convnext_tiny" / "model_inference.pt"
    if fallback.is_file():
        return Predictor(fallback)
    return Predictor(MODEL_PATH)


def _model_ready():
    return MODEL_PATH.is_file() or (
        ROOT_DIR.parent.parent / "outputs" / "experiment_001" / "E7_convnext_tiny" / "model_inference.pt"
    ).is_file()


def _resolve_asset(filename):
    """Locate static chart assets in either standalone folder (`assets/`) or repository paths."""
    candidates = [
        ROOT_DIR / "assets" / filename,
        ROOT_DIR.parent.parent / "outputs" / "experiment_001" / "figures" / filename,
        ROOT_DIR.parent.parent / "outputs" / "final_test_results" / "figures" / filename,
    ]
    for path in candidates:
        if path.is_file():
            return str(path)
    return None


def _discover_examples():
    """Load packaged example images if present, or dynamically locate validation images."""
    examples_dir = ROOT_DIR / "examples"
    if examples_dir.is_dir():
        found = sorted(
            str(p)
            for p in examples_dir.iterdir()
            if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp")
        )
        if found:
            return found[:4]
    local_cfg = ROOT_DIR.parent.parent / "configs" / "data.local.json"
    split_cfg = ROOT_DIR.parent.parent / "data" / "prepared" / "split.json"
    if local_cfg.is_file() and split_cfg.is_file():
        try:
            train_root = Path(json.loads(local_cfg.read_text(encoding="utf-8")).get("train_root", ""))
            val_records = json.loads(split_cfg.read_text(encoding="utf-8")).get("val", [])
            preferred = [
                "Red_headed_Woodpecker_0012_2578571305.jpg",
                "Scarlet_Tanager_0022_2227619316.jpg",
                "American_Goldfinch_0013_146872871.jpg",
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
def classify(image, show_gradcam=True, use_tta=False):
    """Run deterministic CUB-200 inference and optional ConvNeXt-Tiny features.7 Grad-CAM attribution.

    Returns a 6-tuple whose first element ([0]) is the top-k {species_name: probability} dict.
    """
    if image is None:
        raise gr.Error("Please upload a bird photograph first (supported formats: JPEG, PNG, WebP).")
    try:
        result = predictor().predict_with_gradcam(
            image, include_gradcam=bool(show_gradcam), use_tta=bool(use_tta)
        )
    except (FileNotFoundError, ValueError, OSError, TypeError) as error:
        raise gr.Error(str(error)) from error

    scores = result["scores"]
    records = result["records"]
    top1 = records[0]
    best = top1["prob"]
    caution = (
        "Low top-1 margin (<50% softmax score) — multiple visually similar species share probability mass; "
        "inspect the Top-5 candidates and diagnostic plumage markings carefully."
        if best < 0.5
        else "High top-1 concentration — compare key diagnostic plumage landmarks against the Top-5 candidates below."
    )
    mode_desc = (
        "**2-View TTA** (Original + Horizontally Flipped bilateral reflection averaging; 80.32% test accuracy)"
        if use_tta
        else "**Standard Single-View** (224×224 deterministic center-crop; official 80.73% test accuracy)"
    )
    if show_gradcam:
        if use_tta:
            gradcam_status = (
                f"Computed ConvNeXt-Tiny `features.7` (7×7 spatial grid, 768 channels) Grad-CAM on **original image orientation** "
                f"for the TTA-selected target **`[{top1['label']:03d}] {top1['name']}`**."
            )
        else:
            gradcam_status = (
                f"Computed ConvNeXt-Tiny `features.7` (7×7 spatial grid, 768 channels) Grad-CAM for predicted target "
                f"**`[{top1['label']:03d}] {top1['name']}`** via terminal feature-map gradient isolation."
            )
    else:
        gradcam_status = (
            "Grad-CAM computation was skipped (`Compute features.7 Grad-CAM Explainability` unchecked) — "
            "ran fast forward-only `@torch.inference_mode()` pass."
        )

    table_lines = [
        f"### Predicted Species: **{top1['name']}** &nbsp;|&nbsp; Zero-Based Class ID: **`{top1['label']:03d}`** (`{top1['label']}` in `0–199`) &nbsp;|&nbsp; Top-1 Softmax Score: **`{best * 100:.2f}%`**",
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
            f"- **Inference Mode:** {mode_desc}",
            f"- **Prediction Guidance:** {caution}",
            f"- **Explainability Status:** {gradcam_status}",
            "- **Prediction Confidence vs Dataset Accuracy:** Softmax scores reflect relative output activation across the 200 trained classes and **are not calibrated posterior probabilities**. An image assigned 95% softmax confidence is not guaranteed to be 95% accurate. Measured generalization accuracy is **80.73% Top-1** on the 1,204-image held-out test set.",
            "- **Closed-Set Benchmark Scope:** This model recognizes **only the 200 North American bird species (`000–199`)** from the CUB-200 benchmark. Non-bird images or species outside CUB-200 will be forced into the closest learned class.",
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
    """Construct the three-section responsive Gradio interface for E7 ConvNeXt-Tiny."""
    model_comp_path = _resolve_asset("model_comparison.png")
    e3_vs_e7_path = _resolve_asset("ablation_e3_vs_e7.png")
    gradcam_focused_path = _resolve_asset("gradcam_focused.png")
    test_summary_path = _resolve_asset("test_metrics_summary.png")

    with gr.Blocks(title="Bird Species Explorer — ConvNeXt-Tiny (E7) — COS30082") as demo:
        gr.Markdown(
            "# Bird Species Explorer — ConvNeXt-Tiny (E7) — COS30082\n"
            "Fine-grained visual classification across **200 bird species (`000–199`)** using our "
            "exploratory **Fine-Tuned ConvNeXt-Tiny (E7)** model (**27,973,928** parameters; "
            "**80.40% Val Top-1** &nbsp;|&nbsp; **80.73% Official Held-Out Test Top-1** &nbsp;|&nbsp; **80.24% Test Avg/Class** &nbsp;|&nbsp; **95.18% Test Top-5**).\n\n"
            "> **Research & Deployment Status:** E7 demonstrates an empirical **+10.30 percentage-point improvement in official test Top-1 accuracy** over the pre-committed E3 ResNet18 baseline "
            "(70.43% → 80.73%, paired McNemar test: p = 2.48 × 10⁻¹⁵, χ² = 60.04, N = 1,204). "
            "Single-view deterministic center-crop (80.73%) is the primary mode as it outperformed 2-view TTA (80.32%) on the held-out test set."
        )
        if not _model_ready():
            gr.Markdown(
                "⚠️ **Setup pending:** `model.pt` must be present in this directory before predictions can run."
            )

        with gr.Tabs():
            # =============================================================
            # TAB 1 — Bird Identification & Grad-CAM Explainability
            # =============================================================
            with gr.TabItem("1. Bird Identification & ConvNeXt-Tiny Grad-CAM"):
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
                            label="Compute features.7 Grad-CAM Explainability (uncheck for faster forward-only inference)",
                        )
                        use_tta = gr.Checkbox(
                            value=False,
                            label="Optional 2-View TTA (Default OFF: Single-view 80.73% outperformed TTA 80.32% on official test)",
                        )
                        with gr.Row():
                            submit = gr.Button(
                                "Identify Bird Species",
                                variant="primary",
                                interactive=_model_ready(),
                            )
                            clear = gr.Button(value="Clear / Reset")
                        gr.Markdown(
                            "**Closed-Set Scope:** Recognises **only the 200 trained CUB-200 bird species (`000–199`)**. "
                            "Uploading non-bird images or species outside CUB-200 will still yield a 200-way softmax distribution."
                        )
                        example_paths = _discover_examples()
                        if example_paths:
                            gr.Examples(
                                examples=[[p] for p in example_paths],
                                inputs=[photo],
                                label="Sample Bird Photographs (Click to test)",
                            )

                    with gr.Column(scale=6, min_width=320):
                        prediction = gr.Label(
                            num_top_classes=5,
                            label="Top-5 Predicted Species (200-Class Softmax Score Distribution)",
                        )
                        note = gr.Markdown(DEFAULT_NOTE)

                # Grad-CAM Display Section
                gr.Markdown(
                    "---\n"
                    "## Section 2 — ConvNeXt-Tiny Grad-CAM Attribution (`features.7`)\n"
                    "Attribution is computed on the terminal convolutional stage (`features.7`, 7×7 spatial resolution, 768 channels) "
                    "using **terminal feature-map gradient isolation** (gradients backpropagated only to the cached 7×7 feature map, keeping all 27.97M backbone parameters frozen in zero-grad mode)."
                )
                with gr.Row():
                    crop_out = gr.Image(
                        type="pil",
                        label="1. Deterministic Evaluation Crop (224x224 RGB)",
                    )
                    heatmap_out = gr.Image(
                        type="pil",
                        label="2. ConvNeXt-Tiny features.7 Grad-CAM Heatmap (7x7 -> 224x224)",
                    )
                    overlay_out = gr.Image(
                        type="pil",
                        label="3. Grad-CAM Heatmap Overlay on Evaluation Crop",
                    )
                with gr.Accordion("Combined 3-Panel Grad-CAM Strip & Interpretability Limitations", open=False):
                    cam_panel_out = gr.Image(
                        type="pil",
                        label="Combined 3-Panel Strip (Crop | features.7 Heatmap | Overlay)",
                    )
                    gr.Markdown(
                        "### What Grad-CAM Does and Does Not Prove\n"
                        "- **What it shows:** Grad-CAM weights the 768 channels of `features.7` (7×7 spatial grid) by the global-average-pooled gradients of the top-1 predicted class logit, highlighting image regions that positively contributed to that score.\n"
                        "- **ConvNeXt vs ResNet18 Receptive Field:** ConvNeXt utilizes 7×7 depthwise separable convolutions and inverted bottleneck channel expansions, capturing contextual plumage relationships with localized focus on key diagnostic field marks (crown, eyeline, bill, wing bars).\n"
                        "- **What it does NOT prove:** Grad-CAM is a **post-hoc spatial correlation map, not causal proof of biological reasoning**. Coarse 7×7 spatial stride means activations naturally spill onto adjacent perches, foliage, or water, and cannot verify fine morphological features in isolation."
                    )

            # =============================================================
            # TAB 2 — Model Performance & Full Comparative Benchmark
            # =============================================================
            with gr.TabItem("2. Model Performance, Ablations & Full Benchmark"):
                gr.Markdown(
                    "## Section 3 — Comprehensive Model Benchmark & Ablations\n"
                    "All models were trained and selected strictly on the **frozen grouped stratified development split** "
                    "(`3,870` training / `959` validation images across `200` classes, seed `42`). "
                    "Official held-out test evaluation was performed on the `1,204` official test images.\n\n"
                    "### Complete 7-Experiment Benchmark Table\n\n"
                    "<style>\n"
                    ".benchmark-table-wrapper {\n"
                    "    overflow-x: auto !important;\n"
                    "    -webkit-overflow-scrolling: touch;\n"
                    "    width: 100%;\n"
                    "    margin: 10px 0;\n"
                    "}\n"
                    ".benchmark-table-wrapper table {\n"
                    "    width: 100%;\n"
                    "    min-width: 980px;\n"
                    "    border-collapse: collapse;\n"
                    "}\n"
                    ".benchmark-table-wrapper th, .benchmark-table-wrapper td {\n"
                    "    padding: 6px 10px;\n"
                    "    white-space: nowrap;\n"
                    "}\n"
                    ".benchmark-table-wrapper td:nth-child(2) {\n"
                    "    white-space: normal;\n"
                    "    min-width: 260px;\n"
                    "}\n"
                    "</style>\n\n"
                    "| <span style=\"white-space:nowrap\">Exp</span> | Model Architecture & Key Mechanism | <span style=\"white-space:nowrap\">Params</span> | <span style=\"white-space:nowrap\">Best Ep</span> | <span style=\"white-space:nowrap\">Val Top-1</span> | <span style=\"white-space:nowrap\">Val Avg/Class</span> | <span style=\"white-space:nowrap\">Val Top-5</span> | <span style=\"white-space:nowrap\">Test Top-1</span> | <span style=\"white-space:nowrap\">Test Avg/Class</span> | <span style=\"white-space:nowrap\">Test Top-5</span> | <span style=\"white-space:nowrap\">Test Loss</span> |\n"
                    "|:---:|---|---:|:---:|---:|---:|---:|:---:|:---:|:---:|:---:|\n"
                    "| **E1** | Custom CNN (4 Conv-ReLU-Pool blocks, Adam `1e-3`) | 439,816 | 31 | 21.69% | 21.29% | 42.75% | 21.01% | 20.63% | 42.03% | 6.7320 |\n"
                    "| **E2** | Regularized CNN (+BatchNorm, +Dropout 0.4, Aug, AdamW) | 440,296 | 21 | 10.43% | 9.63% | 27.11% | 8.97% | 8.80% | 25.33% | 4.4129 |\n"
                    "| **E4** | Dropout Ablation (Identical to E2 with Dropout 0.0) | 440,296 | 38 | 14.39% | 13.93% | 34.83% | *Not evaluated* | *Not evaluated* | *Not evaluated* | — |\n"
                    "| **E3** | Pretrained ResNet18 (5-ep head + 30-ep `layer4+fc` fine-tune) [Pre-Committed Winner] | 11,279,112 | 34 | 69.66% | 69.10% | 89.57% | 70.43% | 70.12% | 89.12% | 1.2004 |\n"
                    "| <span style=\"white-space:nowrap\">E3+TTA</span> | ResNet18 with 2-View Bilateral Horizontal-Flip TTA | 11,279,112 | 34 | 70.80% | 70.27% | 89.99% | *Not evaluated* | *Not evaluated* | *Not evaluated* | — |\n"
                    "| **E5** | Label Smoothing Ablation (ResNet18 + `label_smoothing=0.05`) | 11,279,112 | 27 | 68.72% | 68.02% | 88.11% | *Not evaluated* | *Not evaluated* | *Not evaluated* | — |\n"
                    "| **E6** | Enhanced ResNet18 (`layer3+layer4+fc` fine-tuning) | 11,279,112 | 24 | 68.40% | 67.88% | 88.95% | *Not evaluated* | *Not evaluated* | *Not evaluated* | — |\n"
                    "| **E7** | **Fine-Tuned ConvNeXt-Tiny (5-ep head + 25-ep `features.7+head`) [Current Model]** | <span style=\"white-space:nowrap\">**27,973,928**</span> | **27** | **80.40%** | **79.75%** | **96.35%** | **80.73%** | **80.24%** | **95.18%** | **0.7595** |\n"
                    "| <span style=\"white-space:nowrap\">E7+TTA</span> | ConvNeXt-Tiny with 2-View Bilateral Horizontal-Flip TTA | <span style=\"white-space:nowrap\">27,973,928</span> | 27 | 81.02% | 80.35% | 96.25% | 80.32% | 79.86% | 94.93% | 0.7489 |\n\n"
                    "> **Footnote on Evaluation Integrity:**\n"
                    "> Official test results are available for E1, E2, E3, E7 and E7+TTA. E3+TTA, E4, E5 and E6 have development-validation results only. E7 was developed following inspection of earlier E3 test results and is therefore disclosed as an exploratory post-test milestone. Missing official test metrics are marked 'Not evaluated'; none have been estimated.\n\n"
                    "### Rigorous Statistical Comparison (E3 vs E7 on Held-Out Test Set)\n"
                    "- **Paired Sample Breakdown (1,204 Test Images):** Both models agree on 952 images (784 both correct, 168 both incorrect). On the 252 discordant test images, E7 is correct on **188 images** that E3 failed on, whereas E3 is correct on only **64 images** that E7 missed (net gain: +124 correctly classified test images).\n"
                    "- **Exact Paired McNemar Test:** Two-sided exact binomial test on discordant pairs (188 vs 64) yields **Exact paired McNemar test: p = 2.48 × 10⁻¹⁵** (Edwards continuity-corrected χ² = 60.04, df = 1, p < 0.0001; exact two-sided p ≈ 2.4849 × 10⁻¹⁵), confirming strong statistical evidence of a performance difference on these paired test outcomes (+10.30 percentage-point improvement in official test Top-1 accuracy, from 70.43% to 80.73%).\n"
                    "- **Methodological Disclosure:** E3 was pre-committed as the production deployment winner prior to held-out test evaluation. E7 was developed as a post-test exploratory milestone after inspecting E3 diagnostics. While cross-split agreement is strong (E3: 69.66% val vs 70.43% test; E7: 80.40% val vs 80.73% test), this parity does not prove the absence of data leakage or guarantee future-domain generalisation: finite test-set uncertainty (N = 1,204, 4–8 samples/species) and four known cross-split pixel-duplicate pairs, corresponding to approximately 0.33% of the 1,204-image official test set, are documented as a benchmark integrity limitation."
                )

                with gr.Row():
                    if test_summary_path:
                        gr.Image(
                            value=test_summary_path,
                            label="Authoritative Official Held-Out Test Benchmark, Paired McNemar Matrix, and Generalisation Parity",
                            interactive=False,
                        )
                    if model_comp_path:
                        gr.Image(
                            value=model_comp_path,
                            label="All 7 Experiments: Clean-Train vs Validation Top-1, Avg/Class, and Generalisation Gap Progression",
                            interactive=False,
                        )
                with gr.Row():
                    if e3_vs_e7_path:
                        gr.Image(
                            value=e3_vs_e7_path,
                            label="Head-to-Head Comparison: E3 ResNet18 vs E7 ConvNeXt-Tiny Learning Curves & Gap Evolution",
                            interactive=False,
                        )
                    if gradcam_focused_path:
                        gr.Image(
                            value=gradcam_focused_path,
                            label="Diagnostic Landmark Focus: E3 layer4 vs E7 features.7 Grad-CAM Case Study",
                            interactive=False,
                        )

            # =============================================================
            # TAB 3 — Architecture & Cloud Engineering Analysis
            # =============================================================
            with gr.TabItem("3. Architecture & Cloud Engineering Analysis"):
                gr.Markdown(
                    "## Section 4 — Architecture & Cloud Engineering Analysis\n\n"
                    "### 1. Modern ConvNeXt Architecture Innovations\n"
                    "ConvNeXt ('A ConvNet for the 2020s', Liu et al., CVPR 2022) modernises standard convolutional backbones with design elements inspired by Vision Transformers (ViT):\n"
                    "- **7×7 Depthwise Separable Convolutions:** Replaces standard 3×3 convolutions with larger 7×7 depthwise filters, dramatically expanding spatial receptive fields without quadratic parameter growth.\n"
                    "- **Inverted Bottleneck Design:** Expands hidden channels by 4× inside each block before projecting back (similar to Transformer MLP blocks and MobileNetV2), enhancing representational expressiveness.\n"
                    "- **LayerNorm over BatchNorm:** Replaces BatchNorm with per-sample LayerNorm, eliminating batch-statistic dependency and running mean/variance drift on small batches.\n"
                    "- **GELU Non-linearities & Fewer Activations:** Uses Gaussian Error Linear Units with fewer activation layers per block, smoothing optimization loss surfaces.\n\n"
                    "### 2. Fine-Grained Transfer Learning & Generalisation Dynamics\n"
                    "- **Why Pretrained Representations Are Essential:** Training from scratch on small fine-grained datasets (~19 images/species across 200 classes) leads to memorization of background shortcuts (E1 baseline: 58.49 pp generalization gap). Pretraining on ImageNet-1K provides generic edge, texture, and contour detectors that transfer directly to avian plumage.\n"
                    "- **Why Two-Stage Warmup Precedes Fine-Tuning:** Initializing a randomly weighted 200-class classifier head with large gradients into an unfrozen backbone causes catastrophic representational destruction. Stage 1 (5 epochs) trains only the classifier head to align logits with frozen feature embeddings; Stage 2 fine-tunes terminal blocks (`layer4` in ResNet18, `features.7` in ConvNeXt) at a lower learning rate.\n"
                    "- **Why a Generalisation Gap Persists in E7:** Despite reaching 80.40% validation Top-1 and 80.73% test Top-1, E7 reaches 99.82% clean-train Top-1, retaining a 19.42 pp generalisation gap (substantially narrower than E3's 30.24 pp gap). On ~19 images per class, high-capacity models naturally fit unique training sample artifacts, but ConvNeXt's 7×7 depthwise design and LayerNorm regularisation generalize far more robustly.\n\n"
                    "### 3. Four Memory Optimization Techniques for Constrained Cloud Containers\n"
                    "To enable high-accuracy ConvNeXt-Tiny inference on lightweight cloud instances:\n"
                    "1. **CPU-Only Wheels:** We specify `--extra-index-url https://download.pytorch.org/whl/cpu` to install the 181 MB CPU wheel rather than the 2.5 GB CUDA distribution.\n"
                    "2. **Memory-Mapped Weight Loading (`mmap=True`):** The 106.8 MiB inference weights are mapped via kernel page cache (`RssFile`), keeping anonymous heap memory (`RssAnon`) small.\n"
                    "3. **Zero-Allocation Meta-Device Instantiation (`assign=True`):** Model skeleton is instantiated on `torch.device('meta')` before weights are assigned in-place, eliminating temporary weight cloning spikes during startup.\n"
                    "4. **Terminal Feature-Map Gradient Isolation:** During Grad-CAM, all 27.97M parameters remain frozen with zero gradient memory. Gradients are tracked exclusively on the detached 7×7 terminal feature map, reducing Grad-CAM memory overhead by >90%.\n\n"
                    "### 4. Production Hosting Platforms: Realistic Operational Trade-Offs\n"
                    "| Dimension | Render Free Web Service (Live Backup App) | Lightning AI Free CPU Studio (E7 Exploratory App) |\n"
                    "|---|---|---|\n"
                    "| **Hardware Tier** | 0.1 vCPU, 512 MiB hard cgroup RAM limit | 4 vCPUs, ~16 GB system RAM, 50 GB persistent SSD |\n"
                    "| **Model Deployed** | E3 ResNet18 (11.28M parameters; 70.43% official test Top-1, single-view). Optional TTA available; official TTA test performance not evaluated. | E7 ConvNeXt-Tiny (27.97M parameters; 80.73% official test Top-1, single-view). Optional TTA available (80.32% test Top-1). |\n"
                    "| **Operating Memory** | ~289 MiB RSS (anonymous heap ~210 MiB, file cache ~79 MiB) | ~430 MiB baseline RSS; ~796 MiB during sequential requests |\n"
                    "| **Container Memory Limit** | Hard 512 MiB cgroup; E7 reached 500.57 MiB peak (near limit) | Multi-GB host capacity provides ample memory headroom |\n"
                    "| **Inactivity & Sleep** | Spins down after 15 min inactivity; wakes on HTTP request (~50s cold start) | Auto-sleeps after 10 min inactivity; public port wake under evaluation |\n"
                    "| **Runtime Boundaries** | Runs continuously under free plan terms; no 4-hour restart rule | **4-Hour Runtime Boundary:** Active Studios require manual restart after 4 hours to avoid paid billing state |\n"
                    "| **Billing & Credits** | Free ($0) on free tier with zero credit card or billing transitions | Promotional credits exist upon signup; active compute advertises 0.44 credits/hr |\n"
                    "| **Role in Coursework** | **Persistent Public Evaluator Link (subject to free-tier spin-up)** | **High-Accuracy Demonstration & Student Video Presentation** |\n"
                )

        clear.click(
            reset_ui,
            inputs=[],
            outputs=[photo, prediction, note, crop_out, heatmap_out, overlay_out, cam_panel_out],
        )
        submit.click(
            classify,
            inputs=[photo, show_cam, use_tta],
            outputs=[prediction, note, crop_out, heatmap_out, overlay_out, cam_panel_out],
            concurrency_limit=1,
        )
    demo.queue(default_concurrency_limit=1, max_size=8)
    return demo


if __name__ == "__main__":
    default_host = os.environ.get(
        "GRADIO_SERVER_NAME",
        "0.0.0.0" if (os.environ.get("PORT") or os.environ.get("LIGHTNING_APP_PORT") or sys.platform.startswith("linux")) else "127.0.0.1"
    )
    port = int(os.environ.get("LIGHTNING_APP_PORT", os.environ.get("PORT", "7863")))
    print(f"Starting E7 ConvNeXt-Tiny Gradio App on {default_host}:{port}...")
    build_app().launch(
        server_name=default_host,
        server_port=port,
    )
