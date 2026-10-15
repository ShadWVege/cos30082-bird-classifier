# COS30082 Bird Species Classification

A complete, generously commented Python/PyTorch workflow for the 200-class Caltech-UCSD Birds 200 (CUB-200) coursework assignment: data integrity audit → frozen grouped stratified validation split → model progression (scratch baseline E1, regularized E2, ablation E4, ResNet-18 transfer learning E3, bilateral TTA, label smoothing E5, enhanced fine-tuning E6, and modern ConvNeXt-Tiny milestone E7) → pre-committed model selection → official held-out test evaluation → Gradio web deployment.

Read [PLAN.md](PLAN.md) for the project blueprint and [STATUS.md](STATUS.md) for current verified metrics.

**For reviewers and presentation:** [CODE_WALKTHROUGH.md](docs/CODE_WALKTHROUGH.md) maps the important algorithms to their source files; [VIDEO_CODE_GUIDE.md](docs/VIDEO_CODE_GUIDE.md) gives a concise 2–3-minute code-demonstration route. [DEPLOYMENT.md](docs/DEPLOYMENT.md) explains the real Hugging Face → Lightning E7 and Render E3 arrangement. [GITHUB_PUBLISHING.md](docs/GITHUB_PUBLISHING.md) explains how to publish full coursework without breaking the E3 Render branch.

---

## Dataset Configuration

The assignment uses the official CUB-200 dataset with 200 bird species (4,829 training images in `Train.zip` and 1,204 held-out test images in `Test.zip`, annotated in `data/train.txt` and `data/test.txt` with zero-indexed labels).

To connect your local dataset images:
1. Copy `configs/data.example.json` to `configs/data.local.json`:
   ```powershell
   Copy-Item configs/data.example.json configs/data.local.json
   ```
2. Edit `configs/data.local.json` to specify your absolute or relative paths to the extracted `Train` and `Test` image folders:
   ```json
   {
     "train_root": "path/to/Train",
     "test_root": "path/to/Test",
     "train_manifest": "data/train.txt",
     "test_manifest": "data/test.txt",
     "prepared_dir": "data/prepared",
     "num_classes": 200,
     "expected_train_count": 4829,
     "seed": 42,
     "val_fraction": 0.2,
     "conflict_policy": "train_only"
   }
   ```

*Data Integrity Note:* An exhaustive RGB pixel hash audit identified three conflicting duplicate pairs in training. As documented in [docs/DATA_DECISIONS.md](docs/DATA_DECISIONS.md), these images are pinned to training (`conflict_policy: train_only`) to eliminate validation leakage. The frozen development split contains **3,870 training + 959 validation images**, with all 200 species represented. An audit of the 1,204 test images identified four cross-split duplicates (including multi-species feeder photo `Test/0197.jpg`), which are preserved without data tampering to protect benchmark integrity.

---

## Environment Setup (Windows / Linux)

The project requires Python 3.10–3.12 (developed and tested on Python 3.12.14).

### Automated Setup (PowerShell)
```powershell
# For CUDA 12.8 GPU acceleration:
.\scripts\setup.ps1 -Compute cu128

# For CPU-only development:
.\scripts\setup.ps1 -Compute cpu

# Verify environment and CUDA status:
.\.venv\Scripts\python.exe -m src.doctor
```

### Manual Installation
```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
```

---

## Empirical Benchmark Summary

All seven experimental pipelines were trained and evaluated on the frozen development split (seed 42). Candidate models (E1, E2, E3) were evaluated once on the official held-out test set; E7 was evaluated post-test on frozen weights with disclosed history:

| Exp | Architecture & Configuration | Params | Best Ep | Val Top-1 | Val Avg/Class | Test Top-1 | Test Avg/Class |
|:---:|---|---:|:---:|---:|---:|---:|---:|
| **E1** | Baseline CustomCNN (4 Conv stages, scratch, Adam `1e-3`) | 439,816 | 31 | 21.69% | 21.29% | 21.01% | 20.63% |
| **E2** | Regularized CustomCNN (+BatchNorm, +Dropout 0.4, AdamW) | 440,296 | 21 | 10.43% | 9.63% | 8.97% | 8.80% |
| **E4** | Controlled Dropout Ablation (Identical to E2, Dropout 0.0) | 440,296 | 38 | 14.39% | 13.93% | *Not evaluated* | *Not evaluated* |
| **E3** | Pretrained ResNet18 (5-ep head + 30-ep `layer4+fc`) [Selected] | 11,279,112 | 34 | **69.66%** | **69.10%** | **70.43%** | **70.12%** |
| E3+TTA | ResNet18 + Two-View Horizontal-Flip TTA (Inference only) | 11,279,112 | 34 | **70.80%** | **70.27%** | *Not evaluated* | *Not evaluated* |
| **E5** | ResNet18 + Label Smoothing ($\alpha=0.05$) | 11,279,112 | 27 | 68.72% | 68.02% | *Not evaluated* | *Not evaluated* |
| **E6** | Enhanced ResNet18 (`layer3+layer4+fc` fine-tuning) | 11,279,112 | 24 | 68.40% | 67.88% | *Not evaluated* | *Not evaluated* |
| **E7** | Modern ConvNeXt-Tiny Milestone (5-ep head + 25-ep Stage 4) | 27,973,928 | 27 | **80.40%** | **79.75%** | **80.73%** | **80.24%** |
| E7+TTA | ConvNeXt-Tiny + Two-View Horizontal-Flip TTA | 27,973,928 | 27 | **81.02%** | **80.35%** | **80.32%** | **79.86%** |

*Key Findings:*
1. **Capacity Starvation in E2:** Dropout 0.4 in a compact 256-channel network reduced validation accuracy from 21.69% to 10.43%. Removing Dropout (E4) restored accuracy to 14.39% (+4.30 pp).
2. **Transfer Learning Breakthrough in E3:** Pretrained ImageNet representations unlocked a massive +47.97 pp gain over baseline.
3. **Statistical Superiority of E7:** On the official held-out test set, E7 reached 80.73% Top-1 (+10.30 pp over E3; paired McNemar test $p = 2.48 \times 10^{-15}$, $\chi^2 = 60.04$).
4. **TTA Dynamics:** Bilateral reflection TTA improved ResNet18 (+1.14 pp on validation), but slightly degraded ConvNeXt-Tiny on test data (-0.41 pp), making single-view inference the primary mode.

---

## Working Commands & Workflow

All commands execute from the repository root using `.venv\Scripts\python.exe` (Windows) or `python3` (Linux/macOS):

### 1. Run Unit & Integration Test Suite
```powershell
.\.venv\Scripts\python.exe -m pytest
```

### 2. Deterministic Model Training & Reproduction
To retrain individual models on the frozen split (`data/prepared/split.json`):
```powershell
# E1 Baseline CustomCNN (439,816 parameters)
.\.venv\Scripts\python.exe -m src.train --config configs/baseline.json --out outputs/experiment_001/baseline

# E2 Regularized CustomCNN (440,296 parameters)
.\.venv\Scripts\python.exe -m src.train --config configs/regularized.json --out outputs/experiment_001/regularized

# E4 Dropout Ablation (440,296 parameters)
.\.venv\Scripts\python.exe -m src.train --config configs/regularized_no_dropout.json --out outputs/experiment_001/E4_regularized_no_dropout

# E3 Pretrained ResNet18 (11,279,112 parameters)
.\.venv\Scripts\python.exe -m src.train --config configs/resnet18.json --out outputs/experiment_001/resnet18

# E7 Pretrained ConvNeXt-Tiny (27,973,928 parameters)
.\.venv\Scripts\python.exe -m src.train --config configs/convnext_tiny.json --out outputs/experiment_001/E7_convnext_tiny
```

### 3. Standalone Gradio Web Application
Launch the local interactive Gradio interface (supports image classification, Top-5 probabilities, optional 2-view TTA toggle, and real-time Grad-CAM explainability):
```powershell
.\.venv\Scripts\python.exe app.py
```

### 4. Compile the Four-Page Report (Optional)
The compiled coursework PDF is submitted separately on Canvas. To compile `reports/report.pdf` from its Typst source (`reports/report.typ`):
```powershell
.\.venv\Scripts\python.exe scripts/compile_report.py
```
*(Requires `typst` binary or Python package and `pypdf`, declared in `requirements-dev.txt`)*.

---

## Coursework Deliverables & Confirmed Cloud Hosting

| Deliverable | URL / Target | Description |
|---|---|---|
| **Primary Project Portal** | [Hugging Face Space](https://huggingface.co/spaces/ShadWVege/cos30082-bird-species-explorer) | Static HTML landing page, architectural diagrams, benchmark summary, and navigation. *(Note: Hugging Face hosts the documentation only; CNN model inference does not run directly on HF).* |
| **Interactive Classifier (E7)** | [Lightning AI Free CPU Studio](https://7863-01m4vaa4swe6b7ha57xrh4ch7s.cloudspaces.litng.ai/) | Live high-accuracy E7 ConvNeXt-Tiny classifier (80.73% test Top-1, port 7863). A tested 71.32 s wake-on-request and subsequent idle sleep (best effort); the separate 4-hour Free Studio active-session/billing boundary still requires monitoring. |
| **Production Backup (E3)** | [Render Free Web Service](https://cos30082-bird-classifier.onrender.com/) | Pre-committed E3 ResNet18 classifier (70.43% single-view test Top-1; optional 2-view TTA) operating within 512 MiB Linux cgroup memory limit (380.7 MiB peak VmHWM, 131 MiB headroom). |
| **Deployment Repository** | [GitHub: ShadWVege/cos30082-bird-classifier](https://github.com/ShadWVege/cos30082-bird-classifier) | Standalone inference-only deployment repository supporting Render Free. |
| **Presentation Video** | `https://youtu.be/... (YOUTUBE_VIDEO_URL_PENDING)` | 10-minute recorded walkthrough and live demonstration (recording pending). |
| **Coursework Source Archive** | `outputs/source_submission.zip` | Commented source, 27 automated tests, configs, report **Typst authoring source**, and publication figures. The final 4-page **PDF is submitted separately**, and model weights are excluded. |

---

## Model Weights & Checkpoint Packaging Policy

As specified in the assignment brief (`references/Assignment1_v2.pdf`):
> *"Python program source code: Create a single zip file with your code, and submit together with the report."*

To comply with this source-code mandate and avoid LMS upload failures caused by multi-hundred-megabyte binaries, trained model checkpoints (`.pt` files; ~112 MB for ConvNeXt-Tiny and ~45 MB for ResNet18) are excluded from `outputs/source_submission.zip`.

**How to access model weights:**
1. **Regeneration:** Execute the deterministic training commands above using the frozen seed-42 split.
2. **Pre-Packaged Deployment Weights:** Verified inference weights (`model.pt`, SHA-256 `b26dce1d5f0883d9612fcc6905c0d0e5f632accc54d7d9df0dd511c9cd208d4c`) are bundled inside the standalone deployment archive `outputs/e7_lightning_deploy_v2.zip` (and split parts `.part1`/`.part2`).
3. **Live Cloud Services:** Checkpoints are actively deployed and testable on Lightning AI Free CPU Studio (`https://7863-01m4vaa4swe6b7ha57xrh4ch7s.cloudspaces.litng.ai/`) and Render Free (`https://cos30082-bird-classifier.onrender.com/`).

---

## Reproducibility & Hardware Variance Disclaimers

- All models share the frozen stratified group split (`data/prepared/split.json`, seed 42).
- Deterministic seeding covers Python, NumPy, PyTorch, and DataLoader generators. PyTorch deterministic operations are requested (`torch.use_deterministic_algorithms(True)` where supported).
- *Hardware Variance Disclaimer:* Bit-for-bit identical floating-point outputs across different GPU microarchitectures (e.g. RTX 30-series vs A100), different cuDNN library versions, or between CPU and GPU cannot be guaranteed due to non-associative atomic floating-point reductions in parallel CUDA kernels. However, metric trajectories and convergence regimes replicate consistently.
- *Cloud Hosting Disclosures:* Render Free typically spins down after inactivity (~50 s observed spin-up); a Lightning AI Gradio-plugin link once woke the Free Studio in 71.32 s and the Studio subsequently auto-slept on idle. Neither outcome guarantees future latency or uptime. The **4-hour Free Studio active-session/billing boundary still needs monitoring**, and Hugging Face hosts static documentation only. Lecturer approval for externally hosted inference remains pending.

