# Project Status and Audit Report — COS30082 Assignment 1

> **Historical development snapshot (October 2026):** This file captures an earlier milestone and includes outdated status, paths, figure inventories, and/or test counts. It is retained to document the experiment chronology, **not** as the current submission checklist. See the root [`STATUS.md`](../STATUS.md), [`README.md`](../README.md), [`VERIFICATION.md`](VERIFICATION.md), and the final [`report.typ`](../reports/report.typ) for the latest verified state. Model parameters E2/E4 were corrected to **440,296**.


**Audit Date:** 2026-10-13 (Updated after Mission 3D: Post-Test Exploratory Validation-Only Label Smoothing Ablation E5)  
**Auditor:** Antigravity Autonomous Pair-Programming Agent  
**Environment:** Windows 11, Python 3.12.14, PyTorch 2.11.0+cu128, CUDA 12.8, NVIDIA GeForce RTX 3060 Laptop GPU (6.0 GiB VRAM)

---

## 1. Executive Summary & Component Classification

| Gate / Component | Status | Evidence / Details |
|---|---|---|
| **G0: Environment & Scaffolding** | **VERIFIED COMPLETE** | Python 3.12.14 in `.venv`, PyTorch 2.11.0+cu128 with CUDA 12.8 active on RTX 3060 Laptop GPU. All 16 `pytest` tests pass (`53.64s`). |
| **G1: Dataset Audit & Frozen Split** | **VERIFIED COMPLETE** | Restored in `D:\COS30082_Datasets\Train` (4,829 images) and `Test` (1,204 images). SHA256 verified against `data/prepared/split.json`. Three conflicting training duplicate pairs pinned to training (`conflict_policy: train_only`). |
| **G2: Modelling Framework & Verification** | **VERIFIED COMPLETE** | Modular pipeline, deterministic transforms, mixed-precision AMP, checkpoint hashing, early stopping, training-only `label_smoothing`, and Windows multiprocessing safeguards verified. |
| **G3: Model Experiments & Ablations** | **VERIFIED COMPLETE (E1, E2, E3, E4, E5 + E6 Post-Test Exploratory)** | <br>• **E1 (Baseline CNN):** 40 epochs (`outputs/baseline_001`). Best ep 31: Val Top-1 21.69%, Avg/class 21.29%, Gap 58.49 pp (71.99 pp at ep 40).<br>• **E2 (Regularised CNN, Drop 0.4):** 29 epochs (`outputs/experiment_001/regularized`). Best ep 21: Val Top-1 10.43%, Avg/class 9.63%, Gap **6.19 pp**.<br>• **E4 (Controlled Ablation, Drop 0.0):** 46 epochs (`outputs/experiment_001/E4_regularized_no_dropout`). Best ep 38: Val Top-1 **14.39%**, Avg/class **13.93%**, Gap **16.08 pp**.<br>• **E3 (ResNet18 Transfer — Frozen Selected Winner):** 35 epochs (`outputs/experiment_001/resnet18`). Verified 11,279,112 params. Best ep 34: Val Top-1 **69.66%**, Avg/class **69.10%**, Top-5 **89.57%**, Gap **30.24 pp**, 10-bin ECE **0.0609** (TTA: **70.80%** Top-1 / **70.27%** Avg/Class).<br>• **E5 (ResNet18 Label Smoothing 0.05 — Post-Test Exploratory):** 35 epochs (`outputs/experiment_001/E5_resnet18_label_smoothing`). Best ep 27: Val Top-1 **68.72%**, Avg/class **68.02%**, Gap **31.18 pp**, 10-bin ECE **0.1341**.<br>• **E6 (ResNet18 Enhanced: layer3+layer4 Disc-LR — Post-Test Exploratory):** 32 epochs (`outputs/experiment_001/E6_resnet18_enhanced`). Best ep 24: Val Top-1 **68.40%**, Avg/class **67.88%**, Top-5 **88.95%**, Gap **31.23 pp**, 10-bin ECE **0.2021** (TTA: **70.70%** Top-1 / **70.14%** Avg/Class). |
| **G4: Held-Out Test Audit, One-Time Evaluation & Visual Evidence** | **VERIFIED COMPLETE** | <br>• **Held-Out Test Integrity Audit (`outputs/final_test_evaluation/test_audit.json`):** All 1,204 test images decode cleanly across all 200 classes (4–8 samples/class). Identified 4 exact pixel-identical images overlapping between official `Train` and `Test` with conflicting labels (including multi-species feeder photo `Test/0197.jpg`; documented in `docs/DATA_DECISIONS.md`).<br>• **One-Time Held-Out Test Evaluation (`outputs/final_test_results`):** Pre-committed winner **E3 ResNet18** achieved **70.43% Top-1 Accuracy** (`848/1,204`), **70.12% Average Accuracy per Class**, **89.12% Top-5 Accuracy**, and **69.90% Macro-F1** (Loss: `1.2004`). Disclosed that E1 (`21.01%`) and E2 (`8.97%`) were also scored because `outputs/experiment_001/selection.json` listed all three initial runs.<br>• **Visual Galleries & Grad-CAM (`scripts/generate_visual_evidence.py`, `scripts/compare_e3_vs_e5.py`, `scripts/compare_e3_vs_e6.py`):** Verified all figures in `outputs/experiment_001/figures/` and `outputs/final_test_results/figures/` with zero text clipping (`bbox_inches="tight"`). |
| **G5: Gradio Web App & Cloud Deployment Export** | **DEPLOYED ON RENDER FREE (`https://cos30082-bird-classifier.onrender.com/`)** | 3-Section Gradio app (`app.py`, `src/inference.py`, `src/export_space.py`) verified across 17 `pytest` tests (`21.76s`), 10 parity checks, 7 live Edge browser scenarios, and hard 512 MiB Linux cgroup v2 tests. Published to `https://github.com/ShadWVege/cos30082-bird-classifier` (`commit 4273963`) and live on Render Free Web Service. |
| **G6: Report Preparation** | **READY TO COMPILE** | All true experimental tables, test metrics, error analyses, E5/E6 ablations, TTA benchmarks, scientific integrity disclosures, and unclipped figures ready. |
| **G7: Video Presentation** | **READY TO RECORD** | 9:30 outline in `docs/SUBMISSION_PLAN.md` and visual explainability guide ready. |
| **G8: Submission Package & Audit** | **READY FOR FINAL PACKAGING** | `src/package_submission.py` verified. `docs/SUBMISSION_AUDIT.md` updated. |

---

## 2. Complete Validation & Held-Out Test Results Summary

| Run ID | Model Architecture | Total Params | Best Ep | Clean Train Top-1 | Val Top-1 ($N=959$) | Val Avg/Class | Val Gap | Held-Out Test Top-1 ($N=1,204$) | Held-Out Test Avg/Class | Test Top-5 | Test Macro-F1 | Test Loss |
|---|---|---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **E1** | Baseline CNN (From scratch) | 439,816 | 31 / 40 | 80.18% | 21.69% | 21.29% | 58.49 pp | 21.01% | 20.63% | 42.03% | 19.91% | 6.7320 |
| **E2** | Regularised CNN (Drop 0.4) | 440,296 | 21 / 29 | 16.61% | 10.43% | 9.63% | **6.19 pp** | 8.97% | 8.80% | 25.33% | 7.00% | 4.4129 |
| **E4** | Ablation: Reg CNN (Drop 0.0) | 440,296 | 38 / 46 | 30.47% | 14.39% | 13.93% | 16.08 pp | *Explanatory ablation (not test-scored)* | — | — | — | — |
| **E3** | **ResNet18 Transfer (Selected)** | **11,279,112** | **34 / 35** | **99.90%** | **69.66%** | **69.10%** | **30.24 pp** | **70.43%** | **70.12%** | **89.12%** | **69.90%** | **1.2004** |
| **E5** | **ResNet18 + Label Smoothing 0.05** | 11,279,112 | 27 / 35 | 99.90% | 68.72% | 68.02% | 31.18 pp | *Post-test validation-only exploratory study* | — | — | — | — |
| **E6** | **ResNet18 Enhanced (layer3+4 disc-LR)** | 11,279,112 | 24 / 32 | 99.64% | 68.40% | 67.88% | 31.23 pp | *Post-test validation-only exploratory study* | — | — | — | — |

---

## 3. Generated Presentation-Ready Visual Evidence & Verification Paths

### A. Model Comparison & Ablation Figures (`outputs/experiment_001/figures/`)
- `outputs/experiment_001/figures/model_comparison.png`: Annotated 3-panel plot (Clean-Train vs Val Loss, Clean-Train vs Val Top-1, Generalisation Gap) across E1, E2, E4, E3.
- `outputs/experiment_001/figures/four_model_bar_comparison.png`: Grouped bar chart of Validation Top-1, Average Accuracy per Class, Top-5, and Generalisation Gap across E1, E2, E4, E3.
- `outputs/experiment_001/figures/ablation_e2_vs_e4.png`: Controlled dropout ablation curves (`p=0.4` vs `p=0.0`) with annotated best epochs and gaps.
- `outputs/experiment_001/figures/ablation_e3_vs_e5_label_smoothing.png`: Controlled post-test transfer-learning ablation (`E3` unsmoothed vs `E5` `label_smoothing=0.05`) showing unsmoothed evaluation loss, smoothed training objective vs theoretical CE floor (`~0.4614`), Top-1 accuracy, and generalisation gap evolution.
- `outputs/experiment_001/E5_resnet18_label_smoothing/e3_vs_e5_comparison.json`: Complete metric and 10-bin ECE calibration comparison between E3 and E5.

### B & C. Validation Prediction Gallery & Grad-CAM Interpretability (`outputs/experiment_001/figures/`)
- `outputs/experiment_001/figures/val_prediction_gallery.png` (and `.json`): 6 deterministic validation predictions (3 correct at P100/P50/P10 confidence; 3 incorrect at P100/Top-Confusion-Pair/P50) with Top-5 horizontal softmax bars and Class IDs.
- `outputs/experiment_001/figures/gradcam_val_comparison.png`: 6-row × 4-column Grad-CAM comparison showing 224×224 Evaluation Crop, `layer4` Activation Heatmap, Attribution Overlay, and Top-5 Softmax bars.
- `outputs/experiment_001/figures/gradcam/01_*.png` .. `06_*.png`: Individual two-panel Grad-CAM outputs produced via `src.gradcam`.

### D. Held-Out Test Visualisations (`outputs/final_test_results/`)
- `outputs/final_test_results/figures/test_metrics_summary.png`: Held-out test metrics across E1, E2, E3 + Validation vs Test parity chart for E3 ResNet18.
- `outputs/final_test_results/figures/test_per_class_and_confusions.png`: 3-panel diagnostic figure (200-species per-class accuracy histogram, per-class test accuracy vs training sample count scatter with Pearson $r = 0.066$, and Top-12 held-out test confusion pairs).
- `outputs/final_test_results/figures/test_prediction_gallery.png` (and `.json`): 6 deterministic held-out test predictions (3 correct, 3 incorrect) including visual proof of multi-species photo `0197.jpg` (Purple Finch + American Goldfinch at a bird feeder).
- `outputs/final_test_results/resnet18/`: Full `metrics.json`, `per_class.csv`, `predictions.csv`, `confused_pairs.csv`, `confused_pairs.png`, `confusion.csv`, `confusion.npy`, `confusion.png`, `errors.png`.

### E. Gradio Parity, Live Browser & Linux 512 MiB Hosting Verification (`outputs/gradio_verification/`)
- `outputs/gradio_verification/task2_parity_verification.json`: Automated verification of all 10 parity criteria and CPU latency benchmark (Grad-CAM off vs on).
- `outputs/gradio_verification/task3_browser_verification.json`: Live Microsoft Edge browser verification results across 7 scenarios.
- `outputs/gradio_verification/linux_hosting_preflight.json`: WSL2 Linux (`Python 3.12.14`, `torch==2.11.0+cpu`, `systemd-run -p MemoryMax=512M -p MemorySwapMax=0`) memory (`VmRSS`, `VmHWM`, `RssAnon`, `RssFile`, `cgroup_anon_mib`) and latency preflight profile.
- `outputs/gradio_verification/phase_b_render_verification.json`: Phase B verification of the exported `outputs/huggingface_space` package inside a 512 MiB Linux cgroup v2 (`max_top5_prob_diff = 1.19e-7`, 200-class output shape, offline `TORCH_HOME`, 12 MP DSLR upload, >25 MP safety guard, and 2 HTTP server startup/restart cycles).
- `outputs/gradio_verification/01_bird1_woodpecker_gradcam_on.png` .. `07_standalone_space_restart_verified.png`: Full-page and mobile (`390x844`) browser screenshots.

### F. Standalone Render Free & Hugging Face Space Export (`outputs/huggingface_space/`)
- `outputs/huggingface_space/app.py`, `render.yaml`, `.python-version`, `model.pt`, `class_names.json`, `model_info.json`, `requirements.txt`, `README.md`, `.gitignore`, `assets/`, `src/` (19 files total, `44.09 MiB`, zero local paths or secrets).
- **Published GitHub Repository:** `https://github.com/ShadWVege/cos30082-bird-classifier` (branch `main`, commit `4273963aa080c3d8d411c39306648193a6b0b6dd`).

---

## 4. Next Execution Step

1. **Review Mission 4A Findings (Phase F):** E3 ResNet18 remains the superior single-view model (**69.66% Val Top-1 / 69.10% Avg/Class**) with superior calibration (**10-bin ECE 0.0609**) compared to E6 (**68.40% Val Top-1 / 67.88% Avg/Class, ECE 0.2021**). Recommendation: **KEEP E3 as the deployed model on Render Free**.
2. **Proceed to Final Academic Deliverables:** Compile the 4-page academic PDF report, record/script the 10-minute video presentation, and bundle the final source ZIP via `src.package_submission`.
