# Submission Audit & Rubric Compliance Matrix

> **Historical development snapshot (October 2026):** This file captures an earlier milestone and includes outdated status, paths, figure inventories, and/or test counts. It is retained to document the experiment chronology, **not** as the current submission checklist. See the root [`STATUS.md`](../STATUS.md), [`README.md`](../README.md), [`VERIFICATION.md`](VERIFICATION.md), and the final [`report.typ`](../reports/report.typ) for the latest verified state. Model parameters E2/E4 were corrected to **440,296**.


**Project:** COS30082 Assignment 1 — Bird Species Classification  
**Date:** 2026-10-13 (Updated after Mission 3D: Post-Test Exploratory Validation-Only Label Smoothing Ablation E5)  
**Target Mark:** 20 / 20

---

## Rubric Item 1: Modelling Framework (7 Marks)

| Criterion | Requirement | Implementation Evidence | Status |
|---|---|---|---|
| **Dataset & Pipeline** | Original CUB-200 split (4,829 train, 1,204 test, 200 classes). Frozen ~80/20 stratified dev split (seed 42). Exact duplicates handled without leakage. | Images in `D:\COS30082_Datasets\Train` (4,829) and `Test` (1,204). All SHA256 hashes verified against `data/prepared/split.json`. Three conflicting training duplicate pairs pinned to train (`conflict_policy: train_only`). Four cross-split `Train`/`Test` exact pixel duplicates (including multi-species photo `Test/0197.jpg`) documented in `docs/DATA_DECISIONS.md`. | **VERIFIED COMPLETE** |
| **Reproducibility** | Seed control, deterministic validation/test transforms, PyTorch DataLoader safeguards. | `src/dataset.py`, `src/engine.py` (`seed_worker`, `pin_memory`, 0 workers default on Windows, training-only `label_smoothing`). Pytest test suite (16/16 passed in 53.64s). | **VERIFIED COMPLETE** |
| **Model Architectures** | E1: Baseline CNN (439,816 params).<br>E2: Regularised CNN (440,296 params, Dropout 0.4).<br>E3: Pretrained ResNet18 (11,279,112 params, `Linear(512, 200)`).<br>E4: Controlled Ablation (440,296 params, Dropout 0.0).<br>E5: Post-Test Exploratory ResNet18 (`label_smoothing=0.05`, 11,279,112 params). | `src/models.py` (`CustomCNN`, `build_model`, staged freezing and BatchNorm eval mode handling in `training_mode`). `outputs/*/architecture.txt`. | **VERIFIED COMPLETE** |
| **E1 Baseline Results** | Authentic training with genuine overfitting evidence. | `outputs/baseline_001`: 40 epochs on RTX 3060 GPU. Best epoch 31: Val Top-1 21.69%, Avg/class 21.29%, Gap 58.49 pp (71.99 pp at epoch 40). Held-out Test Top-1: **21.01%**, Avg/class: **20.63%**. | **VERIFIED COMPLETE** |
| **E2 Regularised Results** | Authentic training with controlled generalisation techniques. | `outputs/experiment_001/regularized`: 29 epochs on RTX 3060 GPU. Best epoch 21: Val Top-1 10.43%, Avg/class 9.63%, Gap **6.19 pp**. Held-out Test Top-1: **8.97%**, Avg/class: **8.80%**. | **VERIFIED COMPLETE** |
| **E4 Controlled Ablation** | Isolate effect of Dropout (`p=0.0` vs `p=0.4` in E2) with all other factors identical. | `outputs/experiment_001/E4_regularized_no_dropout`: 46 epochs on RTX 3060 GPU. Best epoch 38: Val Top-1 **14.39%** (+3.96 pp vs E2), Avg/class **13.93%** (+4.30 pp vs E2), Gap **16.08 pp**. | **VERIFIED COMPLETE** |
| **E3 ResNet18 Results** | Authentic transfer learning with head warmup + partial fine-tuning. | `outputs/experiment_001/resnet18`: 35 epochs on RTX 3060 GPU. Best epoch 34: Val Top-1 **69.66%**, Avg/class **69.10%**, Top-5 **89.57%**, Train-Val Gap **30.24 pp**, 10-bin ECE **0.0609**. Held-out Test Top-1: **70.43%**, Avg/class: **70.12%**, Top-5: **89.12%**, Macro-F1: **69.90%**. | **VERIFIED COMPLETE** |
| **E5 Post-Test Exploratory Ablation** | Investigate `label_smoothing=0.05` on ResNet18 (validation-only; official test set untouched). | `outputs/experiment_001/E5_resnet18_label_smoothing`: 35 epochs on RTX 3060 GPU. Best epoch 27: Val Top-1 **68.72%**, Avg/class **68.02%**, Top-5 **88.11%**, Gap **31.18 pp**, 10-bin ECE **0.1341**. Prevented clean-train loss collapse (`0.0470` vs `0.0060`) and slashed wrong-prediction confidence (`31.31%` vs `53.86%`), while uniform smoothing slightly compressed fine-grained sibling-species margins. | **VERIFIED COMPLETE** |
| **One-Time Held-Out Test Evaluation** | Freeze validation choice before opening test set; compute Top-1 and Average Accuracy per Class over all 200 classes. | `outputs/final_test_results/test_comparison.csv`, `provenance.json`, `test_audit.json`, and per-model folders (`resnet18/metrics.json`, `per_class.csv`, `confused_pairs.csv`, `predictions.csv`). Disclosed in `docs/DATA_DECISIONS.md` that E1/E2 were also scored because `outputs/experiment_001/selection.json` listed all three candidate runs, while E4 and E5 were never evaluated on the held-out test set. | **VERIFIED COMPLETE** |

---

## Rubric Item 2: Written Report (7 Marks)

| Criterion | Requirement | Implementation Evidence | Status |
|---|---|---|---|
| **Length & Sections** | Max 4 pages PDF. Exactly two main sections: **Methodology** and **Result and Discussion**. | `docs/SUBMISSION_PLAN.md` defines 4-page allocation. | **READY TO COMPILE** |
| **Student Info** | Name and Student ID included. | To be provided by student (placeholder retained until confirmed). | **PENDING USER INPUT** |
| **Architectures & Hyperparams** | Model details, conv channels, loss, optimizers, learning rates, schedules, regularisation. | Fully documented in `configs/`, `src/models.py`, `outputs/*/architecture.txt`. | **VERIFIED COMPLETE** |
| **Evidence & Figures** | True comparison tables, learning curves, confusion matrix, per-class analysis, error examples, Grad-CAM. | Visually audited (`bbox_inches="tight"`, zero clipping) in `outputs/experiment_001/figures/` (`model_comparison.png`, `four_model_bar_comparison.png`, `ablation_e2_vs_e4.png`, `val_prediction_gallery.png`, `gradcam_val_comparison.png`) and `outputs/final_test_results/figures/` (`test_metrics_summary.png`, `test_per_class_and_confusions.png`, `test_prediction_gallery.png`). | **VERIFIED COMPLETE** |
| **Authentic Links** | YouTube link and Hugging Face Space link embedded in report. | Real links will be embedded once deployed and recorded. No fabricated links. | **PENDING LIVE ASSETS** |

---

## Rubric Item 3: Gradio Web Application (3 Marks)

| Criterion | Requirement | Implementation Evidence | Status |
|---|---|---|---|
| **UI Functionality** | Section 1 (Bird Identification: upload, preview, species name, zero-based class ID `000–199`, Top-5 percentage bars & softmax table, reset, invalid/tiny upload handling, 200-species closed-set notice), Section 2 (Grad-CAM Explainability: `224x224` crop, `layer4` heatmap, overlay, cautious explanation, no recompute when unchecked), Section 3 (Model Insights: actual curves, E1–E4 comparison, test results). | `app.py` and `src/inference.py` verified via 10 automated parity checks (`outputs/gradio_verification/task2_parity_verification.json`), 7 live Microsoft Edge browser scenarios (`outputs/gradio_verification/01_*.png` .. `07_*.png`), hard 512 MiB Linux cgroup v2 tests (`outputs/gradio_verification/phase_b_render_verification.json`), and `pytest` (`16 passed`). | **VERIFIED COMPLETE** |
| **Pipeline Parity** | Exact preprocessing and class name mapping identical to training/validation/test. | `src/inference.py` loads `model.pt` (`weights_only=True`, `mmap=True` on Linux CPU, `meta` device + `assign=True`) and `class_names.json`, using identical `image_transform` from `src/dataset.py` (`max_top5_prob_diff = 1.19e-7`, `max_overlay_uint8_diff = 0`). | **VERIFIED COMPLETE** |
| **Standalone Export** | Clean export package without training dataset archives or secrets. | `outputs/huggingface_space` generated by `src/export_space.py` with `app.py`, `render.yaml` (`plan: free`), `.python-version` (`3.12.10`), `model.pt` (E3 ResNet18 Epoch 34), `model_info.json`, `class_names.json`, `requirements.txt` (`--extra-index-url https://download.pytorch.org/whl/cpu`), `README.md`, `.gitignore`, `assets/`, `src/`. | **VERIFIED COMPLETE** |
| **Live Deployment** | Published and verified on cloud hosting (Render Free Web Service / Hugging Face Space backup). | Published 19-file package to `https://github.com/ShadWVege/cos30082-bird-classifier` (`main`, commit `4273963aa080c3d8d411c39306648193a6b0b6dd`); awaiting Render Free Web Service creation & live URL verification (Phase E). | **PUBLISHED TO GITHUB / READY FOR RENDER** |

---

## Rubric Item 4: Video Presentation (3 Marks)

| Criterion | Requirement | Implementation Evidence | Status |
|---|---|---|---|
| **Duration** | Maximum 10 minutes (target 9:30). | Structured timeline in `docs/SUBMISSION_PLAN.md`. | **VERIFIED PLAN** |
| **Content Coverage** | Concepts & design, code explanation, training logs/results demonstration, live HF web app demo. | Visual explainability notes and slide/script structure prepared. | **IN PROGRESS** |
| **Delivery** | Recorded and uploaded to YouTube by student. | Student action required. URL to be embedded in report. | **PENDING RECORDING** |

---

## Rubric Item 5: Submission Source Archive

| Criterion | Requirement | Implementation Evidence | Status |
|---|---|---|---|
| **Packaging** | Single ZIP containing commented Python code, configs, dependencies, README. No datasets, envs, or secrets. | `src/package_submission.py` tested in `tests/test_workflow.py`. | **VERIFIED COMPLETE** |
