# Current Project Status

Project: D:\COS30082_BirdClassification  
Updated: 2026-10-15T00:53:00+08:00  

Current stage: **Mission 4M Completed** (Confirmed hosting ecosystem fully integrated across report, documentation, presentation, and submission package: Student Name Jonathan Zheng Li CHAI, Student ID 104400675; Primary Hugging Face Space landing page disclosed as static HTML only; Primary live E7 ConvNeXt-Tiny interactive classifier on Lightning AI Free CPU Studio port 7863 documented with empirical ~71s auto-wake and 4-hour session limits; Pre-committed E3 ResNet18 on Render Free maintained as 512 MiB production backup; Presentation video URL placeholder YOUTUBE_VIDEO_URL_PENDING preserved; Four-page academic report compiled and verified strictly 4 pages; deploy/e7_lightning/gradio_plugin.py provided; 27/27 unit tests passed; all 4 model checkpoint SHA-256 hashes verified untouched; outputs/source_submission.zip rebuilt with 96 files).

## Verified Completed Work

1. **Hosting Ecosystem Integration & Student Metadata (Mission 4M):**
   - **Student Details:** Updated `reports/report.typ`, `presentation/slides.md`, and `presentation/presentation_script.md` with:
     - Name: **Jonathan Zheng Li CHAI**
     - Student ID: **104400675**
     - Unit: **COS30082 Applied Machine Learning**
     - Assessment: **Individual Assignment 1 — Bird Species Classification**
   - **Hugging Face Space (Primary Portal):**
     - URL: `https://huggingface.co/spaces/ShadWVege/cos30082-bird-species-explorer`
     - Clearly disclosed in `reports/report.typ`, `README.md`, `deploy/e7_lightning/README.md`, and presentation materials as the static HTML project landing page and documentation portal (CNN model inference does NOT run directly on HF due to free-tier hardware limits).
   - **Lightning AI Free CPU Studio (Primary Interactive Classifier):**
     - Live URL: `https://7863-01m4vaa4swe6b7ha57xrh4ch7s.cloudspaces.litng.ai/`
     - Hosts exploratory **E7 ConvNeXt-Tiny** (80.73% test Top-1, 27.97M params, terminal `features.7` Grad-CAM).
     - Running on 4 vCPUs / 16 GB RAM via Lightning Gradio Plugin (`deploy/e7_lightning/gradio_plugin.py`, exposing `demo = build_app()`) on port 7863.
     - Documented empirical wake test: sleeping studio woke and responded in 71.32s; auto-sleeps on idle; 4-hour active session limit before manual restart.
   - **Render Free Web Service (Pre-Committed Production Backup):**
     - URL: `https://cos30082-bird-classifier.onrender.com/`
     - Retains pre-committed **E3 ResNet18** model (70.43% single-view official test Top-1; optional 2-view TTA toggle).
     - Verified safe operation under 512 MiB Linux cgroup memory limit (380.7 MiB peak VmHWM, 131 MiB genuine physical headroom).
   - **Deployment Repository:**
     - URL: `https://github.com/ShadWVege/cos30082-bird-classifier` (inference-only deployment repo for Render).
   - **Video URL Placeholder:**
     - `https://youtu.be/... (YOUTUBE_VIDEO_URL_PENDING)` preserved without fabrication.
   - **Four-Page Academic Report:**
     - Recompiled `reports/report.pdf` from `reports/report.typ` via `scripts/compile_report.py`.
     - Verified: strictly 4 pages (0 page overflow).
   - **Testing & Checkpoint Integrity:**
     - 27/27 unit tests passed in 68.82s.
     - All 4 model checkpoint hashes verified 100% untouched.
   - **Source Archive Rebuilt:**
     - `outputs/source_submission.zip`: 96 files, 9,418,646 bytes, SHA-256 `016F3E6FD98C0E621988BF8C044C66D8C9E8AC5542020E557415957C9F067615`.

2. **E7 V2 Gradio Application Corrections (Mission 4L):**
   - **Correction 1 (E3 Single-View vs TTA Labeling):** Updated Section 4 platform comparison table in `deploy/e7_lightning/app.py`:
     `| **Model Deployed** | E3 ResNet18 (11.28M parameters; 70.43% official test Top-1, single-view). Optional TTA available; official TTA test performance not evaluated. | E7 ConvNeXt-Tiny (27.97M parameters; 80.73% official test Top-1, single-view). Optional TTA available (80.32% test Top-1). |`
     Audited all tabs, tooltips, and header; confirmed no attribution of E3 single-view test score to TTA.
   - **Correction 2 (Benchmark Integrity Footnote):** Replaced footnote in Tab 2 with exact authoritative statement:
     `"Official test results are available for E1, E2, E3, E7 and E7+TTA. E3+TTA, E4, E5 and E6 have development-validation results only. E7 was developed following inspection of earlier E3 test results and is therefore disclosed as an exploratory post-test milestone. Missing official test metrics are marked 'Not evaluated'; none have been estimated."`
     Verified all 7 experiment metrics in the table remain unchanged and authoritative.
   - **Correction 3 (Unicode Statistical Notation & Exact Gains):** Replaced raw LaTeX syntax (`$p = 2.48 \times 10^{-15}$`, `$\chi^2 = 60.04$`) with clean Unicode text:
     `Exact paired McNemar test: p = 2.48 × 10⁻¹⁵ (Edwards continuity-corrected χ² = 60.04, df = 1, p < 0.0001; exact two-sided p ≈ 2.4849 × 10⁻¹⁵)`.
     Changed `"+10.30 percentage-point generalisation gain"` to `"+10.30 percentage-point improvement in official test Top-1 accuracy"`.
   - **Correction 4 (Noise Ceiling Terminology Removal):** Removed unsupported "~0.33% noise ceiling" phrase; replaced with:
     `"four known cross-split pixel-duplicate pairs, corresponding to approximately 0.33% of the 1,204-image official test set, are documented as a benchmark integrity limitation."`
   - **Correction 5 (Benchmark Table Readability & Responsive Scrolling):** Embedded `.benchmark-table-wrapper` responsive CSS container (`overflow-x: auto !important; min-width: 980px`); added `white-space: nowrap` spans around `Params` (`27,973,928`), `Exp` IDs (`E3+TTA`, `E7+TTA`), and metric column headers.
   - **Correction 6 (Educational Clarity):** Added Section 4.2 *"Fine-Grained Transfer Learning & Generalisation Dynamics"* explaining why pretrained ImageNet representations solve small-data memorization, why two-stage warmup prevents feature shock, why E7 retains a 19.42 pp generalisation gap, and re-verified Section 2's post-hoc attribution disclosure for Grad-CAM.

2. **Verification & Testing:**
   - Local Gradio app construction verified without warnings or errors.
   - Single-view and 2-view TTA inference + Grad-CAM generation tested and passed.
   - Automated unit test suite passed: **27/27 passed in 37.35s**.
   - Deployment weights hash verified 100% untouched:
     `deploy/e7_lightning/model.pt` SHA-256: `b26dce1d5f0883d9612fcc6905c0d0e5f632accc54d7d9df0dd511c9cd208d4c`.
   - Updated `deploy/e7_lightning/app.py` SHA-256: `6848dbcc29ffd925e0b406f66ae15386888bd2ce1e7d89e05e401092370ad9a4`.
   - Packaged lightweight patch archive: `outputs/e7_lightning_app_v2_patch.zip` (9,951 bytes, SHA-256 `14a63f83b5a060a36c5c95ab9ac4ffabef9aafafac34e739442efd60ef4e12d8`).
   - Rebuilt full deployment archive: `outputs/e7_lightning_deploy_v2.zip` (102.30 MB, SHA-256 `18ab32eebb09a7e5bb9eb6dbce0c03c6061c3250cd50541c50e05ffe5bdeaf29`) and split parts.

3.
   - Corrected `README.md` lines 18–30: changed `train_dir`/`test_dir` to `train_root`/`test_root`.
   - Included all required metadata fields (`train_manifest`, `test_manifest`, `prepared_dir`, `num_classes`, `expected_train_count`, `seed`, `val_fraction`, `conflict_policy`) matching `configs/data.example.json`, `src/utils.py`, and `src/train.py`.
   - Preserved all data loading routines without code modifications.

2. **Figure 1 Caption & Text Precision (Task 2):**
   - Replaced obsolete learning-curve caption in `reports/report.typ` with an accurate description of the actual 3-panel comparative bar charts in `outputs/experiment_001/figures/model_comparison.png`:
     - (a) Clean-train vs validation Top-1 accuracy across all seven pipelines (N=3,870 / 959).
     - (b) Validation Top-1, Average Accuracy per Class, and Top-5 across all seven pipelines.
     - (c) Generalization gap (clean-train minus validation Top-1 in pp).
   - Adjusted Section 3 introductory text to reference benchmarks in Figure 1 and Table 1 without citing Figure 1 for epoch loss trajectories.

3. **Figure 4 Grad-CAM Source Grounding (Task 3):**
   - Corrected Figure 4 caption in `reports/report.typ`: changed "on representative test images" to "on illustrative bird examples selected from the training dataset (`train_root`)".
   - Updated Section 3.4 prose to state that Case 1 (`Red_headed_Woodpecker_0012`, class 190) and Case 2 (`American_Goldfinch_0013`, class 46) are illustrative examples from `train_root`.

4. **Figure 5 Three-Panel Held-Out Test Diagnostics (Task 4):**
   - Corrected Figure 5 caption in `reports/report.typ` to detail all three panels of `outputs/final_test_results/figures/test_per_class_and_confusions.png`:
     - (a) Per-class test accuracy distribution histogram across 200 species (median 75.0%, 45 classes at 100%, 4 classes at <=25%).
     - (b) Per-class test accuracy vs training sample count scatter (Pearson $r = 0.048$, showing discrete recall steps under 4–8 test samples/class).
     - (c) Top misclassified species pairs highlighting fine-grained sibling ambiguities (e.g. vireos, gulls, warblers).

5. **Source Archive Description Update (Task 5):**
   - Updated line 329 in `reports/report.typ` to read: `outputs/source_submission.zip (complete standalone coursework archive; commented source, configs, tests, docs, figures; checkpoint weights excluded per LMS brief)` to avoid brittle file-count contradictions.
   - Recompiled `reports/report.pdf` via `scripts/compile_report.py`; verified page count is strictly 4 pages.

6. **Presentation Script & Slides Scientific Rigor (Task 6):**
   - `presentation/presentation_script.md` Segment 8: removed claims that Grad-CAM proves biological reasoning; framed as spatial feature attribution verifying focus on avian regions rather than background shortcuts.
   - Segment 8 narration: updated to describe the actual 2-model comparison on Slide 9 (E3 ResNet18 vs E7 ConvNeXt-Tiny) instead of asserting Slide 9 displays E1.
   - Segment 9: replaced "rock-solid" claims with realistic free-tier disclosures (131 MiB genuine physical headroom, VmHWM 380.7 MiB, subject to standard 15-minute idle spin-down).
   - Segment 10: clarified that the GitHub repository is inference/deployment focused, with full training workflows packaged in the source ZIP.
   - `presentation/slides.md` Slide 9: updated bullets to reflect E3 vs E7 Grad-CAM comparison with interpretability note.

7. **Verification, Packaging & Regression Testing:**
   - 27/27 unit tests passed in 34.15s (`tests/test_data.py`, `tests/test_inference_tta.py`, `tests/test_metrics_models.py`, `tests/test_workflow.py`).
   - Four model checkpoint SHA-256 hashes verified 100% untouched:
     - E3: `b7b74291077ead2315a28627d40a0cfb5cdda574fcb38334eaa74c57cca3796b`
     - E7: `18cfc800908652c7b055e7a13e4369f995ec4a91b4ae4b1b7f134f4722d015b2`
     - Deploy E7: `b26dce1d5f0883d9612fcc6905c0d0e5f632accc54d7d9df0dd511c9cd208d4c`
     - Deploy E3: `62df652932bfe1e6d6130878e6c4ada4b13e95062d5a2f8705649c066b3a111a`
   - Rebuilt `outputs/source_submission.zip`: 95 files, 9,412,750 bytes (~8.98 MB), SHA-256 `b50e0d6864ec260ef22856f492e91e12d5131c1e1c30c34c690ca8afe4d78023`.

## Active Artifacts & Paths

- Student Code Walkthrough Guide: `docs/CODE_WALKTHROUGH.md` (Portable relative links)
- Standalone Source Code ZIP: `outputs/source_submission.zip` (96 files, 9,418,646 bytes, SHA-256: `016F3E6FD98C0E621988BF8C044C66D8C9E8AC5542020E557415957C9F067615`)
- Four-Page Report PDF: `reports/report.pdf` (verified: strictly 4 pages)
- Typst Source: `reports/report.typ`
- Presentation Slides: `presentation/slides.md`
- Presentation Script: `presentation/presentation_script.md`
- Primary Project Portal (HF Space): `https://huggingface.co/spaces/ShadWVege/cos30082-bird-species-explorer`
- Primary Interactive Classifier (Lightning AI E7): `https://7863-01m4vaa4swe6b7ha57xrh4ch7s.cloudspaces.litng.ai/`
- Backup Live Render Deployment (E3): `https://cos30082-bird-classifier.onrender.com/`
- Deployment Repository (GitHub): `https://github.com/ShadWVege/cos30082-bird-classifier`
- Frozen Selected Checkpoint (E3): `outputs/experiment_001/resnet18/checkpoints/best.pt`
- Exploratory Checkpoint (E7): `outputs/experiment_001/E7_convnext_tiny/checkpoints/best.pt`
- Standalone E7 Deployment Package: `deploy/e7_lightning/`
- Lightning Gradio Plugin Entrypoint: `deploy/e7_lightning/gradio_plugin.py`
- Lightweight App Patch Archive: `outputs/e7_lightning_app_v2_patch.zip` (9,951 bytes, SHA-256: `14a63f83b5a060a36c5c95ab9ac4ffabef9aafafac34e739442efd60ef4e12d8`)
- Updated Deployment Archive: `outputs/e7_lightning_deploy_v2.zip` (102.30 MB, SHA-256: `18ab32eebb09a7e5bb9eb6dbce0c03c6061c3250cd50541c50e05ffe5bdeaf29`)
- Official Held-Out Test Evaluation: `outputs/final_test_results/test_comparison.csv`

## Next Execution Step

- Present the Mission 4M completion summary to the user.
- Student records the 10-minute presentation video following `presentation/presentation_script.md` and `presentation/slides.md`.
- Replace the video placeholder `https://youtu.be/... (YOUTUBE_VIDEO_URL_PENDING)` with the final YouTube URL when published.

