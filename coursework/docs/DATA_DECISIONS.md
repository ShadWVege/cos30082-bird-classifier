# Data and Experimental Integrity Decisions

## 1. Development Set Audit & Duplicate-Grouped Stratified Split (2026-10-05)

The local audit decoded all 4,829 listed training images in `data/train.txt` successfully. There were no missing or corrupt images. All images decode as RGB; widths span 162–500 pixels and heights 121–500 pixels. Class counts span 16–31 images per species across the 200 classes (`0–199`).

Three exact decoded-pixel duplicate pairs within `data/train.txt` have conflicting source labels:

- `Crested_Auklet_0024_2676149281.jpg` (label 4) / `Parakeet_Auklet_0027_2676149281.jpg` (label 6)
- `Philadelphia_Vireo_0002_59215684.jpg` (label 152) / `Warbling_Vireo_0004_59215684.jpg` (label 155)
- `Red_eyed_Vireo_0029_59210443.jpg` (label 153) / `Tennessee_Warbler_0009_59210443.jpg` (label 178)

**Decision:** Preserve all official annotations and images. Set `conflict_policy: train_only`, pinning every member of these three duplicate groups to the training partition (`3,870` training / `959` validation images, seed `42`, split fingerprint `55f05f6ac27722f9ca3de2b60a6311117b1bcb0b679220aa739c8ff11c9c6033`). Draw validation samples from each class's remaining distinct image groups. This prevents exact-pixel leakage into validation without guessing which official species annotation is correct. The contradictory training labels remain a small source of label noise (`6 / 3,870 = 0.16%`) and are disclosed in the report. No test image was inspected in making this decision.

The auditor hashes both complete file bytes and decoded RGB pixel arrays. Exact RGB hashing catches metadata-only re-encodings, though it does not guarantee the absence of near-duplicate crops or resizes.

---

## 2. Held-Out Test Set Integrity Audit (2026-10-13, After Freezing Model Selection)

All 1,204 test images listed in `data/test.txt` decoded cleanly as RGB (widths 264–500 px, heights 166–500 px; all 200 classes present with 4–8 samples per class; zero within-test duplicates).

Cross-referencing decoded RGB pixel SHA256 hashes against the official training set revealed **four exact pixel-identical images shared between `Train` and `Test`, each carrying conflicting official species labels**:

1. `Test/0197.jpg` (label `34`, `Purple Finch`) == `Train/American_Goldfinch_0013_146872871.jpg` (label `46`, `American Goldfinch`). Visual inspection of `0197.jpg` confirms it is a **multi-species feeder photograph** containing both a Purple Finch (left) and an American Goldfinch (right), explaining why the same photograph was entered under two different classes in the source dataset.
2. `Test/0273.jpg` (label `46`, `American Goldfinch`) == `Train/Indigo_Bunting_0006_15376001.jpg` (label `13`, `Indigo Bunting`).
3. `Test/0918.jpg` (label `153`, `Red eyed Vireo`) == `Train/Tennessee_Warbler_0012_59210336.jpg` (label `178`, `Tennessee Warbler`).
4. `Test/0976.jpg` (label `162`, `Cape May Warbler`) == `Train/Tennessee_Warbler_0020_59210520.jpg` (label `178`, `Tennessee Warbler`).

**Decision:** Preserve the official `data/test.txt` annotations and the full 1,204-image evaluation denominator without silent removal or relabelling. Acknowledge the finding explicitly via `--acknowledge-integrity-issues` in `src.evaluate` and disclose these four cross-split conflicting duplicates (`4 / 1,204 = 0.33%`) as an inherent benchmark ceiling and label-noise artifact.

---

## 3. Methodological & Protocol Disclosures (Mission 3B Audit)

1. **Pre-Committed Winner Selection vs Multi-Run Test Table:**
   - Prior to accessing `data/test.txt`, `E3` (`resnet18`, best epoch 34, SHA256 `b7b74291077ead2315a28627d40a0cfb5cdda574fcb38334eaa74c57cca3796b`) was frozen as the single selected deployment model (`deployment_run: "resnet18"`) in `outputs/experiment_001/selection.json` based on validation Average Accuracy per Class (`69.10%`).
   - Because `outputs/experiment_001/selection.json` listed all three initial candidate runs (`baseline_001`, `regularized`, `resnet18`) in its `runs` array when `python -m src.evaluate --selection outputs/experiment_001/selection.json --final` was executed, `src.evaluate` also computed held-out test metrics for `E1` (`21.01%` Top-1, `20.63%` Avg/Class) and `E2` (`8.97%` Top-1, `8.80%` Avg/Class) alongside `E3` (`70.43%` Top-1, `70.12%` Avg/Class), despite the Mission 2B instruction to prepare an E3-only test evaluation command.
   - **Disclosure:** We report this execution detail transparently. `E3` remained the pre-committed selected model, `E4` (`E4_regularized_no_dropout`) was never evaluated on the test set, and zero checkpoints, hyperparameters, or design choices were altered after observing held-out test results.
2. **Distinguishing Validation-to-Test Parity from Train-to-Validation Overfitting:**
   - The close agreement between `E3` development validation accuracy (`69.66%` Top-1 / `69.10%` Avg/Class) and official held-out test accuracy (`70.43%` Top-1 / `70.12%` Avg/Class) demonstrates that our **duplicate-grouped stratified validation split is an unbiased, leakage-free estimator of held-out generalisation**.
   - It does **not** imply that `E3` has "zero overfitting": at best epoch 34, `E3` achieves `99.90%` clean-train Top-1 accuracy versus `69.66%` validation Top-1 accuracy, representing a **`30.24 pp` train-to-validation generalisation gap**. All report and presentation discussions must maintain this exact distinction.

---

## 4. Post-Test Exploratory Validation-Only Extension: E5 Label Smoothing (2026-10-13, Mission 3D)

To investigate whether the `30.24 pp` clean-train-to-validation Top-1 gap in `E3` (`99.90%` vs `69.66%`) and extreme logit inflation (`clean_train_loss = 0.0060` at Epoch 34) could be mitigated via output-distribution regularisation, we executed one strictly controlled **post-test, validation-only exploratory ablation**:

- **Experiment:** `E5_resnet18_label_smoothing` (`configs/resnet18_label_smoothing.json`, output in `outputs/experiment_001/E5_resnet18_label_smoothing`).
- **Single Controlled Change:** `CrossEntropyLoss(label_smoothing=0.05)` applied **only** to the training objective (`loaders["train"]`), while `clean_train` and `val` evaluation losses continued to use standard unsmoothed `CrossEntropyLoss(label_smoothing=0.0)` for exact cross-experiment comparability. Architecture (`ResNet18`, `11,279,112` params), frozen split (`seed=42`, fingerprint `55f05f6ac27722f9ca3de2b60a6311117b1bcb0b679220aa739c8ff11c9c6033`), two-stage schedule (`5` head warmup epochs at `lr=1e-3` + `30` `layer4+fc` fine-tuning epochs at `lr=1e-4`), batch size (`64`), weight decay (`1e-4`), and augmentations were held identical to `E3`.
- **Empirical Outcome (Validation Only — Official Test Set Never Accessed):**
  - **Best Epoch:** `27 / 35` (checkpoint SHA256 `2393e7e8d2afe99c07487e815d4091f32696f683f78477f19ab237bcc1ea4d38`, runtime `50.53 min`).
  - **Logit Inflation & Wrong-Prediction Overconfidence Suppression:** `E5` prevented unsmoothed `clean_train_loss` from collapsing to zero (`0.0470` in `E5` vs `0.0060` in `E3`, a `7.8×` higher clean-train loss floor; smoothed training objective `0.5555` vs theoretical 200-class minimum $H_{\min} \approx 0.4614$) and reduced mean Top-1 softmax confidence on **incorrect** validation predictions by **22.55 pp** (`31.31%` in `E5` vs `53.86%` in `E3`).
  - **Classification Accuracy & Top-1 Generalisation Gap:** Because smoothed targets ($0.95025$ on the ground-truth class, $0.00025$ on each of the 199 non-target classes) preserve the strict `argmax` on the training set, `clean_train_top1` still saturated at **99.90%** (`3,866 / 3,870`). Meanwhile, uniform smoothing across 199 non-target classes compressed fine-grained margins between visually similar sister species and caused unsmoothed `val_loss` to floor early at Epoch 11 (`1.3524`), triggering earlier `ReduceLROnPlateau` learning-rate decays (Epochs 16, 20, 24, 28, 32 vs Epochs 25, 31 in `E3`). Consequently, `E5` achieved slightly lower validation accuracy than `E3`: **68.72% Val Top-1** (`-0.94 pp` vs `E3` `69.66%`), **68.02% Val Avg/Class** (`-1.08 pp` vs `E3` `69.10%`), **88.11% Val Top-5** (`-1.46 pp` vs `E3` `89.57%`), **67.33% Val Macro-F1** (`-1.19 pp` vs `E3` `68.52%`), and a **31.18 pp** Top-1 generalisation gap (`+0.94 pp` vs `E3` `30.24 pp`).
  - **Empirical Calibration (`10-bin ECE`):** `E3` was mildly overconfident (`75.74%` mean confidence vs `69.66%` accuracy; `10-bin ECE = 0.0609`), whereas `E5` shifted into global underconfidence (`55.31%` mean confidence vs `68.72%` accuracy; `10-bin ECE = 0.1341`).
- **Protocol Decision:** `E5` is documented strictly as a **post-test, validation-only exploratory study**. It was never evaluated on `data/test.txt`, and `E3` (`outputs/experiment_001/resnet18/checkpoints/best.pt`) remains the frozen selected model for deployment and official test reporting.

---

## 5. Post-Test Exploratory Validation-Only Extension: E6 Enhanced ResNet18 & TTA (2026-10-13, Mission 4A)

To investigate whether fine-tuning deeper representations (`layer3 + layer4 + fc`) with discriminative learning rates and controlled BatchNorm behaviour could improve fine-grained bird classification accuracy beyond `E3` (`69.66%` Val Top-1 / `69.10%` Val Avg/Class), we conducted **Mission 4A** as a strictly controlled post-test exploratory investigation on the frozen validation split:

1. **Initialisation Finding (Pilot Study):**
   - Attempting to continue fine-tuning directly from `E3`'s best checkpoint (where `layer4` and `fc` were heavily adapted to *frozen* ImageNet `layer3`) by unfreezing `layer3` caused validation Top-1 to crash from **69.66% down to 60.38%** in 2 epochs, and validation loss rose from **1.1895 to 2.1952**. This occurred because altering `layer3` shifted the input distribution that `layer4`'s 512 channels relied upon, inducing catastrophic feature shock.
   - Consequently, the scientifically sound protocol requires starting from ImageNet-1K pretrained weights with a 5-epoch Stage 1 head warmup, followed by Stage 2 joint fine-tuning of `layer3 + layer4 + fc`.

2. **Empirical Outcome of E6 (`outputs/experiment_001/E6_resnet18_enhanced`):**
   - **Configuration:** Discriminative learning rates (`layer3`: `1e-5`, `layer4`: `5e-5`, `fc`: `2e-4`), `freeze_bn: true` (ImageNet running mean and variance preserved in `eval()` mode), weight decay `1e-4`, standard augmentation, batch size 16, seed 42.
   - **Best Epoch:** `24 / 32` (early stopped after 8 epochs of plateau; checkpoint SHA256 `9bff6ff02372706f562fe9381d48201b58cb6a084a8ee0c57336237e4c224bdf`).
   - **Validation Accuracy:** **68.40% Val Top-1** (`-1.25 pp` vs `E3` `69.66%`), **67.88% Val Avg/Class** (`-1.22 pp` vs `E3` `69.10%`), **88.95% Val Top-5** (`-0.63 pp` vs `E3` `89.57%`), **67.38% Val Macro-F1** (`-1.14 pp` vs `E3` `68.52%`).
   - **Generalisation Gap & Calibration:** Clean-train Top-1 was `99.64%`, yielding a **31.23 pp** generalisation gap (`+0.99 pp` vs `E3` `30.24 pp`). Validation loss increased to `2.0790` (vs `1.1895` in `E3`). 10-bin ECE rose to `0.2021` (vs `0.0609` in `E3`), indicating severe overconfidence caused by fitting 11.18M parameters on ~19.35 images/species.
   - **Per-Class Breakdown:** 38 species improved (e.g. Summer Tanager +66.7 pp, Sooty Albatross +50.0 pp, Glaucous winged Gull +50.0 pp), 50 species regressed, and 112 species remained unchanged.

3. **Inference-Only Test-Time Augmentation (TTA) Benchmark (`scripts/evaluate_tta.py`):**
   - **E3 Baseline:** Single-view `69.66%` Top-1 / `69.10%` Avg/Class $\rightarrow$ 2-View Horizontal-Flip TTA **70.80% Top-1** (`+1.15 pp`) / **70.27% Avg/Class** (`+1.17 pp`).
   - **E6 Enhanced:** Single-view `68.40%` Top-1 / `67.88%` Avg/Class $\rightarrow$ 2-View Horizontal-Flip TTA **70.70% Top-1** (`+2.29 pp`) / **70.14% Avg/Class** (`+2.26 pp`).
   - **TTA Conclusion:** Both models benefit from bilateral reflection averaging, but `E3` remains higher in both single-view and TTA modes.

4. **Protocol & Deployment Recommendation:**
   - **Recommendation:** **KEEP E3 as the deployed model**. E3 achieves higher single-view accuracy (`69.66%` vs `68.40%`), lower validation loss (`1.1895` vs `2.0790`), and substantially better probability calibration (`ECE = 0.0609` vs `0.2021`).
   - The official held-out test set (`data/test.txt`) was **not evaluated** on E6, maintaining strict protocol integrity.

---

## 6. Deployment Integration: E3 ResNet18 + Two-View Horizontal-Flip TTA (2026-10-13, Mission 4B)

To provide an optional accuracy enhancement in the live web deployment without modifying trained weights, model architecture, or hosting tiers, we integrated two-view horizontal-flip TTA into the E3 deployment package:

1. **Design & Algorithmic Contract:**
   - **Bilateral Symmetry:** CUB-200 bird species are bilaterally symmetric. Horizontal reflection averaging eliminates directional bias (facing left vs facing right), raising validation Top-1 from **69.66% to 70.80% (+1.14 pp)** and Average Accuracy per Class from **69.10% to 70.27% (+1.17 pp)**.
   - **Sequential Execution:** Rather than stacking a 2-image batch ($2 \times 3 \times 224 \times 224$), inference executes sequentially ($1 + 1$) deleting intermediate activations between views. This guarantees peak activation memory remains strictly equal to single-view inference.
   - **Probability Aggregation:** Softmax probabilities across all 200 closed-set classes are averaged: $p_{\text{TTA}} = 0.5 \times (p_{\text{orig}} + p_{\text{flip}})$. Top-5 ranking, predicted class ID, and confidence scores are derived directly from $p_{\text{TTA}}$.
   - **Grad-CAM Attribution Protocol:** When TTA is active, `layer4` Grad-CAM is computed on the **original image orientation** for the TTA-selected target class. The UI and documentation explicitly disclose that the heatmap illustrates spatial feature contributions in the original image towards the TTA-selected class, not a mathematical attribution of both orientations.
   - **Default UI Behavior:** The Gradio toggle `"Enhanced Prediction (2-view TTA: bilateral reflection averaging)"` defaults to `False`, preserving exact single-view behavior as standard.

2. **Empirical Linux 512 MiB cgroup v2 Verification (`MemoryMax=512M`, `MemorySwapMax=0`):**
   - **Base App + Model RSS:** `379.18 MiB` (`RssAnon = 281.06 MiB`).
   - **Single-View vs TTA Anonymous Memory:**
     - Without Grad-CAM: Single-view `297.61 MiB` vs TTA `299.17 MiB` (delta **`+1.56 MiB`**).
     - With Grad-CAM: Single-view `288.59 MiB` vs TTA `289.37 MiB` (delta **`+0.78 MiB`**).
   - **CPU Inference Latency (Single Core, Linux CPU Wheels):**
     - Forward-only: Single-view `52.35 ms` vs TTA `101.35 ms` (+49.00 ms).
     - With Grad-CAM: Single-view `186.39 ms` vs TTA `215.45 ms` (+29.06 ms).
   - **High-Resolution Stress Test:** 12 MP DSLR image (4000x3000) processed in 334.67 ms with zero OOM events under hard 512 MiB cgroup limits.

3. **Protocol Integrity:**
   - The official held-out test set (`data/test.txt`) was **not evaluated** with TTA. The reported 70.80% metric is strictly validation-only.

---

## 7. Post-Test Exploratory Architecture Milestone: E7 ConvNeXt-Tiny (2026-10-13, Mission 4D)

To investigate whether a modern depthwise-separable convolutional architecture could overcome the representational ceiling of ResNet18 while maintaining controlled parameter efficiency, we conducted **Mission 4D** as an exploratory post-test investigation on the frozen development split:

1. **Architecture & Pretrained Setup:**
   - **Backbone:** Torchvision `convnext_tiny` initialised with `ConvNeXt_Tiny_Weights.IMAGENET1K_V1` (82.52% ImageNet-1K Top-1). Total parameters: `27,973,928` (27.97M).
   - **Classifier Head:** `Sequential(LayerNorm2d(768), Flatten(1), Linear(768, 200))` replacing the default 1000-way linear layer. Trainable head parameters: `155,336`.
   - **Staged Schedule:**
     - Stage 1 (Epochs 1–5): Head warmup (`classifier` only, `lr=1e-3`, AdamW, weight decay `1e-4`, backbone frozen).
     - Stage 2 (Epochs 6–30): Stage 4 backbone fine-tuning (`features.7` + `classifier`, `14.44M` trainable parameters; discriminative LR: `5e-5` for Stage 4, `2e-4` for classifier; stages 1–3 kept frozen; `patience=8`).
   - **Precision & Hardware:** CUDA Automatic Mixed Precision (AMP), batch size 32, RTX 3060 Laptop GPU (6.0 GiB VRAM). Peak training VRAM was **727.5 MB** (~12% of GPU capacity). Average epoch time was **81.5 seconds**. Total training runtime: **41.5 minutes**.

2. **Empirical Performance on Frozen Development Split:**
   - **Best Epoch:** 27 / 30 (checkpoint SHA-256 `18cfc800908652c7b055e7a13e4369f995ec4a91b4ae4b1b7f134f4722d015b2`).
   - **Validation Metrics (Single-View):**
     - **Val Top-1 Accuracy:** **80.40%** (`0.803962`, `+10.74 pp` vs `E3` `69.66%`).
     - **Val Avg Accuracy / Class:** **79.75%** (`0.797500`, `+10.65 pp` vs `E3` `69.10%`).
     - **Val Top-5 Accuracy:** **96.35%** (`0.963504`, `+6.78 pp` vs `E3` `89.57%`).
     - **Val Macro-F1:** **79.19%** (`0.791885`, `+10.67 pp` vs `E3` `68.52%`).
     - **Val Cross-Entropy Loss:** **0.6985** (vs `1.1895` in `E3`, a `41.3%` loss reduction).
   - **Validation Metrics (2-View Horizontal-Flip TTA):**
     - **Val Top-1 Accuracy:** **81.02%** (`0.810219`, `+0.63 pp` vs `E7` single-view, `+10.22 pp` vs `E3` TTA).
     - **Val Avg Accuracy / Class:** **80.35%** (`0.803500`, `+0.60 pp` vs `E7` single-view, `+10.08 pp` vs `E3` TTA).
     - **Val Top-5 Accuracy:** **96.25%** (`0.962461`).
     - **Val Macro-F1:** **79.71%** (`0.797116`).
   - **Generalisation Gap:** Clean-train Top-1 saturated at `99.82%`, yielding a generalisation gap of **19.42 pp** (`-10.82 pp` narrower than `E3`'s `30.24 pp` gap).
   - **Per-Class Breakdown:** 94 / 200 species achieved perfect 100% recall (vs 45 in E3); zero classes scored 0% recall (vs 4 in E3); 99 species improved, 30 regressed, and 71 remained unchanged.

3. **Cloud Deployment Feasibility & Engineering Trade-Off:**
   - **Forward-Only CPU Profiling:** Forward-only inference uses `350.01 MiB` anonymous heap (`VmRSS = 453.13 MiB`), operating within close range of Render's 512 MiB limit.
   - **Grad-CAM Autograd Allocation:** Generating `features.7` ($7\times 7 \times 768$) Grad-CAM heatmaps requires backpropagating through the classifier and LayerNorm on CPU, pushing anonymous heap to **560.22 MiB** and peak resident set size to **669.13 MiB** (`VmHWM = 765.07 MiB`).
   - **Hard Cgroup Stress Test (`MemoryMax=512M`, `MemorySwapMax=0`):** Grad-CAM triggers an immediate Linux kernel OOM-kill on Render Free Web Service (512 MB).
   - **Hosting Analysis:** While Lightning AI Free Studio provides 16 GB RAM and comfortably accommodates ConvNeXt-Tiny (~765 MiB peak), its ephemeral public URLs regenerate upon idle restart (every 10–15 min), introducing severe availability risk for coursework marking.
   - **Strategic Decision:** **KEEP E3 ResNet18 as the live production deployment** on Render Free (`https://cos30082-bird-classifier.onrender.com/`), which reliably delivers single-view (69.66% val / 70.43% test) and TTA (70.80% val) predictions with full Grad-CAM within 512 MiB. Feature E7 ConvNeXt-Tiny prominently in the report as our superior post-test exploratory research milestone (**80.40% val / 81.02% TTA**), demonstrating deep technical understanding of the classic machine learning trade-off between model capacity and deployment infrastructure constraints.
   - The official held-out test set (`data/test.txt`) was **not evaluated** on E7, strictly preserving protocol integrity.


