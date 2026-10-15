# COS30082 Bird Classification — Student Source Code Walkthrough & Defense Guide

**Project:** Fine-Grained Bird Species Classification on CUB-200  
**Author / Candidate:** COS30082 Student  
**Target Video Runtime:** < 10 Minutes  
**Primary Deliverables:** Python Source ZIP, 4-Page PDF Report, Presentation Video, Cloud Application  

---

## 1. Quick Codebase Architecture Map

| Logical Component | Primary File & Line References | Key Class / Function | Architectural Purpose |
|---|---|---|---|
| **Data Ingestion & Manifests** | [`src/data.py:17-50`](../src/data.py#L17-L50) | `read_manifest()` | Sanitizes paths, enforces zero-based labels (0–199), rejects path traversal attacks. |
| **Species Vocabulary Extraction** | [`src/data.py:53-73`](../src/data.py#L53-L73) | `infer_class_names()` | Regex parses structured training filenames to extract 200 unique human-readable species names. |
| **Data Auditing & Pixel Hashing** | [`src/data.py:95-131`](../src/data.py#L95-L131) | `audit_images()` | Hashes decoded RGB pixel arrays with SHA-256 to uncover metadata-only image duplicates. |
| **Grouped Stratified Splitting** | [`src/data.py:134-180`](../src/data.py#L134-L180) | `stratified_group_split()` | Subset-sum search keeping pixel duplicates in the same fold; pins 6 conflicting images to train. |
| **Image Preprocessing & Dataset** | [`src/dataset.py:21-72`](../src/dataset.py#L21-L72) | `image_transform()`, `BirdDataset` | ImageNet normalization; deterministic 256-resize + 224-crop; EXIF orientation correction. |
| **CNN Architecture Definitions** | [`src/models.py:13-98`](../src/models.py#L13-L98) | `CustomCNN`, `build_model()` | 4-stage CustomCNN (E1/E2/E4); ResNet-18 (E3/E5/E6); ConvNeXt-Tiny (E7). |
| **Transfer Staging & BN Controls** | [`src/models.py:101-168`](../src/models.py#L101-L168) | `set_stage()`, `training_mode()` | Stage 1 head warmup vs Stage 2 partial fine-tuning; controls BatchNorm vs LayerNorm dynamics. |
| **DataLoader Construction** | [`src/engine.py:23-45`](../src/engine.py#L23-L45) | `make_loader()` | Seeds PyTorch Generator & worker_init_fn; sets num_workers=0 on Windows. |
| **Training & Evaluation Epoch** | [`src/engine.py:48-120`](../src/engine.py#L48-L120) | `run_epoch()` | AMP fp16 autocast; grad scaler; gradient clipping (5.0); label smoothing toggle. |
| **Atomic Checkpointing** | [`src/engine.py:145-167`](../src/engine.py#L145-L167) | `save_checkpoint()`, `load_checkpoint()` | Atomic write (.tmp -> rename); weights_only=True; Linux mmap=True. |
| **Evaluation Metrics Contract** | [`src/metrics.py:20-80`](../src/metrics.py#L20-L80) | `classification_metrics()` | Vectorized C x C confusion matrix; Top-1; Average Accuracy per Class; Top-5; Macro-F1. |
| **Training Pipeline & Gates** | [`src/train.py:41-175`](../src/train.py#L41-L175) | `train()` | 3 dataloaders (train/clean_train/val); discriminative LRs; ReduceLROnPlateau; early stopping. |
| **Scientific Model Comparison** | [`src/compare.py:9-39`](../src/compare.py#L9-L39) | `select_runs()` | Freezes winning model based on val Avg/Class before any test evaluation. |
| **Official Held-Out Evaluation** | [`src/evaluate.py:26-58`](../src/evaluate.py#L26-L58) | `evaluate()` | Enforces pre-commitment check; audits test integrity; runs zero-tuning evaluation. |
| **Inference & Grad-CAM Engine** | [`src/inference.py:53-270`](../src/inference.py#L53-L270) | `Predictor` | Meta-device loading; terminal feature map isolation; bilateral TTA; NumPy jet colormap. |
| **Production Web Application** | [`app.py:36-345`](../app.py#L36-L345) | `classify()`, `build_app()` | Gradio UI; single-view / TTA toggle; Grad-CAM panel; input validation guards. |

---

## 2. Walkthrough by Topic (With Viva Defense Q&A)

### Topic 1: Which file loads the dataset?
- **Module:** [`src/dataset.py`](../src/dataset.py) (`BirdDataset`, lines 46–72) and [`src/engine.py`](../src/engine.py) (`make_loader`, lines 23–45).
- **Implementation Detail:** `BirdDataset.__getitem__` opens the image via PIL, applies `ImageOps.exif_transpose` to correct camera rotation metadata, converts to 3-channel RGB, and runs the torchvision `image_transform`.
- **Oral Defense Q&A:**
  - *Q: Why do you need `ImageOps.exif_transpose`?*
  - *A:* Digital cameras and smartphones store orientation in EXIF metadata without rotating raw pixels. Without `exif_transpose`, upright photographs could be fed into the CNN sideways or upside down, corrupting learned spatial features and Grad-CAM heatmaps.

---

### Topic 2: How are the 200 classes represented?
- **Module:** [`src/data.py`](../src/data.py) (`read_manifest`, lines 26–50; `infer_class_names`, lines 53–73) and [`src/inference.py`](../src/inference.py) (lines 70–73).
- **Implementation Detail:** The dataset manifests (`data/train.txt` and `data/test.txt`) use space-delimited `<path> <label>` pairs with integer indices from 0 to 199. `infer_class_names` parses the species prefix from training filenames (e.g., `Red_headed_Woodpecker_0012_2578571305.jpg` -> `Red headed Woodpecker`) and verifies a bijective 1-to-1 mapping for all 200 classes.
- **Oral Defense Q&A:**
  - *Q: Why can't you infer species names from `test.txt`?*
  - *A:* `test.txt` uses anonymized numeric filenames (`0001.jpg` to `1204.jpg`) which contain zero species strings. The class names must be learned exclusively from the training set and stored in `class_names.json`.

---

### Topic 3: How is the CNN constructed?
- **Module:** [`src/models.py`](../src/models.py) (`CustomCNN`, lines 13–48; `build_model`, lines 51–98).
- **Implementation Detail:**
  - **CustomCNN (E1/E2/E4):** 4 Conv-ReLU-Pool stages doubling channels $32 \rightarrow 64 \rightarrow 128 \rightarrow 256$, followed by `AdaptiveAvgPool2d(1)` and a linear classifier. In E2/E4, `BatchNorm2d` and `Dropout` (0.4) are toggled.
  - **ResNet-18 (E3):** ImageNet-1K pretrained backbone. The default 1,000-way linear layer (`model.fc`) is replaced with `nn.Linear(512, 200)` (11,279,112 parameters).
  - **ConvNeXt-Tiny (E7):** Modern depthwise-separable architecture. `model.classifier[2]` is replaced with `nn.Linear(768, 200)` while preserving `model.classifier[0]` (`LayerNorm2d(768)`) and `model.classifier[1]` (`Flatten(1)`), totalling 27,973,928 parameters.
- **Parameter Count Verification & Arithmetic:**
  - **Baseline E1 (439,816 params):** 387,936 conv weights + 480 conv biases (32+64+128+256) + 51,400 classifier = 439,816.
  - **Regularized E2 & Ablation E4 (440,296 params):** 387,936 conv weights + 0 conv biases + 960 BatchNorm (480 $\gamma$ + 480 $\beta$) + 51,400 classifier = 440,296.
  - **Reconciliation Note:** Early coursework drafts cited 440,776 by adding 960 BatchNorm parameters directly to E1 without subtracting the 480 omitted conv biases ($439,816 + 960 = 440,776$). The actual trained and saved checkpoints strictly instantiate 440,296 parameters.
- **Oral Defense Q&A:**
  - *Q: Why did you omit Conv bias when using BatchNorm in CustomCNN?*
  - *A:* Because BatchNorm applies a learnable affine shift $\beta$ after normalizing channel mean $\mu$. Mathematically, $\text{BN}(W x + b) = \text{BN}(W x)$, making the convolutional bias parameter redundant and wasteful.

---

### Topic 4: Where do Stage 1 and Stage 2 training occur?
- **Module:** [`src/train.py`](../src/train.py) (lines 108–133) and [`src/models.py`](../src/models.py) (`set_stage`, lines 101–125; `training_mode`, lines 128–168).
- **Implementation Detail:**
  - **Stage 1 ('head', Epochs 1–5):** `set_stage` sets `requires_grad = False` across the entire backbone. Only the 200-class linear head trains. This prevents large random gradients from destroying delicate ImageNet representations.
  - **Reload Transition Gate (Line 113–115):** Before beginning Stage 2, the model reloads `checkpoints/best.pt` from Stage 1.
  - **Stage 2 ('finetune', Epochs 6–35):** Unfreezes terminal convolutional blocks (`layer4` for ResNet-18, `features.7` for ConvNeXt-Tiny) alongside the classifier using discriminative learning rates.
- **Oral Defense Q&A:**
  - *Q: Why do you reload the best Stage 1 checkpoint before fine-tuning?*
  - *A:* Stage 1 head warmup fluctuates; the final epoch of warmup is rarely the optimal classifier state. Starting fine-tuning from the highest-scoring warmup checkpoint guarantees the cleanest foundation for backbone adaptation.

---

### Topic 5: How is overfitting addressed?
- **Implementation Modules:**
  1. **Data Augmentation:** [`src/dataset.py:25-30`](../src/dataset.py#L25-L30) (`RandomResizedCrop(scale=(0.7, 1.0))`, `RandomHorizontalFlip()`, `RandomRotation(10)`, `ColorJitter`).
  2. **Architectural Regularization:** `BatchNorm2d` and `Dropout(0.4)` in [`src/models.py:13-33`](../src/models.py#L13-L33).
  3. **Optimizer Weight Decay:** AdamW with $L_2$ regularization $\lambda = 10^{-4}$ in [`src/train.py:126`](../src/train.py#L126).
  4. **Staged Partial Freezing:** Keeping stages 1–3 frozen to prevent over-parameterization on ~19 images/class ([`src/models.py:101-125`](../src/models.py#L101-L125)).
  5. **Early Stopping:** Monitors validation score with `patience = 8` epochs ([`src/train.py:175`](../src/train.py#L175)).
  6. **Clean Train DataLoader:** Evaluates training data deterministically without augmentations to measure the true generalization gap: $\text{generalisation\_gap} = \text{clean\_train\_top1} - \text{val\_top1}$ ([`src/train.py:173`](../src/train.py#L173)).
- **Oral Defense Q&A:**
  - *Q: Why do you need `clean_train` to measure the generalization gap?*
  - *A:* Augmented training accuracy fluctuates because images are cropped and distorted on the fly. Running deterministic evaluation transforms on the training set isolates true model fit from augmentation artifacts, yielding a mathematically pure generalization gap.

---

### Topic 6: How is the best checkpoint selected?
- **Module:** [`src/train.py`](../src/train.py) (lines 155–164) and [`src/compare.py`](../src/compare.py) (lines 31–38).
- **Implementation Detail:**
  - Primary metric: Validation **Average Accuracy per Class** (macro recall over all 200 species).
  - Tie-breaker: Validation cross-entropy loss (`val_loss`).
  - Checkpoint saving: `save_checkpoint(checkpoints / "best.pt", checkpoint)` saves only when a new validation record is established.
- **Oral Defense Q&A:**
  - *Q: Why use Average Accuracy per Class instead of Top-1 for model selection?*
  - *A:* CUB-200 has class imbalances (16 to 31 images per species). Macro recall gives every species equal weight (1/200th), preventing the model selector from favoring models that excel on over-represented classes while failing on rare classes.

---

### Topic 7: How are Top-1 and Average Accuracy per Class calculated?
- **Module:** [`src/metrics.py`](../src/metrics.py) (`classification_metrics`, lines 20–80).
- **Mathematical Formulations:**
  - **Top-1 Accuracy:** $\text{Top-1} = \frac{\sum_i \mathbb{I}(y_i = \hat{y}_i)}{N} = \frac{\text{Trace}(C)}{N}$.
  - **Average Accuracy per Class (Macro Recall):**
    $$\text{AvgAcc} = \frac{1}{200}\sum_{c=0}^{199} \text{Recall}_c = \frac{1}{200}\sum_{c=0}^{199} \frac{C_{c,c}}{\sum_{j=0}^{199} C_{c,j}}$$
- **Oral Defense Q&A:**
  - *Q: How does Average Accuracy per Class differ from binary one-vs-rest accuracy?*
  - *A:* In a 200-class problem, binary one-vs-rest accuracy $\frac{\text{TP}+\text{TN}}{\text{Total}}$ is trivially 99.5% for an always-negative baseline (since true negatives comprise $\frac{199}{200} = 99.5\%$ of samples in a balanced dataset). The coursework rubric explicitly defines Average Accuracy per Class as unweighted macro recall across all classes.

---

### Topic 8: Why did E7 ConvNeXt-Tiny outperform E3 ResNet18?
- **Architectural & Representational Differences:**
  1. **Depthwise-Separable Convolutions:** $7 \times 7$ depthwise kernels capture broad contextual shape and proportions (e.g. crests, bill-to-head ratio) without quadratic parameter explosion.
  2. **Inverted Bottleneck Design:** Expands channel width $4\times$ inside the block before projecting down, preserving fine plumage textures through non-linearities.
  3. **LayerNorm vs BatchNorm:** Computes statistics across channel dimensions per sample independently, eliminating batch-statistic noise on small batch sizes.
  4. **Empirical Superiority:** Achieved **80.73% official test Top-1** vs **70.43%** for E3 (+10.30 pp gain; paired McNemar test $p = 2.48 \times 10^{-15}$).

---

### Topic 9: How does the Gradio application load the model?
- **Module:** [`app.py`](../app.py) (lines 36–45) and [`src/inference.py`](../src/inference.py) (lines 53–90).
- **Implementation Detail:**
  1. `predictor()` is wrapped in `@lru_cache(maxsize=1)` for lazy singleton initialization.
  2. Threading is limited: `torch.set_num_threads(1)` prevents cloud host thread-arena allocation.
  3. Memory-mapped checkpoint: `load_checkpoint(mmap=True)` keeps weights backed by Linux page cache (`RssFile`).
  4. Zero-allocation initialization: Instantiates architecture on `torch.device("meta")`, then executes `model.load_state_dict(checkpoint["state_dict"], assign=True)`.
  5. Gradient disablement: `model.requires_grad_(False)` suppresses gradient buffer allocations.

---

### Topic 10: How does Grad-CAM work?
- **Module:** [`src/inference.py`](../src/inference.py) (`predict_with_gradcam`, lines 188–255).
- **Mathematical Formulations:**
  1. Target layer: Hooks terminal feature stage (`layer4` in ResNet-18, `features.7` in ConvNeXt).
  2. Forward Hook Isolation: `out = out.detach().requires_grad_(True); out.retain_grad()` isolates gradients exclusively to the $7 \times 7$ feature volume, saving >90% memory.
  3. Importance weights: $\alpha_k^c = \frac{1}{Z}\sum_i \sum_j \frac{\partial y^c}{\partial A_{i,j}^k}$.
  4. Weighted activation: $L_{\text{Grad-CAM}}^c = \text{ReLU}\left(\sum_k \alpha_k^c A^k\right)$.
  5. Bilinear upsampling to $224 \times 224$ and overlay on deterministic CenterCrop.

---

### Topic 11: Why is E7 single-view the default mode?
- **Module:** [`deploy/e7_lightning/app.py`](../deploy/e7_lightning/app.py) (lines 149–165).
- **Scientific Rationale:**
  - On the official 1,204-image held-out test set, single-view achieved **80.73% Top-1**, whereas 2-view TTA achieved **80.32% Top-1** (-0.41 pp regression).
  - Single-view is $2\times$ faster (52 ms vs 101 ms CPU latency).
  - In fine-grained recognition, bilateral averaging can wash out subtle asymmetric plumage patterns. Respecting empirical test evidence, single-view is default.

---

### Topic 12: Which known limitations must be disclosed?
- **Four Explicit Disclosures:**
  1. **Dataset Metadata Conflicts:** 3 duplicate pairs in training (0.16% label noise); 4 cross-split duplicate pairs in test (0.33%), including multi-species feeder photo `Test/0197.jpg`. Full 1,204 test denominator preserved without data tampering.
  2. **Closed-Set Assumption:** Softmax forces non-avian inputs into one of 200 bird species; requires out-of-distribution rejection for open-world deployment.
  3. **Cloud Infrastructure Boundaries:** Render Free spins down after 15 minutes of inactivity; Lightning AI Free Studio operates under 4-hour runtime limits before manual restart.
  4. **Generalization Gap:** E7 clean-train Top-1 reaches 99.82%, leaving a 19.42 pp generalization gap despite high 80.73% test accuracy.

---

### Topic 13: How are model weights reproduced or obtained?
- **Source Submission Archive Policy:**
  - As per the assignment brief ("Python program source code: Create a single zip file with your code"), `outputs/source_submission.zip` intentionally excludes multi-hundred-megabyte `.pt` binaries (~112 MB for ConvNeXt-Tiny, ~45 MB for ResNet18) to avoid LMS upload rejections.
- **Reproduction & Access:**
  1. **Deterministic Training:** `python -m src.train --config configs/train_convnext_tiny.json` reproduces the exact E7 checkpoint from the frozen seed-42 split.
  2. **Standalone Deployment Archive:** Verified deployment weights (`model.pt`, SHA-256 `b26dce1d5f0883d9612fcc6905c0d0e5f632accc54d7d9df0dd511c9cd208d4c`) are packaged in `outputs/e7_lightning_deploy_v2.zip`.
  3. **Live Deployed Services:** Pretrained weights are hosted live on Render Free (`https://cos30082-bird-classifier.onrender.com/`) and Lightning AI Studio.

