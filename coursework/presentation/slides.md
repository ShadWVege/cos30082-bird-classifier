# COS30082 Assignment 1: Bird Species Classification
## 10-Minute Video Presentation Slide Deck

**Student Name:** Jonathan Zheng Li CHAI  
**Student ID:** 104400675  
**Unit:** COS30082 Applied Machine Learning  
**Faculty:** Faculty of Science, Computing and Technologies, Swinburne University of Technology  
**Primary Project Portal (HF Space):** [https://huggingface.co/spaces/ShadWVege/cos30082-bird-species-explorer](https://huggingface.co/spaces/ShadWVege/cos30082-bird-species-explorer)  
**Interactive Classifier (Lightning AI E7):** [https://7863-01m4vaa4swe6b7ha57xrh4ch7s.cloudspaces.litng.ai/](https://7863-01m4vaa4swe6b7ha57xrh4ch7s.cloudspaces.litng.ai/)  
**Backup Web Application (Render E3):** [https://cos30082-bird-classifier.onrender.com/](https://cos30082-bird-classifier.onrender.com/)  
**Deployment Repository (GitHub):** [https://github.com/ShadWVege/cos30082-bird-classifier](https://github.com/ShadWVege/cos30082-bird-classifier)  
**Presentation Video:** `https://youtu.be/... (YOUTUBE_VIDEO_URL_PENDING)`  
  

---

### Slide 1: Title & Project Overview (0:00 – 0:45)
- **Title:** Fine-Grained Bird Species Classification on CUB-200: Overcoming Memorization Ceilings via Staged Transfer Learning, Bilateral Test-Time Augmentation, and Memory-Constrained Cloud Deployment
- **Core Challenge:** Classifying 200 bird species from the Caltech-UCSD Birds 200 (CUB-200) dataset.
- **Why Fine-Grained Classification is Hard:**
  - High inter-class visual similarity (e.g., sister species of vireos, warblers, and gulls differ only by subtle eye-stripes or beak shapes).
  - High intra-class pose, lighting, and background variation (in-flight vs perched, juvenile vs breeding plumage).
  - Extreme data sparsity: Only 4,829 training images (~24 images per class) across 200 classes.
- **Key Project Goals:**
  - Evaluate from-scratch CNN architectures vs staged transfer learning.
  - Implement rigorous data audit protocols preventing data leakage.
  - Deploy a live, free interactive web application on cloud infrastructure under strict resource limits.

---

### Slide 2: Dataset Integrity Audit & Leakage Prevention (0:45 – 1:30)
- **Training Audit (4,829 Images):**
  - All 4,829 images in `train.txt` decoded cleanly as RGB; zero corrupt files; dimensions span 162×121 to 500×500 px.
  - **Decoded Pixel Hashing Discovery:** Three exact duplicate image pairs carry contradictory labels in the source data:
    1. Crested Auklet (4) vs Parakeet Auklet (6)
    2. Philadelphia Vireo (152) vs Warbling Vireo (155)
    3. Red-eyed Vireo (153) vs Tennessee Warbler (178)
- **Policy Decision (`conflict_policy: train_only`):**
  - Pin all 6 conflicting duplicate images to the training fold.
  - Eliminates exact-pixel leakage into validation while transparently acknowledging 0.16% source label noise.
- **Stratified Group Split:**
  - Seed 42, 3,870 training images (80.1%) / 959 validation images (19.9%).
  - Group-aware: guarantees duplicate hashes never cross folds.
- **Held-Out Test Set Audit (1,204 Images):**
  - Four exact cross-split pixel duplicates identified (including `Test/0197.jpg`, a garden feeder photo showing both Purple Finch and American Goldfinch). Full 1,204 test denominator preserved without data tampering.

---

### Slide 3: Training from Scratch: Baseline Memorization (E1) (1:30 – 2:15)
- **E1 Architecture:**
  - 4-stage convolutional hierarchy ($3 \to 32 \to 64 \to 128 \to 256$ channels), $3\times 3$ convs, ReLU, $2\times 2$ MaxPool.
  - Global Average Pooling (GAP) $\to$ `Linear(256, 200)`. Total parameters: 439,816.
  - Optimizer: Adam ($10^{-3}$), standard Cross-Entropy Loss, 40 epochs.
- **Observed Empirical Trajectory:**
  - Training accuracy rapidly climbed to **93.05%** at epoch 40 (`clean_train_loss` = 0.7679).
  - Validation accuracy peaked early at **21.69%** (Epoch 31) and stagnated.
  - Validation loss exploded from 4.70 to **10.30**.
- **Diagnostic Finding: Catastrophic Memorization:**
  - Enormous **71.99 pp generalization gap** (58.49 pp at best epoch).
  - With ~19 images/class and no pre-trained priors, the 440K parameters memorized background artifacts (foliage, branches) rather than anatomical features.

---

### Slide 4: Regularization Attempts & The Capacity Starvation Trap (E2 vs E4) (2:15 – 3:00)
- **E2 Regularised CNN (Dropout 0.4):**
  - Added 2D Batch Normalization after each conv block, Dropout ($p=0.4$) before classifier, AdamW ($3\times 10^{-4}$, weight decay $10^{-4}$), and multi-modal data augmentations (crops, flips, rotations, jitter).
  - Result: Generalization gap collapsed from 58.49 pp to **6.19 pp**.
  - But validation Top-1 plunged to **10.43%**, and Average Accuracy per Class fell to **9.63%**!
- **Controlled Causal Ablation (E4: Dropout 0.0):**
  - Isolated the exact effect of Dropout ($p=0.0$ vs $p=0.4$ in E2), keeping BatchNorm, augmentations, and optimizer identical.
  - Result: Validation Top-1 jumped to **14.39%** (+3.96 pp) and Average Accuracy per Class rose to **13.93%** (+4.30 pp).
- **Core Engineering Takeaway:**
  - Severe **Capacity Starvation**: Zeroing 40% of feature activations in a compact 256-channel embedding prevented gradient propagation across fine-grained sub-species distinctions under heavy data augmentation.

---

### Slide 5: The Transfer Learning Breakthrough: Pretrained ResNet18 (E3) (3:00 – 4:15)
- **E3 Architecture & Staged Schedule:**
  - Torchvision ImageNet-1K pretrained ResNet18 (11,279,112 parameters).
  - Replaced 1000-way head with `Linear(512, 200)`.
  - **Stage 1 (Head Warmup, 5 epochs):** Frozen backbone; optimize head with AdamW ($10^{-3}$, weight decay $10^{-4}$). Backbone BatchNorm locked in `eval()` mode. Reached 53.60% validation Top-1 in 5 epochs!
  - **Stage 2 (Partial Backbone Fine-Tuning, 30 epochs):** Unfreeze `layer4` (residual blocks 4.0 and 4.1) and `fc` with AdamW ($10^{-4}$).
- **Validation Results (Selected Model):**
  - **Validation Top-1:** **69.66%** (+47.97 pp over E1).
  - **Validation Average Accuracy per Class:** **69.10%** (+47.81 pp over E1).
  - **Validation Top-5 Accuracy:** **89.57%**.
  - **Validation Loss:** **1.1895** (vs 6.68 in E1).
- **Official Held-Out Test Evaluation (Single-View, Evaluated ONCE on Frozen Model):**
  - **Single-View Test Top-1 Accuracy:** **70.43%** (848/1,204).
  - **Test Average Accuracy per Class:** **70.12%**.
  - **Test Top-5 Accuracy:** **89.12%** (1,073/1,204).
- **Validation-to-Test Consistency:** Agreement within 0.77 pp confirms strong alignment between validation (69.66%) and held-out test (70.43%) performance on our frozen split, while the 30.24 pp clean-train gap quantifies residual capacity.

---

### Slide 6: Codebase Architecture & Design Choices (4:15 – 5:45)
- **Modular Pipeline Structure (`src/`):**
  - `src/data.py`: Manifest parsing, path traversal sanitization, and grouped stratified fold assignment preserving duplicate hashes.
  - `src/dataset.py`: Shared transform pipelines (deterministic 256-resize + 224-crop) across validation, test, and inference.
  - `src/models.py`: Modular model factory supporting CustomCNN (439,816 E1 / 440,296 E2/E4 params), ResNet18 (11.28M), and ConvNeXt-Tiny (27.97M).
  - `src/engine.py` & `src/train.py`: Unified training loop with Automatic Mixed Precision (AMP), gradient scaling, `ReduceLROnPlateau` resets between stages, and early stopping.
  - `src/metrics.py`: Computes Top-1 Accuracy, Average Accuracy per Class ($C=200$ macro recall over all classes), Top-5, and Macro-F1.
  - `src/inference.py`: Production inference engine supporting memory-mapped weights (`mmap=True`), `meta` device allocation, sequential 2-view TTA, and autograd Grad-CAM hooks.
- **Reproducibility & Verification:**
  - Deterministic seeds, Windows DataLoader 0 workers safeguard, comprehensive automated pytest test suite (26+ passed).

---

### Slide 7: Advanced Post-Test Explorations (E5, E6, & Bilateral TTA) (5:45 – 7:00)
- **E5 Label Smoothing ($\alpha=0.05$):**
  - Slashed wrong-prediction overconfidence by **22.55 pp** (31.31% vs 53.86% in E3) and elevated clean-train loss floor ($0.0470$ vs $0.0060$).
  - However, uniform non-target smoothing slightly compressed margins between visually similar sibling species, lowering validation Top-1 to **68.72%** (-0.94 pp).
- **E6 Enhanced ResNet18 Fine-Tuning (`layer3 + layer4 + fc`):**
  - Unfreezing `layer3` with discriminative learning rates ($10^{-5}, 5\times 10^{-5}, 2\times 10^{-4}$) failed to beat E3 (**68.40%** val Top-1, val loss 2.0790, ECE 0.2021) due to representation drift on small sample sizes.
- **Inference-Only Two-View Bilateral Reflection TTA:**
  - Birds are bilaterally symmetric: average softmax probabilities of original and horizontal-flip orientations: $p_{\text{TTA}} = 0.5 \cdot (p_{\text{orig}} + p_{\text{flip}})$.
  - Boosts E3 validation Top-1 to **70.80%** (+1.14 pp) and Average Accuracy per Class to **70.27%** (+1.17 pp) at zero training cost!
  - Implemented sequentially ($1+1$) to avoid increasing peak GPU/CPU RAM.

---

### Slide 8: The Modern Convolutions Milestone: E7 ConvNeXt-Tiny (7:00 – 7:45)
- **Architecture & Setup:**
  - Pretrained ImageNet-1K ConvNeXt-Tiny (27.97M parameters), 200-class head.
  - Staged schedule: 5 head warmup epochs + 25 Stage 4 (`features.7` + classifier, 14.44M params) fine-tuning epochs with discriminative LR ($5\times 10^{-5} / 2\times 10^{-4}$) and AMP.
- **Remarkable Empirical Results (Validation vs Official Test):**
  - **Validation Top-1:** **80.40%** (+10.74 pp over E3; **81.02%** with 2-view TTA).
  - **Average Accuracy per Class:** **79.75%** (+10.65 pp over E3; **80.35%** with 2-view TTA).
  - **Validation Top-5:** **96.35%** (+6.78 pp over E3); Loss: **0.6985** (-41.3%).
  - **Official Held-Out Test Evaluation (Single-View Milestone):**
    - **Single-View Test Top-1:** **80.73%** (972/1,204, +10.30 pp over E3).
    - **Test Average Accuracy per Class:** **80.24%** (+10.12 pp over E3).
    - **Test Top-5 Accuracy:** **95.18%** (1,146/1,204).
    - **Paired McNemar Significance Test:** $p = 2.48 \times 10^{-15}$ ($\chi^2 = 60.04$, $N=1,204$).
    - *Test TTA:* 80.32% Top-1 (-0.41 pp), demonstrating single-view superiority.
  - **Generalization Gap:** Saturated at **19.42 pp** (-10.82 pp narrower than E3's 30.24 pp gap).
  - **Per-Class Recall:** 94 out of 200 species achieved 100% recall (vs 45 in E3); zero classes scored 0% recall (vs 4 in E3).
- **Architectural Rationale:**
  - $7\times 7$ depthwise convolutions capture macro-spatial bird proportions; inverted bottleneck preserves channel richness; LayerNorm and GELU ensure smooth gradient propagation.

---

### Slide 9: Model Explainability (Grad-CAM) & Qualitative Failure Modes (7:45 – 8:30)
- **Focused Grad-CAM Attribution on Key Diagnostic Regions (E3 ResNet18 vs E7 ConvNeXt-Tiny):**
  - **E3 ResNet18 (`layer4`):** Saliency concentrates on diagnostic visual landmarks: crown, eye stripe, throat plumage, and mantle, suppressing background bark and foliage.
  - **E7 ConvNeXt-Tiny (`features.7`):** Tighter boundary adherence along complete avian contours and localized plumage patches via 7x7 depthwise receptive fields.
  - *Interpretability note:* Heatmaps indicate spatial correlation with target class logits, not proof of human-like biological reasoning.
- **Systematic Qualitative Error Cases:**
  - **Sister Species Ambiguity:** Philadelphia Vireo vs Warbling Vireo (identical yellow wash and white supercilium); Glaucous-winged Gull vs Western Gull (subtle mantle gray gradients).
  - **Multi-Species Photos:** `Test/0197.jpg` contains both a Purple Finch and an American Goldfinch at a feeder. A closed-set single-label classifier is mathematically forced to misclassify one species!
  - **Camouflage & Foliage Occlusion:** Marsh birds against dense reeds trigger misclassifications when reed textures dominate the crop.

---

### Slide 10: Multi-Tier Cloud Deployment Architecture & Trade-offs (8:30 – 9:30)
- **Coursework Cloud Ecosystem:**
  1. **Primary Project Portal (Hugging Face Space):** [https://huggingface.co/spaces/ShadWVege/cos30082-bird-species-explorer](https://huggingface.co/spaces/ShadWVege/cos30082-bird-species-explorer)
     - Central landing page, documentation, and architecture diagrams.
     - *Important Disclosure:* Static HTML only; PyTorch CNN inference does not execute directly on HF due to free-tier hardware limits.
  2. **Interactive High-Accuracy Classifier (Lightning AI Free CPU Studio):** [https://7863-01m4vaa4swe6b7ha57xrh4ch7s.cloudspaces.litng.ai/](https://7863-01m4vaa4swe6b7ha57xrh4ch7s.cloudspaces.litng.ai/)
     - Live **E7 ConvNeXt-Tiny** application (80.73% test Top-1, 27.97M parameters, terminal `features.7` Grad-CAM).
     - Running on 4 vCPUs / 16 GB RAM via Gradio plugin on port 7863.
     - Auto-sleep on idle to conserve free compute; wakes on request in ~71 seconds (~1–2 min cold start); operates within 4-hour active session limits.
  3. **Pre-Committed Production Backup (Render Free Web Service):** [https://cos30082-bird-classifier.onrender.com/](https://cos30082-bird-classifier.onrender.com/)
     - Pre-committed **E3 ResNet18** model (70.43% single-view official test Top-1, 11.28M parameters; optional 2-view TTA toggle).
     - Standalone deployment repository: [https://github.com/ShadWVege/cos30082-bird-classifier](https://github.com/ShadWVege/cos30082-bird-classifier).
- **Render Free Web Service Challenge (512 MiB Hard RAM Limit):**
  - Standard PyTorch loads >1.2 GB RAM (immediate OOM-kill).
- **Four Verified Memory Optimizations:**
  1. CPU-only PyTorch wheels (shed 700 MB disk & memory overhead).
  2. Memory-mapped checkpoint loading (`torch.load(..., mmap=True, weights_only=True)`).
  3. Meta-device initialization (`torch.device("meta")` + `assign=True`).
  4. Sequential TTA ($1+1$) and thread budgeting (`OMP_NUM_THREADS=1`).
- **Empirical Memory Benchmark & Trade-offs:**
  - E3 ResNet18 + TTA uses only **289 MiB RSS** (Peak HWM 380 MiB), safely below Render's 512 MiB limit with 131 MiB genuine physical margin. Latency: ~101 ms (TTA), ~215 ms (Grad-CAM).
  - While terminal feature map isolation prevents OOM kills in ConvNeXt-Tiny, container memory reaches the 512 MiB ceiling (`VmHWM` 501.0 MiB, 1,366 pressure events), making E3 the safer choice for unmonitored production on Render Free.
- **Architectural Harmony:**
  - Lightning AI hosts our flagship 80.73% E7 model with ample memory for deep explainability, Hugging Face provides the primary discoverable project portal, and Render Free guarantees a stable, memory-safe fallback. Alternative hosting arrangements subject to instructor confirmation.


