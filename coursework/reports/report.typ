// ==============================================================================
// COS30082 Machine Learning — Assignment 1: Bird Species Classification
// Four-Page Comprehensive Academic Research Report (Revised Mission 4E + 5A)
// Authoritative Benchmark: Caltech-UCSD Birds 200 (CUB-200)
// ==============================================================================

#set page(
  paper: "a4",
  margin: (x: 1.25cm, top: 1.25cm, bottom: 1.25cm),
  header: align(right)[
    #text(size: 7.2pt, fill: luma(100), font: "Liberation Sans")[
      COS30082 Machine Learning | Individual Assignment 1: Bird Species Classification
    ]
  ],
  footer: context align(center)[
    #text(size: 7.2pt, fill: luma(100), font: "Liberation Sans")[
      Page #counter(page).display("1 of 1", both: true)
    ]
  ]
)

#set text(
  font: "Liberation Serif",
  size: 8.2pt,
  lang: "en",
  spacing: 106%
)

#set par(
  justify: true,
  leading: 0.48em,
  first-line-indent: 0em
)

#show heading: set text(fill: rgb("#111827"), font: "Liberation Sans")
#show heading.where(level: 1): it => block(
  above: 0.75em, below: 0.4em,
  text(size: 10.2pt, weight: "bold", fill: rgb("#0f3460"))[
    #upper(it.body)
  ]
)
#show heading.where(level: 2): it => block(
  above: 0.55em, below: 0.3em,
  text(size: 8.6pt, weight: "bold", fill: rgb("#16213e"))[
    #it.body
  ]
)

// Document Title & Metadata (Full Page Width)
#align(center)[
  #block(width: 100%, stroke: none, inset: (bottom: 0.3em))[
    #text(size: 12.8pt, weight: "bold", fill: rgb("#0f3460"))[
      Fine-Grained Bird Species Classification on CUB-200: Staged Transfer Learning, Test-Time Augmentation, and Resource-Constrained Deployment
    ]
    
    #v(0.25em)
    #text(size: 8.8pt, weight: "medium")[
      *Student Name:* Jonathan Zheng Li CHAI #h(1.5em) | #h(1.5em) *Student ID:* 104400675
    ]
    
    #text(size: 7.8pt, fill: luma(80))[
      Faculty of Science, Computing and Technologies, Swinburne University of Technology #h(1em) | #h(1em) COS30082 Applied Machine Learning
    ]
    
    #v(0.35em)
    #rect(
      width: 100%,
      fill: rgb("#f8fafc"),
      stroke: 0.6pt + rgb("#cbd5e1"),
      radius: 3pt,
      inset: (x: 6pt, y: 3.5pt)
    )[
      #grid(
        columns: (1.15fr, 1.25fr, 1.15fr, 1.05fr),
        gutter: 4pt,
        align: center,
        [#text(size: 6.6pt, weight: "bold")[Primary Project (HF Space):] \ #link("https://huggingface.co/spaces/ShadWVege/cos30082-bird-species-explorer")[#text(size: 6.2pt, fill: rgb("#1d4ed8"))[cos30082-bird-species-explorer]]],
        [#text(size: 6.6pt, weight: "bold")[Interactive Classifier (Lightning E7):] \ #link("https://7863-01m4vaa4swe6b7ha57xrh4ch7s.cloudspaces.litng.ai/")[#text(size: 6.2pt, fill: rgb("#1d4ed8"))[7863-...litng.ai (Live Demo)]]],
        [#text(size: 6.6pt, weight: "bold")[Backup Application (Render E3):] \ #link("https://cos30082-bird-classifier.onrender.com/")[#text(size: 6.2pt, fill: rgb("#1d4ed8"))[cos30082-bird-classifier (Backup)]]],
        [#text(size: 6.6pt, weight: "bold")[Presentation Video (10 min):] \ #text(size: 5.9pt, fill: rgb("#b45309"))[youtu.be/... (YOUTUBE_VIDEO_URL_PENDING)]]
      )
      #v(0.12em)
      #text(size: 5.8pt, fill: luma(100))[
        \*Primary E7 live demo served on Lightning AI Free CPU Studio (port 7863, auto-wake on visit ~71s, auto-sleep on idle); Hugging Face Space hosts project landing page and documentation only (CNN inference does not run on HF). Render Free serves pre-committed E3 backup (512 MiB limit). Alternative hosting requires instructor approval.
      ]
    ]
  ]
]

#v(0.15em)
#block(
  width: 100%,
  fill: rgb("#f1f5f9"),
  stroke: (left: 2.5pt + rgb("#0f3460")),
  inset: (x: 7pt, y: 4.5pt)
)[
  #text(size: 7.6pt, weight: "bold", fill: rgb("#0f3460"))[Abstract—]
  #text(size: 7.4pt)[
    Fine-grained visual categorization on Caltech-UCSD Birds 200 (CUB-200; 200 classes, 4,829 training images) [1] poses substantial machine learning challenges due to subtle inter-species morphological differences, large intra-species pose variations, and sample sparsity (~24 images/class). In this work, we systematically investigate the progression from custom convolutional networks trained from scratch to staged transfer learning, inference test-time augmentation (TTA), and memory-constrained cloud deployment. An initial dataset audit verified 4,829 training images, uncovered three conflicting duplicate pairs pinned to training to eliminate validation leakage, and identified four cross-split test duplicates preserved under benchmark protocol. We show that a 4-block custom CNN (E1) collapses into severe memorization (58.49 pp best-epoch gap; 71.99 pp final gap). Regularization (E2: BatchNorm, Dropout 0.4) reduces the gap to 6.19 pp but starves capacity (9.63% average class accuracy), while an ablation (E4: Dropout 0.0) restores capacity to 13.93% (+4.30 pp). Staged transfer learning using ResNet18 [2] (E3: head warmup followed by partial fine-tuning) overcomes this bottleneck, achieving 69.66% validation Top-1, 69.10% Average Accuracy per Class, and 70.43% on the official held-out test set (70.12% avg/class). Post-test investigations reveal that label smoothing (E5) reduces incorrect prediction confidence by 22.55 pp, bilateral reflection TTA boosts validation Top-1 to 70.80% (+1.14 pp), and ConvNeXt-Tiny [3] (E7) establishes an exploratory milestone of 80.40% validation Top-1 and 80.73% on the official held-out test set (80.24% avg/class; 80.32% TTA), outperforming E3 by +10.30 pp. Finally, we diagnose deployment memory constraints under Linux cgroup v2 limits: while isolating the terminal feature map during Grad-CAM [4] cuts anonymous heap to 285.14 MiB, peak container memory hits the 512 MiB ceiling (VmHWM = 501.0 MiB, 1,366 pressure events), explaining why E3 ResNet18 serves as our low-memory Render backup while higher-capacity E7 ConvNeXt-Tiny is hosted on Lightning AI Free CPU.
  ]
]
#v(0.3em)

#show: rest => columns(2, rest)

= Methodology

== Problem Formulation & Benchmark Integrity
The CUB-200 benchmark [1] poses a closed-set fine-grained classification problem over $C = 200$ avian species. The objective is to learn a mapping $f_theta: cal(X) -> cal(Y)$ where $cal(X) subset bb(R)^(3 times H times W)$ denotes RGB photographs and $cal(Y) = {0, 1, dots, 199}$ represents zero-indexed taxonomic categories.

*Data Audit & Annotation Conflicts:* A pre-experimental audit verified all 4,829 training images listed in `train.txt` without corruption. Dimensions range from $162 times 121$ to $500 times 500$ pixels with 16 to 31 samples per class. Decoded RGB pixel hashing uncovered three exact duplicate pairs with contradictory annotation labels in the benchmark metadata:
1. `Crested_Auklet_0024` (label 4) vs `Parakeet_Auklet_0027` (label 6)
2. `Philadelphia_Vireo_0002` (label 152) vs `Warbling_Vireo_0004` (label 155)
3. `Red_eyed_Vireo_0029` (label 153) vs `Tennessee_Warbler_0009` (label 178)

Rather than arbitrarily altering ground truth, we set `conflict_policy: train_only`, pinning all six conflicting images to the training partition. This eliminated data leakage into validation while disclosing a 0.16% annotation conflict rate. Similarly, an audit of the 1,204 held-out test images (`test.txt`) identified four cross-split pixel duplicates with conflicting labels, including `Test/0197.jpg` (Purple Finch vs American Goldfinch), visually confirmed as a multi-species feeder photo. We retained the official 1,204-image denominator without silent modification, preserving benchmark integrity.

== Grouped Stratified Splitting & Augmentations
The 4,829 images were partitioned into a frozen development split of 3,870 training images (80.14%) and 959 validation images (19.86%) using seed 42 (fingerprint `55f05f6a...`). The split preserves per-class representation (~19.35 train / ~4.80 val) while grouping duplicate hashes into identical folds.

*Preprocessing & Augmentation Pipeline:* Input images are normalized using ImageNet channel statistics ($mu=[0.485, 0.456, 0.406]$, $sigma=[0.229, 0.224, 0.225]$):
- *Training Pipeline (E2–E7):* Random resized cropping to $224 times 224$ (conservative scale 0.7–1.0 to preserve fine-grained plumage details), horizontal flipping ($p=0.5$), mild rotation ($plus.minus 10 degree$), and subtle color jitter (brightness 0.15, contrast 0.15, saturation 0.10). *Crucially, baseline E1 used NO data augmentation*, employing only deterministic cropping during training.
- *Evaluation Pipeline (All Models):* Deterministic bicubic resize to $256 times 256$ followed by a center crop to $224 times 224 times 3$.

== Architectural Progression
We investigated three distinct architectural families across seven experimental regimes:

*1. Custom CNN Family (Scratch):*
- *Baseline CNN (E1; 439,816 parameters):* 4 convolutional stages ($3 -> 32 -> 64 -> 128 -> 256$ channels, $3 times 3$ kernels, stride 1, padding 1) with ReLU and $2 times 2$ MaxPool, collapsed via Global Average Pooling (GAP) into `Linear(256, 200)`.
- *Regularised CNN (E2; 440,296 parameters):* Incorporates 2D Batch Normalization after each convolution (conv bias omitted under BatchNorm) and Dropout ($p=0.4$) preceding the classifier.
- *Controlled No-Dropout Ablation (E4; 440,296 parameters):* Identical to E2 in optimizer, augmentations, and BatchNorm (440,296 parameters), but isolates the effect of Dropout by setting $p=0.0$.

*2. Residual Network Family (Transfer Learning) [2]:*
- *Pretrained ResNet18 Transfer (E3; 11,279,112 parameters):* Pretrained ImageNet-1K backbone (`IMAGENET1K_V1`). The default 1000-way head is replaced with `Linear(512, 200)` (102,600 parameters).
- *Label Smoothing Regularisation (E5; 11,279,112 parameters):* Identical to E3, evaluating output-space regularization $alpha = 0.05$ on the training objective [5].
- *Enhanced ResNet18 Fine-Tuning (E6; 11,279,112 parameters):* Evaluates joint unfreezing of `layer3` (256 ch), `layer4` (512 ch), and `fc` with discriminative learning rates.

*3. Modern Inverted-Bottleneck ConvNet Family [3]:*
- *ConvNeXt-Tiny Milestone (E7; 27,973,928 parameters):* Modern architecture featuring $7 times 7$ depthwise convolutions, LayerNorm, inverted bottleneck design (channel expansion 4$times$), and GELU activations (~2.5$times$ the parameters of ResNet18). The classifier is adapted via `Sequential(LayerNorm2d(768), Flatten(1), Linear(768, 200))` (155,336 parameters).

== Optimisation & Staged Schedules
*Scratch Models (E1, E2, E4):* E1 uses Adam [7] ($"lr"=10^(-3)$); E2 and E4 use AdamW [8] ($"lr"=3 times 10^(-4)$, weight decay $10^(-4)$). A `ReduceLROnPlateau` scheduler (factor 0.5, patience 3) monitors validation loss with early stopping (patience 8 epochs).

*Staged Transfer Learning (E3, E5, E6, E7):* To prevent disrupting pretrained representations, training proceeds in two stages:
- *Stage 1 (Head Warmup; 5 epochs):* The backbone is frozen; only the 200-class head is trained with AdamW ($"lr"=10^(-3)$, weight decay $10^(-4)$). Backbone BatchNorm layers are locked in evaluation mode (`eval()`) to preserve ImageNet running statistics.
- *Stage 2 (Backbone Fine-Tuning; up to 30 epochs):*
  - For E3 and E5: `layer4` (8.4M params) and `fc` are fine-tuned with AdamW ($"lr"=10^(-4)$, weight decay $10^(-4)$).
  - For E6: `layer3`, `layer4`, and `fc` use discriminative rates ($10^(-5)$, $5 times 10^(-5)$, $2 times 10^(-4)$).
  - For E7: Stage 4 (`features.7`, 14.3M params) and `classifier` use discriminative rates ($5 times 10^(-5)$ and $2 times 10^(-4)$) with CUDA Automatic Mixed Precision (AMP).
  The plateau scheduler is reset between stages to provide a clean fine-tuning schedule.

== Inference Enhancements & Explainability
*Bilateral Reflection Test-Time Augmentation (TTA):* Because bird morphology is bilaterally symmetric, TTA averages softmax probabilities across the original image $x$ and its horizontal reflection $x_("flip")$:
$ p_("TTA")(y | x) = 1/2 [ sigma(f_theta(x)) + sigma(f_theta(x_("flip"))) ] $
Inference runs sequentially ($1 + 1$), deleting intermediate tensors to avoid batch memory spikes.

*Grad-CAM Feature Attribution [4]:* To compute spatial heatmaps without allocating autograd graphs across the entire backbone, we attach a forward hook to the terminal convolutional stage (`layer4` for ResNet18; `features.7` for ConvNeXt-Tiny). The hook detaches the feature map $A^k$ and marks it with `requires_grad_(True)`:
$ alpha_k^c = 1/Z sum_(i=1)^H sum_(j=1)^W (partial y^c)/(partial A_(i,j)^k), wide L_("Grad-CAM")^c = "ReLU"(sum_k alpha_k^c A^k) $
Backpropagation computes gradients through head layers only, bypassing upstream backbone parameters and conserving CPU memory.

== Evaluation Metrics Contract
Performance is evaluated across all $C = 200$ classes:
- *Top-1 Accuracy:* Overall sample accuracy:
  $ "Top-1" = 1/N sum_(k=1)^N bb(I){arg max_c (p_c^k) = y_k^"gt"} $
- *Average Accuracy per Class (Macro Recall):*
  $ "Ave" = 1/C sum_(i=1)^C T_i, wide T_i = 1/N_i sum_(k in C_i) bb(I){arg max_c (p_c^k) = i} $
- *Top-5 Accuracy:* Evaluates whether $y_k^"gt" in "Top-5"(p^k)$.
- *Expected Calibration Error (ECE) [6]:* Evaluates probability reliability across $M=10$ confidence bins:
  $ "ECE" = sum_(m=1)^M (|B_m|)/N | "acc"(B_m) - "conf"(B_m) | $

= Result and Discussion

== Quantitative Benchmark Comparison
Table 1 reports quantitative performance across all seven investigated pipelines. Pre-committed candidate models (E1, E2, E3) were evaluated once on the official held-out test set; exploratory models (E4–E6) were evaluated on the validation split. Following frozen development, E7 was evaluated on the official test set as an exploratory post-test benchmark with disclosed history.

#place(top, scope: "parent", float: true)[
  #figure(
    caption: [Empirical Evaluation across CUB-200 Development and Held-Out Test Splits. All models trained on the frozen 3,870-image split. Best epoch selected via validation Average Accuracy per Class. Clean-Train Top-1 evaluated without augmentation. Official test set evaluated once on pre-committed models (E1, E2, E3); E7 evaluated post-test as an exploratory milestone on frozen weights without tuning.],
    table(
      columns: (1.45fr, 0.65fr, 0.58fr, 0.75fr, 0.75fr, 0.75fr, 0.7fr, 0.68fr, 0.75fr, 0.75fr, 0.75fr),
      align: (left, center, center, center, center, center, center, center, center, center, center),
      stroke: 0.4pt + luma(180),
      fill: (col, row) => if row == 0 { rgb("#e2e8f0") } else if row == 4 { rgb("#f0fdf4") } else if row == 8 { rgb("#eff6ff") } else { none },
      [ *Experiment* ], [ *Params* ], [ *Best Ep* ], [ *Clean Tr 1* ], [ *Val Top-1* ], [ *Val Avg/Cl* ], [ *Val Top-5* ], [ *Val Loss* ], [ *Gap (pp)* ], [ *Test Top-1* ], [ *Test Avg/Cl* ],
      [ E1 Baseline CNN ], [ 0.44M ], [ 31/40 ], [ 80.18% ], [ 21.69% ], [ 21.29% ], [ 42.75% ], [ 6.6823 ], [ 58.49\* ], [ 21.01% ], [ 20.63% ],
      [ E2 Regularised (Dr 0.4) ], [ 0.44M ], [ 21/29 ], [ 16.61% ], [ 10.43% ], [ 9.63% ], [ 27.11% ], [ 4.4516 ], [ 6.19 ], [ 8.97% ], [ 8.80% ],
      [ E4 No Dropout (Dr 0.0) ], [ 0.44M ], [ 38/46 ], [ 30.47% ], [ 14.39% ], [ 13.93% ], [ 34.83% ], [ 4.1777 ], [ 16.08 ], [ — ], [ — ],
      [ *E3 ResNet18 (Selected)* ], [ 11.28M ], [ 34/35 ], [ 99.90% ], [ *69.66%* ], [ *69.10%* ], [ *89.57%* ], [ *1.1895* ], [ 30.24 ], [ *70.43%* ], [ *70.12%* ],
      [ E3 + 2-View TTA ], [ 11.28M ], [ 34/35 ], [ 99.90% ], [ 70.80% ], [ 70.27% ], [ 89.99% ], [ — ], [ 29.10 ], [ — ], [ — ],
      [ E5 Label Smoothing (0.05) ], [ 11.28M ], [ 27/35 ], [ 99.90% ], [ 68.72% ], [ 68.02% ], [ 88.11% ], [ 1.4237 ], [ 31.18 ], [ — ], [ — ],
      [ E6 Enhanced ResNet18 ], [ 11.28M ], [ 24/32 ], [ 99.64% ], [ 68.40% ], [ 67.88% ], [ 88.95% ], [ 2.0790 ], [ 31.23 ], [ — ], [ — ],
      [ *E7 ConvNeXt-Tiny (Milestone)* ], [ 27.97M ], [ 27/30 ], [ 99.82% ], [ *80.40%* ], [ *79.75%* ], [ *96.35%* ], [ *0.6985* ], [ *19.42* ], [ *80.73%* ], [ *80.24%* ],
      [ E7 + 2-View TTA ], [ 27.97M ], [ 27/30 ], [ 99.82% ], [ *81.02%* ], [ *80.35%* ], [ *96.25%* ], [ — ], [ *18.80* ], [ 80.32% ], [ 79.86% ]
    )
  )
  #v(0.15em)
  #figure(
    placement: none,
    caption: [Cross-model comparative benchmark across all seven experimental pipelines on the frozen development split. (a) Clean-train vs validation Top-1 accuracy (N=3,870 / 959). (b) Validation metrics: Top-1, Average Accuracy per Class, and Top-5. (c) Generalization gap (clean-train minus validation Top-1 in percentage points), illustrating scratch memorization (E1), capacity starvation (E2), representation breakthrough (E3), and architectural scaling (E7).],
    image("/outputs/experiment_001/figures/model_comparison.png", width: 98%)
  )
  #v(0.25em)
]

== Learning Dynamics & The Overfitting Spectrum
The empirical benchmarks in Figure 1 and Table 1 illustrate distinct learning regimes across the seven pipelines:

*1. Scratch Baseline Memorization (E1):* In the unregularized CNN, training accuracy reached 93.05% by epoch 40, while validation accuracy peaked at 21.69% at epoch 31. Validation loss diverged from 4.70 to 10.30. Clean-train evaluation revealed a generalization gap of *58.49 pp at best epoch 31* (80.18% vs 21.69%) and widened to *71.99 pp at final epoch 40* (93.05% vs 21.06%). Although CNNs possess spatial translation equivariance, training 440K parameters from scratch on only ~19 images/class without visual priors leads to memorizing idiosyncratic background features rather than robust avian landmarks.

*2. Scratch Regularization & Capacity Starvation (E2 vs E4):* Adding regularization in E2 (BatchNorm, Dropout $p=0.4$, weight decay, augmentations) successfully reduced the generalization gap to 6.19 pp. However, validation Top-1 fell to 10.43% and Average Accuracy per Class dropped to 9.63%. 

To isolate the mechanism, controlled ablation E4 removed Dropout ($p=0.0$). As shown in Figure 2, E4 restored validation Top-1 to 14.39% (+3.96 pp) and Average Accuracy per Class to 13.93% (+4.30 pp), while clean-train accuracy rose to 30.47%. This single-run comparison indicates that Dropout 0.4 induced *capacity starvation*: zeroing 40% of feature activations in a compact 256-channel embedding restricted gradient propagation for fine-grained distinctions under heavy augmentation.

*3. Pretrained Transfer Representation Breakthrough (E3):* Pretrained ImageNet-1K features fundamentally resolved this bottleneck. During Stage 1 warmup (epochs 1–5), linear classification on frozen features reached 53.60% validation Top-1 in 5 epochs. Fine-tuning `layer4` at epoch 6 converged at epoch 34 to *69.66% validation Top-1* and *69.10% Average Accuracy per Class* (val loss 1.1895).

*Generalization Consistency vs Overfitting:* On the official held-out test set (1,204 images), E3 achieved *70.43% Top-1* and *70.12% Average Accuracy per Class*, aligning closely with validation scores (+0.77 pp and +1.02 pp). However, this consistency across partitions does not prove an unbiased estimator or guarantee the absence of leakage: finite sample variance (binomial standard error $"SE" = 1.32$ pp; 95% Wald CI $[67.85%, 73.01%]$) and four known cross-split duplicate pairs constrain exact generalization guarantees, while E3 clean-train Top-1 reached 99.90% ($"loss" = 0.0060$), demonstrating an underlying *30.24 pp generalization gap*.

== Post-Test Exploratory Investigations

*1. Output Regularization via Label Smoothing (E5):* E5 tested whether label smoothing ($alpha = 0.05$) [5] could mitigate logit saturation. E5 elevated the clean-train loss floor from 0.0060 to 0.0470 and reduced average softmax confidence on *incorrect* validation predictions from 53.86% to 31.31% (-22.55 pp). However, uniform non-target penalties compressed margins between closely related sister species (e.g. vireos and warblers). Validation loss plateaued prematurely at epoch 11, slightly reducing validation Top-1 to 68.72% (-0.94 pp) and shifting calibration from mild overconfidence (ECE 0.0609) to underconfidence (ECE 0.1341).

*2. Deeper Backbone Fine-Tuning (E6):* Unfreezing `layer3` (256 ch) alongside `layer4` with discriminative rates ($10^(-5), 5 times 10^(-5), 2 times 10^(-4)$) did not improve accuracy (68.40% val Top-1, -1.25 pp). Adapting 11.18M parameters on fine-grained data inflated validation loss to 2.0790 and increased calibration error to ECE 0.2021.

#place(top, scope: "parent", float: true)[
  #grid(
    columns: (1fr, 1.22fr),
    gutter: 8pt,
    [
      #figure(
        caption: [Controlled Dropout Ablation (E2 Dropout 0.4 vs E4 Dropout 0.0). Removing dropout restored feature capacity, raising Average Accuracy per Class from 9.63% to 13.93% (+4.30 pp).],
        image("/outputs/experiment_001/figures/ablation_e2_vs_e4.png", width: 100%)
      )
    ],
    [
      #figure(
        caption: [Architectural Comparison: E3 ResNet18 vs E7 ConvNeXt-Tiny. ConvNeXt-Tiny lifted validation Top-1 from 69.66% to 80.40% (+10.74 pp), reduced loss by 41.3%, and narrowed the generalization gap from 30.24 pp to 19.42 pp.],
        image("/outputs/experiment_001/figures/ablation_e3_vs_e7.png", width: 100%)
      )
    ]
  )
  #v(0.15em)
  #figure(
    placement: none,
    caption: [Focused Grad-CAM Attribution Comparison: E3 ResNet18 (`layer4`) vs E7 ConvNeXt-Tiny (`features.7`) on illustrative bird examples selected from the training dataset (`train_root`). Both models highlight salient diagnostic regions (crown, throat, plumage) for target logits with high confidence. Note: Saliency heatmaps indicate spatial correlation with the predicted class logit, not biological reasoning proof.],
    image("/outputs/experiment_001/figures/gradcam_focused.png", width: 100%)
  )
  #v(0.2em)
]

*3. Bilateral Test-Time Augmentation (TTA):* Applying horizontal-flip reflection averaging to frozen E3 improved validation Top-1 to *70.80%* (+1.14 pp) and Average Accuracy per Class to *70.27%* (+1.17 pp). For E7, TTA lifted Top-1 from 80.40% to *81.02%* (+0.63 pp). Reflection averaging reduces viewing-orientation variance at zero training cost.

*4. Modern Convolutions Milestone (E7 ConvNeXt-Tiny) [3]:*
Developed as a post-test exploratory milestone on validation data and evaluated once on the frozen official test set with disclosed history (Figure 3):
- *Validation Top-1:* *80.40%* (+10.74 pp over E3; *81.02%* with TTA).
- *Official Test Top-1:* *80.73%* (+10.30 pp over E3; *80.32%* with TTA).
- *Official Test Average Accuracy per Class:* *80.24%* (+10.12 pp over E3; *79.86%* with TTA).
- *Official Test Top-5 & Loss:* *95.18%* Top-5 (+6.06 pp over E3); test loss *0.7595* (36.7% lower than E3).
- *Generalization Consistency:* Test Top-1 (80.73%, $"SE" = 1.14$ pp; 95% Wald CI $[78.50%, 82.96%]$) aligns with validation (80.40%), confirming statistically significant gains over E3 ($[67.85%, 73.01%]$).
- *Per-Class Recall:* 94 of 200 species achieved 100% test recall (vs 45 in E3); zero classes scored 0% recall (vs 4 in E3).

*Architectural Factors:* ConvNeXt-Tiny's 27.97M parameters incorporate $7 times 7$ depthwise convolutions, an inverted bottleneck design with LayerNorm, and GELU non-linearities, providing richer feature representations than ResNet18's $3 times 3$ residual blocks.

== Model Interpretability & Grad-CAM Analysis
Figure 4 visualizes Grad-CAM saliency maps [4] comparing E3 ResNet18 and E7 ConvNeXt-Tiny on illustrative examples selected from the training dataset (`train_root`):
- *Case 1 (Red-headed Woodpecker, class 190; Demo Image):* Both E3 (99.8%) and E7 (99.98%) concentrate saliency on the crimson head and dark mantle, suppressing background tree bark.
- *Case 2 (American Goldfinch, class 46):* Saliency localizes on the yellow breast and dark crown. E7 displays tighter boundary adherence along the bird's silhouette.
Saliency maps indicate localized spatial regions contributing to the predicted logit; they do not establish human-like visual reasoning.

== Qualitative Error Analysis & Confused Species
Figure 5 displays official held-out test diagnostics for E3 ResNet18:
1. *Sister Species Ambiguity:* Confusion concentrates within taxonomic genera:
   - *Vireos:* `Philadelphia Vireo` (152) vs `Warbling Vireo` (155) share yellowish breasts, white eye-stripes, and gray crowns. In E3, 40% of Warbling Vireo validation samples were misclassified as Philadelphia Vireo.
   - *Gulls:* `Glaucous-winged Gull` (59) vs `Western Gull` (64) differ only in subtle mantle gray shading and wingtip melanism.
2. *Ambiguous Data & Annotation Conflicts:* Our test audit identified `Test/0197.jpg`, an exact duplicate of `Train/American_Goldfinch_0013` (label 46) annotated as `Purple Finch` (label 34). The photo captures both species at a garden feeder. Under closed-set single-label softmax, any model must incur an error on one of the valid species.
3. *Camouflage & Foliage Occlusion:* Marsh birds (e.g. `American Bittern`) against dense reeds trigger misclassifications when background textures overwhelm bird contours.

#place(top, scope: "parent", float: true)[
  #figure(
    caption: [Official held-out test evaluation diagnostics on E3 ResNet18 (1,204 images, 200 classes). (a) Per-class test accuracy distribution (histogram across 200 species, median 75.0%, 45 classes at 100%, 4 classes at $<= 25%$). (b) Per-class test accuracy vs training sample count (scatter with linear fit, Pearson $r = 0.048$, showing discrete recall steps under 4–8 test samples/class). (c) Top misclassified species pairs highlighting fine-grained sibling ambiguities (e.g. vireos, gulls, warblers).],
    image("/outputs/final_test_results/figures/test_per_class_and_confusions.png", width: 100%)
  )
  #v(0.15em)
]

== Cloud Resource Engineering & Deployment Diagnosis
To fulfill the interactive web deployment without paid hosting, we evaluated our pipeline for the *Render Free Web Service* (512 MiB hard RAM limit, no swap).

#figure(
  caption: [Runtime CPU memory profiling under Linux cgroup v2 constraints (`MemoryMax=512M`, single-core CPU execution).],
  table(
    columns: (1.25fr, 0.82fr, 0.82fr, 0.88fr, 1.25fr),
    align: (left, center, center, center, center),
    stroke: 0.4pt + luma(180),
    fill: (col, row) => if row == 0 { rgb("#e2e8f0") } else { none },
    [ *Configuration* ], [ *RssAnon* ], [ *VmRSS* ], [ *Peak HWM* ], [ *cgroup Status* ],
    [ E3 ResNet18 (Single) ], [ 210.4 MiB ], [ 288.6 MiB ], [ 379.2 MiB ], [ SAFE (+133 MiB) ],
    [ E3 ResNet18 (2-View TTA) ], [ 211.2 MiB ], [ 289.4 MiB ], [ 380.7 MiB ], [ SAFE (+131 MiB) ],
    [ E7 ConvNeXt (Single CAM) ], [ 282.0 MiB ], [ 485.4 MiB ], [ 499.4 MiB ], [ MARGINAL (+12 MiB) ],
    [ E7 ConvNeXt (TTA CAM) ], [ 285.1 MiB ], [ 485.1 MiB ], [ 501.0 MiB ], [ CRITICAL (Hit 512M) ]
  )
)
#v(0.1em)

*Memory Optimizations & Diagnosis:*
1. *CPU Wheels & Memory Mapping:* CPU PyTorch wheels eliminated CUDA runtime libraries. Checkpoint loading uses `mmap=True` and `weights_only=True` on Linux, mapping weights into reclaimable page cache (`RssFile`).
2. *Meta-Device Initialization:* Model parameters are allocated on `torch.device("meta")` and assigned directly (`assign=True`), preventing heap duplication during startup.
3. *Terminal Feature Map Isolation & Container Pressure:* Initial diagnostics reported a 765 MiB peak from whole-backbone autograd tracking. Isolating the terminal feature map (`features.7`) computes gradients exclusively through the classifier head, cutting anonymous heap (`RssAnon`) to 285.14 MiB. However, under an enforced 512 MiB Linux cgroup with zero swap, total resident memory (`VmRSS`) reaches 485.14 MiB and peak memory (`VmHWM`) reaches *501.0 MiB*, driving cgroup `memory.peak` to the 512.0 MiB limit. While zero OOM kills occurred (`oom_kill = 0`), the kernel triggered 1,366 direct reclaim events (`max = 1366`). Because `RssAnon` represents only private heap and excludes mapped libraries (`RssFile`), it cannot be used alone to claim safe headroom: E7 operates with less than 11 MiB of true physical margin.

*Deployment Selection Rationale:*
Although E7 achieves 0 OOM kills via feature isolation, its 501.0 MiB VmHWM and 1,366 memory pressure events make it unsafe for unmonitored production on Render Free (512 MiB limit). We therefore adopt a multi-tier hosting architecture: the primary landing page and documentation reside on Hugging Face Spaces (#link("https://huggingface.co/spaces/ShadWVege/cos30082-bird-species-explorer")[#text(fill: rgb("#1d4ed8"))[cos30082-bird-species-explorer]]), while live interactive E7 inference is deployed on Lightning AI Free CPU Studio (#link("https://7863-01m4vaa4swe6b7ha57xrh4ch7s.cloudspaces.litng.ai/")[#text(fill: rgb("#1d4ed8"))[7863-...litng.ai]], port 7863, ~71 s cold start, auto-sleep). Pre-committed *E3 ResNet18* (with optional 2-view TTA; Figure 6; #link("https://cos30082-bird-classifier.onrender.com/")[#text(fill: rgb("#1d4ed8"))[cos30082-bird-classifier.onrender.com]]) serves as low-memory fallback on Render Free with 131 MiB physical headroom (VmHWM 380.7 MiB) and 101 ms latency. Alternative hosting requires instructor approval.

#colbreak()

#figure(
  caption: [Live Gradio application running on Render Free Web Service. Displays Top-5 predictions for Red-headed Woodpecker (class 190, 99.98%), 2-view TTA toggle, and real-time Grad-CAM explainability overlay.],
  image("/outputs/experiment_001/figures/live_render_tta_view.png", width: 100%, height: 4.8cm, fit: "contain")
)
#v(0.15em)

== Limitations, Ethical Considerations, & Future Work
*Limitations:* CUB-200 metadata contains annotation conflicts (0.16% training, 0.33% cross-split), small class sample sizes (~19 images/class), and closed-set assumption breakdown on non-avian inputs. Free-tier cloud hosting on Render Free Web Service is subject to ephemeral container lifecycle, idle spin-down after 15 minutes of inactivity (~50 s cold start), and best-effort availability rather than permanent uptime.
*Ethics & Environment:* Deep learning deployment balances accuracy against inference carbon footprint. While ConvNeXt-Tiny adds +10.74 pp accuracy, it increases parameters by ~2.5$times$ and doubles inference latency.
*Future Work:* Knowledge distillation from ConvNeXt-Tiny into MobileNetV4, hierarchical taxonomy-aware loss formulations, and open-set out-of-distribution rejection.

== Submission Verification & Deliverables
All coursework deliverables are archived and reproducible:
- *Presentation Video:* #text(fill: rgb("#b45309"))[https://youtu.be/... (YOUTUBE_VIDEO_URL_PENDING)]
- *Primary Project Portal (HF Space):* #link("https://huggingface.co/spaces/ShadWVege/cos30082-bird-species-explorer")[#text(fill: rgb("#1d4ed8"))[cos30082-bird-species-explorer]] (Static landing page & architecture docs)
- *Interactive E7 Classifier (Lightning AI):* #link("https://7863-01m4vaa4swe6b7ha57xrh4ch7s.cloudspaces.litng.ai/")[#text(fill: rgb("#1d4ed8"))[7863-...litng.ai]] (Live 80.73% ConvNeXt-Tiny demo, port 7863)
- *Backup E3 Application (Render Free):* #link("https://cos30082-bird-classifier.onrender.com/")[#text(fill: rgb("#1d4ed8"))[cos30082-bird-classifier.onrender.com]] (ResNet18 512 MiB production fallback)
- *Deployment Repository:* #link("https://github.com/ShadWVege/cos30082-bird-classifier")[#text(fill: rgb("#1d4ed8"))[ShadWVege/cos30082-bird-classifier]] (Inference-only deployment codebase)
- *Coursework Source Archive:* `outputs/source_submission.zip` (Complete reproducible standalone archive; commented source, configs, 27 unit tests, docs, figures; checkpoint weights excluded per LMS brief).

== References
#set text(size: 6.8pt)
#set par(leading: 0.32em)
[1] P. Welinder, S. Branson, T. Mita, C. Wah, F. Schroff, S. Belongie, and P. Perona, "Caltech-UCSD Birds 200," California Institute of Technology, Tech. Rep. CNS-TR-2010-001, 2010. \
[2] K. He, X. Zhang, S. Ren, and J. Sun, "Deep Residual Learning for Image Recognition," in _IEEE CVPR_, 2016, pp. 770–778. \
[3] Z. Liu, H. Mao, C. Y. Wu, C. Feichtenhofer, T. Darrell, and S. Xie, "A ConvNet for the 2020s," in _IEEE CVPR_, 2022, pp. 11976–11986. \
[4] R. R. Selvaraju, M. Cogswell, A. Das, R. Vedantam, D. Parikh, and D. Batra, "Grad-CAM: Visual Explanations from Deep Networks," in _IEEE ICCV_, 2017, pp. 618–626. \
[5] C. Szegedy, V. Vanhoucke, S. Ioffe, J. Shlens, and Z. Wojna, "Rethinking the Inception Architecture for Computer Vision," in _IEEE CVPR_, 2016, pp. 2818–2826. \
[6] C. Guo, G. Pleiss, Y. Sun, and K. Q. Weinberger, "On Calibration of Modern Neural Networks," in _ICML_, 2017, pp. 1321–1330. \
[7] D. P. Kingma and J. Ba, "Adam: A Method for Stochastic Optimization," in _ICLR_, 2015. \
[8] I. Loshchilov and F. Hutter, "Decoupled Weight Decay Regularization," in _ICLR_, 2019.
