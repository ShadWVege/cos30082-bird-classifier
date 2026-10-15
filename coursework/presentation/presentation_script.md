# COS30082 Assignment 1: Bird Species Classification
## Verbatim 10-Minute Video Presentation Script

**Presenter:** Jonathan Zheng Li CHAI  
**Student ID:** 104400675  
**Unit:** COS30082 Applied Machine Learning  
**Target Duration:** 9 minutes 20 seconds (Strict ≤ 10:00 limit)  
**Primary Project Portal (HF Space):** https://huggingface.co/spaces/ShadWVege/cos30082-bird-species-explorer  
**Interactive Classifier (Lightning AI E7):** https://7863-01m4vaa4swe6b7ha57xrh4ch7s.cloudspaces.litng.ai/  
**Backup Web Application (Render E3):** https://cos30082-bird-classifier.onrender.com/  
**Deployment Repository (GitHub):** https://github.com/ShadWVege/cos30082-bird-classifier  
**Presentation Video:** https://youtu.be/... (YOUTUBE_VIDEO_URL_PENDING)  

---

### Segment 1: Title & Problem Overview (0:00 – 0:45)
**[Visual: Slide 1 — Title, Student Information, Live Links]**

> "Hello everyone. My name is Jonathan Zheng Li CHAI, Student ID 104400675, and welcome to my presentation for COS30082 Applied Machine Learning Assignment 1: Fine-Grained Bird Species Classification on the Caltech-UCSD Birds 200 dataset.
>
> In this assignment, our task is to classify images into 200 distinct bird species using a provided dataset of 4,829 training images. Fine-grained visual classification is notoriously difficult because differences between sibling species—such as warblers, vireos, and gulls—are extraordinarily subtle, often coming down to minor eye-stripes, beak contours, or wing bars. Meanwhile, intra-class variations in bird posture, lighting, and foliage backgrounds are massive, all under extreme data sparsity of only about 19 to 24 images per species.
>
> In this presentation, I will walk you through our complete machine learning pipeline: from rigorous dataset auditing and leakage prevention, to our progression from custom CNNs to staged transfer learning, our explainability analysis, and our live multi-tier cloud deployment."


---

### Segment 2: Data Audit, Duplicate Policy, & Split Protocol (0:45 – 1:30)
**[Visual: Slide 2 — Data Audit Table & Duplicate Hashing Diagram]**

> "Before training any model, we conducted a rigorous local audit of all 4,829 training images. Every single image decoded cleanly as RGB, with dimensions spanning 162 by 121 up to 500 by 500 pixels.
>
> During pixel-level SHA256 hashing, we uncovered a critical finding: three pairs of exact pixel-duplicate images existed in `train.txt` with contradictory ground-truth labels. For example, the exact same image was labelled both Crested Auklet and Parakeet Auklet. Rather than silently deleting them or guessing which label was correct, we enforced a strict policy: `conflict_policy: train_only`. All six images were explicitly pinned to our training split, completely eliminating data leakage into validation while transparently disclosing 0.16% label noise.
>
> Using seed 42, we established a frozen stratified group split of 3,870 training images and 959 validation images—preserving an exact 80/20 ratio per species while ensuring duplicate hashes never cross folds.
>
> Later, when auditing the official 1,204 held-out test images, we uncovered four cross-split pixel duplicates with conflicting labels, including `Test/0197.jpg`, which visual inspection confirmed is a multi-species feeder photo containing both an American Goldfinch and a Purple Finch. In accordance with benchmark integrity, we preserved the official test set untouched without any data tampering."

---

### Segment 3: Scratch Baseline (E1) & Catastrophic Memorization (1:30 – 2:15)
**[Visual: Slide 3 — E1 Architecture Diagram & Diverging Loss Curves]**

> "Our first model, E1, is a 4-stage convolutional baseline trained from scratch with 440,000 parameters. It uses four blocks of conv, ReLU, and max pooling, collapsed via Global Average Pooling into a linear classifier, optimized with Adam at a learning rate of 10 to the minus 3.
>
> The empirical results clearly demonstrated the classic machine learning trap: catastrophic memorization. Over 40 epochs, training accuracy climbed to 93.05%, while validation accuracy peaked at only 21.69% at epoch 31. Worse, validation loss exploded from 4.70 up to 10.30, resulting in a staggering 71.99 percentage point generalization gap.
>
> Because the model was initialized with random weights and given only 19 images per class, it lacked visual priors. Instead of learning invariant avian morphology, it simply memorized idiosyncratic background details like tree branches and sky patches."

---

### Segment 4: Regularization Traps & Dropout Ablation (E2 vs E4) (2:15 – 3:00)
**[Visual: Slide 4 — E2 vs E4 Learning Curves & Dropout Mechanism]**

> "To counter this severe overfitting, we developed E2, incorporating textbook regularization: 2D Batch Normalization after every convolution, 40% Dropout before the classifier, weight decay of 10 to the minus 4, and aggressive data augmentations including random resized crops, horizontal flips, mild rotations, and color jitter.
>
> E2 successfully collapsed the generalization gap from 58.49 down to 6.19 percentage points. However, it severely damaged model capacity: validation Top-1 accuracy collapsed to 10.43%, and Average Accuracy per Class dropped to just 9.63%!
>
> To discover whether Dropout or Batch Normalization was responsible, we conducted a strictly controlled causal ablation: Experiment E4, where we set Dropout to zero while keeping all other factors identical.
>
> As shown on the slide, removing Dropout boosted validation Average Accuracy per Class from 9.63% to 13.93%—a 4.30 percentage point gain—and clean-train accuracy rose from 16.6% to 30.5%. This proved our hypothesis: in a compact 256-channel network trained from scratch, dropping 40% of activations caused severe capacity starvation, preventing the network from learning subtle fine-grained features under heavy data augmentation."

---

### Segment 5: The Transfer Learning Breakthrough (E3 ResNet18) (3:00 – 4:15)
**[Visual: Slide 5 — E3 Staged Fine-Tuning Diagram & Official Test Results]**

> "To break this representational ceiling, we turned to transfer learning with an ImageNet-1K pretrained ResNet18 in Experiment E3.
>
> To avoid destroying pretrained weights, we designed a two-stage training schedule. In Stage 1, we froze the entire backbone and trained only the new 200-class linear classification head for 5 warmup epochs with AdamW at 10 to the minus 3, keeping backbone Batch Normalization in evaluation mode. This achieved 53.60% validation Top-1 in just 5 epochs!
>
> In Stage 2, we unfroze `layer4` and fine-tuned it alongside the head at a lower learning rate of 10 to the minus 4. The model converged at epoch 34, achieving 69.66% validation Top-1, 69.10% Average Accuracy per Class, and 89.57% Top-5 accuracy.
>
> Once our model selection was frozen in `outputs/selection.json`, we performed our single, official evaluation on the held-out test set of 1,204 images. In single-view evaluation, E3 achieved 70.43% Top-1 accuracy and 70.12% Average Accuracy per Class. The close agreement between our single-view validation and test scores—within 0.77 percentage points—demonstrates consistent generalization across both partitions of our frozen split, while the 30.24 percentage point clean-train gap accurately measures residual model capacity."

---

### Segment 6: Codebase Architecture & Design Walkthrough (4:15 – 5:45)
**[Visual: Slide 6 & VS Code Screen Share — `src/` modular components]**

> "Now, let me briefly show you how our code is structured. Everything is modularized under the `src` directory, and generously commented to explain each design choice:
>
> First, in `src/data.py`, we implement manifest parsing, path traversal sanitization, and grouped stratified fold assignment preserving duplicate hashes.
>
> Second, in `src/dataset.py`, we implement our dataset classes and data transforms. Notice that our validation, test, and inference pipelines all share the exact same deterministic center-crop transforms and ImageNet normalization, ensuring zero pipeline skew.
>
> Third, in `src/models.py`, our model factory encapsulates custom heads, BatchNorm bias omission, and staged freezing logic. Notice the `freeze_bn` flag: during head training, we explicitly keep BatchNorm layers in `eval` mode to ensure ImageNet running mean and variance are not corrupted by small fine-grained bird batches.
>
> Fourth, in `src/engine.py` and `src/train.py`, we manage our two-stage training loop with PyTorch Automatic Mixed Precision and gradient scaling. Notice how the learning rate plateau scheduler is explicitly reset between stages, giving the fine-tuning phase its own independent patience budget.
>
> Fifth, in `src/metrics.py`, our evaluation functions compute sample-level Top-1 accuracy, Average Accuracy per Class—which is unweighted macro recall across all 200 classes—as well as Top-5 accuracy and Macro-F1.
>
> Finally, our complete automated test suite in `tests/` contains 26+ unit tests, all passing cleanly."

---

### Segment 7: Advanced Post-Test Explorations (E5, E6, TTA, & ConvNeXt-Tiny) (5:45 – 7:00)
**[Visual: Slide 7 & Slide 8 — Ablation Tables & ConvNeXt-Tiny Comparison Curves]**

> "After freezing our official assignment model, we explored several advanced techniques strictly on our validation split:
>
> In Experiment E5, we investigated Label Smoothing with alpha equals 0.05. It successfully prevented clean-train loss collapse and reduced overconfidence on incorrect predictions by 22.55 percentage points. However, uniform smoothing slightly compressed margins between visually similar sister species, reducing validation Top-1 slightly to 68.72%.
>
> In Experiment E6, unfreezing deeper layers (`layer3` plus `layer4`) with discriminative learning rates increased validation loss and caused overfitting.
>
> However, our Test-Time Augmentation—or TTA—investigation was a great success. Because birds are bilaterally symmetric, we average softmax probabilities from the original image and its horizontally flipped view. This simple inference technique boosted our E3 validation Top-1 from 69.66% to 70.80%—a 1.14 percentage point boost at zero training cost!
>
> Finally, in Experiment E7, we benchmarked a modern depthwise-separable architecture: ConvNeXt-Tiny. By combining 7-by-7 depthwise convolutions, inverted bottlenecks, LayerNorm, and Stage 4 fine-tuning, E7 achieved a breakthrough: 80.40% validation Top-1, 79.75% Average Accuracy per Class, and 81.02% with TTA. In our post-test official evaluation on the 1,204 held-out test images, single-view E7 reached 80.73% Top-1 accuracy (+10.30 percentage points over E3) and 80.24% Average Accuracy per Class, with a statistically significant paired McNemar p-value of 2.48 times 10 to the minus 15! 94 out of 200 species achieved perfect 100% test recall."

---

### Segment 8: Explainability (Grad-CAM) & Failure Modes (7:00 – 7:45)
**[Visual: Slide 9 — Grad-CAM Heatmaps & Confused Species Examples]**

> "To examine where spatial feature attribution concentrates and verify that our models attend to relevant avian regions rather than obvious background shortcuts, we implemented Grad-CAM explainability:
>
> Looking at the heatmaps on Slide 9, which compare our pre-committed ResNet18 E3 and modern ConvNeXt-Tiny E7 on representative bird images, both models focus intensely on high-entropy morphological landmarks: the bill structure, crown, eye stripe, throat color, and wing bars, while suppressing background tree bark and foliage. While earlier scratch baselines attended diffusely to background clutter, pretrained representations isolate avian contours with remarkable specificity.
>
> When we analyze residual errors, we find three systematic failure modes:
> First, visually indistinguishable sister species—such as the Philadelphia Vireo versus Warbling Vireo, or Glaucous-winged Gull versus Western Gull, which share almost identical plumage.
> Second, ambiguous multi-species photographs, like `Test/0197.jpg`, where both a Purple Finch and an American Goldfinch appear together at a feeder.
> Third, extreme camouflage, such as marsh bitterns blending into tall reeds."

---

### Segment 9: Multi-Tier Cloud Deployment & Live Demonstration (7:45 – 8:45)
**[Visual: Slide 10 & Live Browser Demonstration: Hugging Face Landing Page -> Lightning AI E7 App -> Render E3 Backup]**

> "Now, let us examine our live cloud deployment ecosystem. We designed a resilient three-tier architecture to balance model capacity, memory safety, and free-tier cloud limits.
>
> **[Demonstrating Hugging Face Space in browser]**
> First, our primary project portal is hosted on Hugging Face Spaces at `https://huggingface.co/spaces/ShadWVege/cos30082-bird-species-explorer`. This provides our public landing page, interactive architecture documentation, and direct navigation links. Crucially, Hugging Face hosts the static presentation layer only, ensuring zero risk of free-tier compute exhaustion.
>
> **[Demonstrating Lightning AI E7 Application in browser]**
> From the landing page, we navigate to our primary interactive classifier on Lightning AI Free CPU Studio at `https://7863-01m4vaa4swe6b7ha57xrh4ch7s.cloudspaces.litng.ai/`, running via Gradio plugin on port 7863. Under Lightning AI's free tier, the studio automatically sleeps when idle to preserve compute within the 4-hour active session limit. When accessed, it automatically wakes up on demand in approximately 71 seconds.
>
> With 16 gigabytes of RAM on Lightning AI, we can comfortably run our flagship model: E7 ConvNeXt-Tiny, with 27.97 million parameters and 80.73% test accuracy. When we upload a bird photograph—such as this American Goldfinch:
> The model instantly predicts Class 046: American Goldfinch with high confidence, showing calibrated Top-5 probability bars.
> Furthermore, in the explainability section, we generate high-resolution Grad-CAM heatmaps from the terminal `features.7` stage, demonstrating that the network's 7x7 depthwise filters attend tightly to avian morphology rather than background noise.
>
> **[Demonstrating Render Free Web Service in browser]**
> Finally, we deployed our pre-committed E3 ResNet18 model as a resilient backup on the Render Free Web Service at `https://cos30082-bird-classifier.onrender.com/`. Render enforces a hard 512 megabyte cgroup memory ceiling. To run PyTorch within 512 megabytes, we engineered CPU-only wheels, memory-mapped loading with `mmap=True`, meta-device parameter assignment, and sequential TTA.
>
> In our Linux stress tests, E3 ResNet18 runs with 131 megabytes of genuine physical headroom (VmHWM 380.7 megabytes). While terminal feature isolation prevents OOM kills in ConvNeXt-Tiny, its container memory touches 501 megabytes and triggers kernel pressure events. Therefore, keeping E3 on Render provides a rock-solid, memory-safe production fallback, while E7 provides the high-capacity research demonstration on Lightning AI."


---

### Segment 10: Conclusion & Key Takeaways (8:45 – 9:30)
**[Visual: Slide 1 — Conclusion, Submission Links & Q&A Summary]**

> "To conclude, this project provided deep, practical insights into fine-grained machine learning and production engineering:
>
> 1. We demonstrated that training from scratch on sparse fine-grained data collapses into memorization, while heavy dropout starves model capacity.
> 2. Staged transfer learning using ResNet18 resolved this bottleneck, achieving 70.43% test accuracy and 70.12% Average Accuracy per Class on the official benchmark.
> 3. Bilateral reflection TTA and ConvNeXt-Tiny demonstrated that modern depthwise architectures reach over 80% accuracy with a dramatically narrower generalization gap.
> 4. Finally, through memory-mapped loading and sequential TTA, we successfully deployed a fully functional, explainable Gradio application within Render Free's 512 megabyte ceiling.
>
> All verified training pipelines, configs, test suites, and documentation are packaged in our complete standalone source submission ZIP, with the inference application deployed on GitHub and cloud hosting.
>
> Thank you very much for your time."

