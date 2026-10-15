# COS30082 Assignment 1: implementation and execution plan

## Authoritative basis

Use `references/Assignment1_v2.pdf`, `data/train.txt`, and `data/test.txt` from the original conversation. The assignment is CUB-200 with 200 classes and 4,829 supplied training images. Do not substitute CUB-200-2011 or download another split. Source annotations use zero-based class labels. The PDF requires Top-1 Accuracy and Average Accuracy per Class; a PDF of at most four pages with **Methodology** and **Result and Discussion**, name and student ID; generously commented Python source in one ZIP; a video of at most 10 minutes; and a Hugging Face application. Include YouTube and Hugging Face links in the report.

## Sequence and completion gates

1. **Verify sources and scaffold.** Preserve the three original files, confirm annotation syntax/counts/labels, derive readable species names from training filenames, document setup and save this plan before implementation.
2. **Audit local training images.** Configure external image roots without copying large folders. Check missing files, decode failures, dimensions, colour modes, class counts, exact file/pixel duplicates and conflicting duplicate labels. Resolve problems before splitting. Never silently discard files. Test-image integrity checks are deferred to final evaluation.
3. **Freeze the development split.** Use seed 42 and approximately 20% validation per class. Keep exact duplicates in the same partition; save membership, source fingerprints, class names, audit and configuration. All experiments use exactly this split. If grouping makes a stratified split impossible, stop for a documented decision.
4. **Verify runnable code.** Test parsing, grouped splitting, metric definitions, preprocessing, checkpoint round trips, model outputs, stage transitions, and a tiny synthetic train/evaluate/export/app workflow. Synthetic results are software checks, never assignment results.
5. **Run the three experiments.** Start with a small smoke run on real training data, then run full experiments. Preserve run configs, source hashes, environment, learning curves, best/last checkpoints and validation metrics. Choose using validation Average Accuracy per Class, breaking ties with validation loss. Never promise or manufacture overfitting or improvement.
6. **Investigate generalisation.** Compare clean-training and validation accuracy (identical deterministic transforms), loss curves, per-class recall and confusion pairs. Treat combined regularisation as a bundle; optional single-factor ablations or repeat seeds are required for stronger causal/variance claims.
7. **Freeze model choices, then evaluate test.** Record selected checkpoints in a selection file before loading test annotations/images. Audit held-out integrity and train/test duplicates at this stage; flag contamination and never silently alter the official split. Compute required metrics, optional Top-5/macro F1, predictions and error figures for each frozen model. Do not tune from test results.
8. **Deploy the validation-selected model.** Export inference-only weights, training-derived class names and preprocessing into a standalone Gradio Space folder. Test locally, then publish to the user's Hugging Face Space once the destination/account is available. No fabricated checkpoint or demo metrics.
9. **Prepare final submission from real evidence.** Fit methodology, architecture/hyperparameters, comparison table, learning curves, qualitative errors, limitations and links into four pages. Plan a 9:30 video with room under the limit. Check all deliverables before packaging.

## Initial experiments (starting values, not claimed optimal)

| Setting | Baseline | Regularised CNN | ResNet18 transfer |
|---|---|---|---|
| Shared input / split / seed | 224 RGB / fixed stratified split / 42 | same | same |
| Architecture | 4 Conv-ReLU-Pool blocks, global average pool, classifier | same widths + BatchNorm + dropout 0.4 | ImageNet ResNet18, replace 1000-way head with 200-way head |
| Training augmentation | none | crop, horizontal flip, mild rotation and colour jitter | same |
| Optimizer | Adam | AdamW | AdamW |
| Initial LR | 0.001 | 0.0003 | head 0.001; final block 0.0001 |
| Weight decay | 0 | 0.0001 | 0.0001 |
| Loss | cross entropy | cross entropy | cross entropy |
| Budget | 40 epochs | up to 50, patience 8 | 5 head epochs + up to 30 final-block epochs, patience 8 |
| Batch size | 16 initially | 16 initially | 16 initially |
| Scheduler | none | validation-loss plateau | validation-loss plateau, reset between stages |

The baseline is a valid simple comparator. BatchNorm stabilises optimisation; it is not a guarantee of less overfitting. Hold backbone BatchNorm statistics fixed in head training and earlier blocks fixed during partial fine-tuning. Save a global best checkpoint across stages, but allow the fine-tuning stage its own patience budget.

## Metric contract

Top-1 = correct / total. Average Accuracy per Class = mean of correct-in-class / class-support over all 200 classes (macro recall), **not** one-vs-rest accuracy. Missing-class coverage is an error in official evaluation, not a reason to silently average fewer classes. Top-5 measures whether the correct label is among five logits; its justification is fine-grained ambiguity. All saved scores are fractions in [0, 1]; display percentages in the report.

## Present status

Plan and runnable scaffold are implemented. All 4,829 local training images passed decoding; three conflicting duplicate pairs were identified and explicitly pinned to training (see docs/DATA_DECISIONS.md). The frozen split is 3,870 training / 959 validation. Fourteen software tests and a two-batch real-image CPU smoke run passed. Full model training, final test evaluation, public deployment, final report and video remain to be executed. No assignment performance results exist yet.

## Technical references

- PyTorch transfer learning: https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
- ResNet18 weights and preprocessing: https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html
- Hugging Face Gradio Spaces: https://huggingface.co/docs/hub/spaces-sdks-gradio
