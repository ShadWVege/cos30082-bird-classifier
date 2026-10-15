# Current handoff

Updated: 2026-10-05. Phase: runnable scaffold and verified local data preparation; full experiments pending.

## Completed

- Retrieved and preserved the original three attachments; read all three PDF pages.
- Created PLAN.md before implementation, plus source files, configs, agent instructions and execution guides.
- Connected external image roots to `C:/Users/60103/Downloads/Train` and `C:/Users/60103/Downloads/Test`.
- Audited all 4,829 training images: no missing/corrupt images, all RGB, 200 classes, 16-31 images per class.
- Found three exact duplicate pairs with conflicting labels; kept original annotations and pinned all six images to training. See `docs/DATA_DECISIONS.md`.
- Saved `data/prepared/split.json`: 3,870 training / 959 validation, seed 42, every class in both partitions, no exact duplicate crossing the split.
- Implemented three models, deterministic preprocessing, staged ResNet fine-tuning, checkpoint/resume with RNG state, learning curves, required metrics, error analysis, optional Top-5/macro F1 and Grad-CAM.
- Implemented validation-based checkpoint selection, an explicit final-test gate, portable inference and Gradio Space export.
- Fourteen tests passed in the development test environment, including interrupted/resumed training matching uninterrupted training exactly on CPU.
- A two-batch real-image baseline smoke run passed at 224 pixels and batch size 4 on CPU. It is explicitly marked as a smoke test and is ineligible for final comparison. Its accuracy values are not assignment evidence.
- Source packaging, Windows setup, cloud training, deployment and report/video planning are provided.
- An isolated Python 3.12 CPU environment is installed in `.venv`; use its interpreter directly. `src.doctor` confirms PyTorch 2.14.1+cpu and no CUDA runtime in this development build. The hardware GPU is present but a CUDA build is still needed for GPU training.

## Current artifacts

- `data/prepared/train_audit.json`: measured image inventory and integrity findings.
- `data/prepared/split.json`, `class_names.json`, `class_counts.csv`: common data definitions.
- `outputs/smoke_baseline_cpu/`: real-image plumbing check only; best/last checkpoints and figures.
- `docs/SUBMISSION_PLAN.md`: exact report/video requirements and evidence checklist.

## Next execution step

1. Use the PyTorch installation selector to install a compatible CUDA pair for the detected RTX 3060 Laptop GPU (6 GB). System `python` is Python 2.7; use the working `.venv/Scripts/python.exe`. The setup guide includes a force-reinstall command so the existing CPU build is actually replaced.
2. Confirm `python -m src.doctor` reports CUDA and run a small GPU smoke test.
3. Run `python -m src.run_experiments --out outputs/experiment_001 --device cuda`; preserve epoch logs and checkpoints. It does not touch test images.
4. Inspect validation evidence; add justified ablations only if useful. Freeze choices before explicitly running `src.evaluate --final`.

## Not yet done / needed later

- No full training or official test accuracy has been produced. The supplied test images remain unopened by the workflow.
- The pretrained ResNet18 weight download has not been exercised yet; synthetic tests instantiate it without downloading weights.
- No trained app has been published. Need the user's Hugging Face account/destination when the real model is ready.
- The final report must use real results and the user's name/student ID. Need an actual recorded/published video URL; no final report or video is fabricated.
- Package runtime/library versions are recorded per run; repeat GPU checks after changing the PyTorch build.

Continue from this state; do not rebuild or re-split the dataset.
