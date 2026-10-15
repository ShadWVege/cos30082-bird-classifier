# Bird Species Classification on CUB-200

**COS30082 Applied Machine Learning · Individual Coursework · PyTorch / Gradio**

An end-to-end fine-grained image classification project covering **200 bird species** in the Caltech–UCSD Birds 200 benchmark. The work explores convolutional networks trained from scratch, regularisation, transfer learning, modern ConvNeXt architecture, and explainable predictions with Grad-CAM.

**[Project Portal (Hugging Face)](https://huggingface.co/spaces/ShadWVege/cos30082-bird-species-explorer)** · **[Live E7 Classifier](https://7863-01m4vaa4swe6b7ha57xrh4ch7s.cloudspaces.litng.ai/)** · **[Complete Coursework Source](https://github.com/ShadWVege/cos30082-bird-classifier/tree/coursework-e7-source/coursework)** · **[E3 Backup Demo](https://cos30082-bird-classifier.onrender.com/)**

## Highlights

- **200-class bird recognition** with Top-5 species predictions and optional horizontal-flip test-time augmentation (TTA).
- **Seven experimental configurations** spanning scratch CNNs, regularisation and dropout ablations, ResNet18 transfer learning, and ConvNeXt-Tiny.
- **Two-stage fine-tuning:** five-epoch frozen-backbone head warmup followed by targeted backbone adaptation.
- **Explainable inference** using Grad-CAM to visualise class-associated regions of the photograph.
- **Deployment engineering:** lightweight ResNet18 on Render Free; higher-accuracy ConvNeXt-Tiny on a Lightning AI CPU Studio.

## Results

| Model | Parameters | Validation Top-1 | Official test Top-1 | Test avg. accuracy / class | Test Top-5 |
|---|---:|---:|---:|---:|---:|
| **E7 · ConvNeXt-Tiny** (primary demo) | 27.97M | **80.40%** | **80.73%** | **80.24%** | **95.18%** |
| **E3 · ResNet18** (Render backup) | 11.28M | 69.66% | 70.43% | 70.12% | 89.12% |

Results use **single-view inference**, except where explicitly stated otherwise. E7's optional two-view TTA achieved **80.32%** official test Top-1; E3's two-view TTA was assessed on validation only, not the official test set.

**Research-integrity note:** E3 was selected before the initial official test evaluation. E7 was developed after earlier test results had been examined and is therefore reported as an **exploratory post-test milestone**, not an independently blind pre-committed comparison.

The dataset contains 4,829 supplied training images (frozen development split: **3,870 training / 959 validation**) and **1,204 official held-out test images**. The project documents duplicate-image and conflicting-label audits, including the decision to keep identified training duplicates within the training partition.

## Live application and hosting

| Destination | Purpose | Notes |
|---|---|---|
| [Hugging Face Space](https://huggingface.co/spaces/ShadWVege/cos30082-bird-species-explorer) | **Primary project landing page** | Static HTML only — the CNN does **not** execute on Hugging Face |
| [Lightning AI · E7 Gradio](https://7863-01m4vaa4swe6b7ha57xrh4ch7s.cloudspaces.litng.ai/) | **Main interactive classifier** | ConvNeXt-Tiny, Top-5 predictions, benchmark explanations and `features.7` Grad-CAM |
| [Render Free · E3 Gradio](https://cos30082-bird-classifier.onrender.com/) | **Memory-efficient backup** | ResNet18 baseline designed for Render's 512 MiB memory limit |

The Lightning Studio sleeps when idle and has successfully been observed to wake on a public Gradio request (**~71 seconds in one test**). Please allow **1–2 minutes or longer** for a cold start. Free-tier services are best-effort and may be temporarily unavailable. The externally hosted inference is distinct from the Hugging Face landing page; this alternative hosting arrangement requires course-instructor approval.

## Where to find the code

This repository deliberately separates the working Render deployment from the complete experimental project:

| GitHub location | Contents |
|---|---|
| **[`main` branch](https://github.com/ShadWVege/cos30082-bird-classifier/tree/main)** | Deployed **E3 ResNet18** Gradio app, `model.pt`, `src/`, `render.yaml` and runtime requirements |
| **[`coursework-e7-source/coursework/`](https://github.com/ShadWVege/cos30082-bird-classifier/tree/coursework-e7-source/coursework)** | **Complete E1–E7 coursework source**: training, testing, experiment configurations, E7 deployment code, report source, figures and documentation |

For a quick technical review of the coursework:

| File | What it demonstrates |
|---|---|
| [`src/models.py`](https://github.com/ShadWVege/cos30082-bird-classifier/blob/coursework-e7-source/coursework/src/models.py) | Scratch CNN, pretrained ResNet18 and ConvNeXt-Tiny architectures |
| [`src/train.py`](https://github.com/ShadWVege/cos30082-bird-classifier/blob/coursework-e7-source/coursework/src/train.py) | Training configuration, staged fine-tuning and checkpoint selection |
| [`src/metrics.py`](https://github.com/ShadWVege/cos30082-bird-classifier/blob/coursework-e7-source/coursework/src/metrics.py) | Top-1, Top-5 and macro-averaged per-class recall |
| [`src/inference.py`](https://github.com/ShadWVege/cos30082-bird-classifier/blob/coursework-e7-source/coursework/src/inference.py) / [`src/gradcam.py`](https://github.com/ShadWVege/cos30082-bird-classifier/blob/coursework-e7-source/coursework/src/gradcam.py) | Prediction pipeline, optional TTA and Grad-CAM |
| [`tests/`](https://github.com/ShadWVege/cos30082-bird-classifier/tree/coursework-e7-source/coursework/tests) | **27 passing automated tests** (latest local verification) |
| [`reports/report.typ`](https://github.com/ShadWVege/cos30082-bird-classifier/blob/coursework-e7-source/coursework/reports/report.typ) | Source of the four-page academic report |
| [Code walkthrough guide](https://github.com/ShadWVege/cos30082-bird-classifier/blob/coursework-e7-source/coursework/docs/VIDEO_CODE_GUIDE.md) | Concise map of the implementation for reviewers and presentation |

### Running the Render E3 version locally

With a compatible Python environment, from the repository's **`main` branch**:

```bash
python -m pip install -r requirements.txt
python app.py
```

The `main` branch includes E3's deployment checkpoint. To reproduce the experimental E1–E7 pipeline, see the [coursework setup, dataset configuration and training instructions](https://github.com/ShadWVege/cos30082-bird-classifier/blob/coursework-e7-source/coursework/README.md).

**Reproducibility scope:** The coursework source publication omits raw dataset photographs, private environment files and E7 checkpoint weights; its model training and evaluation pipeline, configuration files, tests and report figures are included.

---

*Built for COS30082 Applied Machine Learning. This repository's default branch intentionally remains the E3 Render deployment branch to avoid disrupting the live backup service.*
