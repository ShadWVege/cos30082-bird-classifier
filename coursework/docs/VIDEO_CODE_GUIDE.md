# COS30082 — Code Walkthrough Guide for the ≤10-Minute Video

The assignment asks you to **present and explain your code** as well as show the application. Show **4 short code locations**, not all 96 ZIP files. Use the actual Python files and highlight the lines indicated by the symbols below.

| Suggested screen time | File and symbols | What to say / show |
|---|---|---|
| 30–40 seconds | `src/models.py`: `build_model()`, `set_stage()` | Compare 4-convolution scratch CNN, pretrained ResNet18, and E7 ConvNeXt-Tiny. Show the 200-class replacement classifier and selective unfreezing of `features.7`. |
| 35–45 seconds | `src/train.py`: `train()`, `src/engine.py`: `run_epoch()` | Explain 5-epoch frozen-head warmup, then final-block fine-tuning, AdamW and validation-based checkpoint selection. The official test set is not used to tune the pipeline. |
| 25–35 seconds | `src/metrics.py`: `classification_metrics()` | Explain **Top-1**, per-class recall averaged over 200 species (**Average Accuracy per Class**), and optional Top-5. Distinguish validation and official test results. |
| 35–45 seconds | `src/inference.py`: `Predictor`, `predict()`; `deploy/e7_lightning/app.py`: `build_app()` | Describe image normalization, single-view inference, optional two-view TTA, Top-5 probabilities, and `features.7` Grad-CAM. Say heatmaps are post-hoc spatial attributions, not causal proof. |

**Suggested total code walkthrough:** 2–3 minutes; allow ~3 minutes for results and model comparison, ~2 minutes for the actual application/HF Space demonstration, and the remainder for dataset integrity, intro, limitations and conclusion. Stay **below 10 minutes** including screen switching.

## Live demo procedure

1. Open the **Hugging Face static project landing page**: https://huggingface.co/spaces/ShadWVege/cos30082-bird-species-explorer
2. Follow its button to the **Lightning E7 Gradio application**: https://7863-01m4vaa4swe6b7ha57xrh4ch7s.cloudspaces.litng.ai/
3. **Before recording**, open/wake the application; a cold start took **71.32 seconds** in one observed test. Do not claim it runs directly on Hugging Face.
4. Upload one example bird photo, keep **2-view TTA OFF** for the standard E7 single-view path, and show a Top-5 prediction and Grad-CAM.
5. If you mention Render, label https://cos30082-bird-classifier.onrender.com/ explicitly as the **E3 ResNet18 backup** (70.43% *single-view* official test Top-1). E3 TTA has no official test score.

## Accuracy statements for narration

- E7 ConvNeXt-Tiny official test Top-1 **80.73%**, Average Accuracy per Class **80.24%**, Top-5 **95.18%** (1,204 test images, single-view).
- E7 two-view TTA official test Top-1 **80.32%** (slightly below single-view); the checkbox is optional.
- E3 ResNet18 single-view official test Top-1 **70.43%**; this was the pre-committed model. E7 was developed after earlier official-test inspection and is therefore an **exploratory post-test comparison**.

## Student hand-in checklist

- Four-page report PDF with name, ID, Hugging Face URL, explanation of external inference, and **real** YouTube video URL when recorded.
- One source ZIP with commented Python source, tests, configs, figures, and report source; checkpoint binaries excluded and kept separately.
- Working HF Static Space and Lightning E7 link, plus Render E3 backup.
- Do not claim lecturer approval until a reply is received.
