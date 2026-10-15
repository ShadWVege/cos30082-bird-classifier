

# COS30082 Bird Species Classification (CUB-200)

## Full coursework project and E7 ConvNeXt-Tiny demo

**Primary E7 model:** Fine-tuned ConvNeXt-Tiny, 200 bird species, **80.73% official test Top-1**, **80.24% Average Accuracy per Class**, and **95.18% Top-5** (single-view inference; exploratory post-test milestone).

- **Hugging Face project page:** https://huggingface.co/spaces/ShadWVege/cos30082-bird-species-explorer — Static HTML landing page only.
- **Live E7 inference and Grad-CAM:** https://7863-01m4vaa4swe6b7ha57xrh4ch7s.cloudspaces.litng.ai/ — Lightning AI Free CPU Studio; allow approximately 1–2 minutes or longer for a sleeping Studio to wake.
- **E3 ResNet18 backup:** https://cos30082-bird-classifier.onrender.com/ — **70.43% single-view official test Top-1**. The E3 two-view TTA test metric was not evaluated.

**Repository scope:** Branch `main` intentionally retains the working **E3/Render deployment source and checkpoint**. The full annotated E7 coursework code is being prepared on a separate `coursework-e7-source` branch. Do not confuse this E3 deployment implementation with the E7 model running on Lightning; neither inference model runs directly on the Hugging Face static landing page.

---


200-class North American bird species classifier (`000–199`) powered by our validation-selected **Staged ResNet18 Transfer Learning (`E3`)** model.

- **Validation Performance (959 images):** `69.66%` Top-1 Accuracy | `69.10%` Average Accuracy per Class | `89.57%` Top-5 Accuracy.
- **Official Held-Out Test Performance (1,204 images):** `70.43%` Top-1 Accuracy | `70.12%` Average Accuracy per Class | `89.12%` Top-5 Accuracy | `69.90%` Macro-F1.

See `model_info.json` for checkpoint provenance and SHA-256 verification. Softmax scores sum to 100% across the 200 closed-set classes and are not calibrated confidence. Compatible with both **Render Free Web Service** (`render.yaml`, 512 MiB RAM CPU profile) and **Hugging Face Spaces** (`app.py`). No dataset archives or raw training/test photographs are distributed in this package.
