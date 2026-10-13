---
title: Bird Species Explorer
emoji: 🐦
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: 6.29.1
app_file: app.py
python_version: '3.12'
pinned: false
---

# COS30082 Bird Species Classification (CUB-200)

200-class North American bird species classifier (`000–199`) powered by our validation-selected **Staged ResNet18 Transfer Learning (`E3`)** model.

- **Validation Performance (959 images):** `69.66%` Top-1 Accuracy | `69.10%` Average Accuracy per Class | `89.57%` Top-5 Accuracy.
- **Official Held-Out Test Performance (1,204 images):** `70.43%` Top-1 Accuracy | `70.12%` Average Accuracy per Class | `89.12%` Top-5 Accuracy | `69.90%` Macro-F1.

See `model_info.json` for checkpoint provenance and SHA-256 verification. Softmax scores sum to 100% across the 200 closed-set classes and are not calibrated confidence. Compatible with both **Render Free Web Service** (`render.yaml`, 512 MiB RAM CPU profile) and **Hugging Face Spaces** (`app.py`). No dataset archives or raw training/test photographs are distributed in this package.
