# COS30082 Assignment 1 — E7 ConvNeXt-Tiny Deployment Package

Standalone Gradio web application for our exploratory high-accuracy **Fine-Tuned ConvNeXt-Tiny (E7)** bird classifier on the CUB-200 benchmark.

---

## 1. Verified Model Performance & Key Metrics

- **Architecture:** Pretrained `convnext_tiny` with 200-class classification head (**`27,973,928`** parameters).
- **Training Strategy:** 5-epoch classification head warmup (155,336 trainable parameters, `lr=1e-3`) + 25-epoch fine-tuning of `features.7` and classifier (14,444,744 trainable parameters, `lr=5e-5` / `2e-4`, AdamW, weight decay `1e-4`, cosine-annealed plateau scheduler).
- **Official Held-Out Test Evaluation ($N=1,204$ held-out images):**
  - **Top-1 Accuracy:** **`80.73%`** (`972 / 1,204`)
  - **Average Accuracy per Class (Macro Recall):** **`80.24%`**
  - **Top-5 Accuracy:** **`95.18%`** (`1,146 / 1,204`)
  - **Macro-F1 Score:** **`79.94%`**
  - **Test Loss:** `0.759525`
  - **Paired McNemar Test vs E3 ResNet18 ($N=1,204$):**
    - Both models correct: **784** (65.1%)
    - E3 correct / E7 incorrect: **64** (5.3%)
    - E3 incorrect / E7 correct: **188** (15.6%)
    - Both models incorrect: **168** (14.0%)
    - Discordant pairs: **252** (188 vs 64)
    - Two-sided exact Binomial McNemar test: **$p = 2.48 \times 10^{-15}$** (Edwards continuity-corrected $\chi^2 = 60.04$, $df=1$, $p < 0.0001$), confirming that the **`+10.30 pp` gain** is statistically decisive.
- **Development Validation Set ($N=959$ images):**
  - Single-View: **`80.40%`** Top-1 &nbsp;|&nbsp; **`79.75%`** Avg/Class &nbsp;|&nbsp; **`96.35%`** Top-5 &nbsp;|&nbsp; Loss `0.6985`
  - 2-View TTA: **`81.02%`** Top-1 &nbsp;|&nbsp; **`80.35%`** Avg/Class &nbsp;|&nbsp; **`96.25%`** Top-5 &nbsp;|&nbsp; Loss `0.6874`
- **Test-Time Augmentation (TTA) Finding:**
  - On the official held-out test set, 2-view TTA achieved **`80.32%`** Top-1, which is slightly lower than single-view deterministic center-crop (**`80.73%`**). Single-view inference is therefore the primary deployment mode.

---

## 2. Directory Contents

```
deploy/e7_lightning/
├── app.py                 # Gradio application (3 tabs: Inference, Benchmark, Architecture)
├── gradio_plugin.py       # Lightning Gradio Plugin entrypoint exposing demo = build_app()
├── model.pt               # Exported CPU inference checkpoint (106.8 MiB, state_dict identical to best.pt)
├── class_names.json       # Exact 200 bird species names (000–199)
├── model_info.json        # Detailed provenance, epoch 27, SHA-256 hashes, validation & test scores
├── requirements.txt       # Lightweight CPU-only PyTorch, Torchvision, Gradio, Pillow, NumPy
├── README.md              # This guide
├── src/                   # Standalone inference modules (zero external library dependency)
│   ├── __init__.py
│   ├── dataset.py         # Shared deterministic preprocessing (256-resize -> 224-crop)
│   ├── engine.py          # Checkpoint loader with mmap support
│   ├── inference.py       # Predictor with meta-device loading & isolated features.7 Grad-CAM
│   ├── metrics.py         # Top-1, Avg/Class, Top-5, Macro-F1 metrics
│   ├── models.py          # ConvNeXt-Tiny architecture builder
│   └── utils.py           # Logging and JSON helpers
└── assets/                # Benchmark figures for Section 3
    ├── ablation_e3_vs_e7.png
    ├── gradcam_focused.png
    ├── model_comparison.png
    └── test_metrics_summary.png
```

---

## 3. Running Locally (Windows, macOS, or Linux)

1. **Activate your Python 3.10+ virtual environment:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

2. **Install the lightweight CPU requirements:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Launch the application:**
   ```bash
   # Linux/macOS (run from deploy/e7_lightning/)
   PORT=7863 python app.py
   ```
   ```powershell
   # Windows PowerShell (run from deploy\e7_lightning)
   $env:PORT = "7863"
   python app.py
   ```
   Open your browser to `http://127.0.0.1:7863`. `app.py` reads the `PORT` or `LIGHTNING_APP_PORT` environment variable; it does **not** parse a `--port` command-line flag.

---

## 4. Deploying on Lightning AI Free CPU Studio (Step-by-Step)

Lightning AI offers a Free CPU Studio, subject to free-session limits and possible billing/credit consumption if active outside the free allowance:

### Step 1: Create a Free Studio
1. Sign in to [Lightning AI](https://lightning.ai/) with your account.
2. Select the **Free CPU** hardware option (4 vCPUs, ~16 GB RAM, 50 GB persistent storage).

### Step 2: Upload the Deployment Files
Upload the deployment bundle containing `app.py`, `gradio_plugin.py`, `src/`, `assets/`, `class_names.json`, `model_info.json`, `requirements.txt`, and the verified **E7 `model.pt` checkpoint**. The coursework source ZIP deliberately excludes checkpoints and cannot serve inference by itself.

Extract the deployment bundle and verify that all files are located together under a directory such as `/teamspace/studios/this_studio/bird_classifier_v2/`. On the existing Studio these files are already installed; avoid reinstalling packages needlessly.

### Step 3: Check the Existing Environment
```bash
cd /teamspace/studios/this_studio/bird_classifier_v2
python -c "import torch, gradio; print(torch.__version__, gradio.__version__)"
ls -l app.py gradio_plugin.py model.pt class_names.json
```
Use the already-configured single `cloudspace` Conda environment where possible. Install missing dependencies only if needed; creating a second environment may be blocked on the Free Studio.

### Step 4: Run the Application or Use Gradio Plugin
Option A (Direct Execution; no wake-on-request guarantee):
```bash
cd /teamspace/studios/this_studio/bird_classifier_v2
PORT=7860 GRADIO_SERVER_NAME=0.0.0.0 python app.py
```
This starts a manual development server on port 7860; **it is not the final lecturer-facing public link**.

Option B (**verified final hosting method**):
1. Open the existing Studio's **Gradio** plugin, then **New App**.
2. Select `bird_classifier_v2/gradio_plugin.py`, name the app `e7-bird-classifier`, and assign port **7863**.
3. Check **Auto start** and click **Run**. The plugin invokes `GRADIO_SERVER_PORT=7863 gradio bird_classifier_v2/gradio_plugin.py`.
4. Use its **Public link** button and confirm the URL below in an Incognito browser.
5. Retain Lightning's **Auto-sleep when idle** setting (tested at 10 minutes). The `on_start.sh` hook was left as `exit 0` to avoid starting a duplicate E7 process on port 7860; the Gradio plugin now manages E7.

**Final public E7 link:** `https://7863-01m4vaa4swe6b7ha57xrh4ch7s.cloudspaces.litng.ai/`.

A sleeping Studio successfully woke on a public Gradio-plugin request in one controlled test, returning HTTP 200 in **71.32 s**. It later returned to sleep after inactivity. These are observations, **not guarantees** of response time, unattended uptime, or zero-credit usage beyond the Free plan terms.

---

## 5. Coursework Hosting Ecosystem & Operational Boundaries

The project utilizes a resilient 3-tier cloud architecture designed to satisfy grading requirements while managing free-tier compute boundaries:

### Tier 1: Primary Project Landing Page (Hugging Face Space)
- **URL:** [https://huggingface.co/spaces/ShadWVege/cos30082-bird-species-explorer](https://huggingface.co/spaces/ShadWVege/cos30082-bird-species-explorer)
- **Role:** Central project portal, methodology summary, benchmark charts, and direct navigation links.
- **Important Disclosure:** Hugging Face hosts the **static HTML landing page and documentation only**. PyTorch CNN model inference is intentionally not run on Hugging Face due to free-tier hardware limits and memory constraints.

### Tier 2: Primary Interactive E7 Classifier (Lightning AI Free CPU Studio)
- **URL:** [https://7863-01m4vaa4swe6b7ha57xrh4ch7s.cloudspaces.litng.ai/](https://7863-01m4vaa4swe6b7ha57xrh4ch7s.cloudspaces.litng.ai/)
- **Role:** Interactive high-accuracy inference demonstration running the exploratory **E7 ConvNeXt-Tiny** model (80.73% test Top-1, 27.97M parameters, isolated `features.7` Grad-CAM explainability).
- **Compute Environment:** 4 vCPUs, 16 GB RAM on Lightning AI Free CPU Studio.
- **Operational Dynamics:**
  - **Auto-Sleep:** The Studio enters sleep mode during periods of inactivity ($0 compute consumption).
  - **Cold-Start Wake:** A request to the **registered Gradio-plugin public URL on port 7863** successfully woke the sleeping Studio in one controlled test (**71.32 seconds**). The earlier **ordinary Ports-plugin link on 7860 did not wake** the Studio. Budget 1–2 minutes or longer for cold starts; behaviour is best effort.
  - **Free-plan boundary:** A 4-hour active-session restart requirement has been documented for Free Studios. **Auto-sleep does not by itself guarantee a restart before the billing boundary or indefinite RM0 hosting.** Verify Studio status and credit balance; manually stop the Studio when necessary.

### Tier 3: Pre-Committed Production Backup (Render Free Web Service)
- **URL:** [https://cos30082-bird-classifier.onrender.com/](https://cos30082-bird-classifier.onrender.com/)
- **Role:** Resilient fallback web application serving the pre-committed **E3 ResNet18** model (70.43% single-view official test Top-1, 11.28M parameters; optional 2-view TTA toggle). Official TTA test performance was not evaluated.
- **Memory Safety:** Operates safely within Render's strict 512 MiB Linux cgroup limit (peak `VmHWM` = 380.7 MiB, providing 131 MiB of true physical headroom).
- **Deployment Source:** [https://github.com/ShadWVege/cos30082-bird-classifier](https://github.com/ShadWVege/cos30082-bird-classifier) (inference-only deployment repository).
- **Uptime:** Standard free-tier idle spin-down after 15 minutes of inactivity (~50 s cold start).
