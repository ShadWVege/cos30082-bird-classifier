# Verified Deployment Architecture (October 2026)

**Coursework:** COS30082 Bird Species Classification — E7 ConvNeXt-Tiny primary classifier, E3 ResNet18 backup.

## Lecturer-facing links and actual execution locations

| Entry point | URL | What runs there |
|---|---|---|
| Hugging Face project Space | https://huggingface.co/spaces/ShadWVege/cos30082-bird-species-explorer | Free **Static HTML landing page only**, not PyTorch inference |
| Lightning AI E7 interactive classifier | https://7863-01m4vaa4swe6b7ha57xrh4ch7s.cloudspaces.litng.ai/ | **Actual E7 ConvNeXt-Tiny inference** on the Free CPU Studio via the Gradio plugin |
| Render E3 backup | https://cos30082-bird-classifier.onrender.com/ | **E3 ResNet18 inference** under the Render Free 512 MiB memory limit |
| Render E3 deployment source | https://github.com/ShadWVege/cos30082-bird-classifier | Deployment-only source on branch `main`; **not** the entire E7 training code |

**Instructor approval remains pending** for inference served outside Hugging Face. Do not describe the Static Space as executing E7 or as independently satisfying direct-HF-inference requirements.

## How E7 was deployed on Lightning AI

1. The standalone E7 bundle (code, figures, 200-class vocabulary, and locally verified `model.pt`) lives under `/teamspace/studios/this_studio/bird_classifier_v2/` in the existing Free 4-vCPU CPU Studio.
2. Lightning's **Gradio plugin** was registered with `bird_classifier_v2/gradio_plugin.py`, name `e7-bird-classifier`, port **7863**, **Auto start** enabled. The launcher exposes `demo = build_app()` from `app.py`.
3. The plugin's **Public link** created the verified **7863** URL. Do **not** substitute the previous **7860** Ports-plugin link; during sleep it returned HTTP 404 and did not wake the Studio.
4. The Studio `on_start.sh` now does nothing (`exit 0`) because the Gradio plugin manages the web process. A backup of the earlier 7860 startup script was retained remotely.
5. In a real test, the sleeping Studio woke following an external request and the 7863 endpoint responded HTTP **200** after **71.32 seconds**. The Gradio UI successfully performed bird prediction and `features.7` Grad-CAM. The Studio also subsequently entered sleep automatically after inactivity.

**Availability and cost:** These are observations from controlled tests, not uptime guarantees. Inform the evaluator to allow **1–2 minutes or longer** after opening the link. Free-plan active-session/billing boundaries remain relevant. No always-on paid deployment, automated keep-alive, or paid GPU endpoint was configured.

## Running the E7 bundle manually (developer fallback)

Run from its actual extracted directory, which must contain `app.py`, `model.pt`, `src/`, and metadata. Do not install a fresh environment if the Lightning Studio's `cloudspace` environment already has the dependencies.

**Linux / Lightning terminal:**

```bash
cd /teamspace/studios/this_studio/bird_classifier_v2
PORT=7860 GRADIO_SERVER_NAME=0.0.0.0 python app.py
```

**Windows PowerShell (from `deploy\e7_lightning` with checkpoint available):**

```powershell
$env:PORT = "7863"
python app.py
```

`app.py` reads `PORT` or `LIGHTNING_APP_PORT` rather than parsing a `--port` flag. Manual startup does **not** imply traffic-triggered Studio wake-up. To run with the Lightning Gradio plugin use the `gradio_plugin.py` entry point instead of `app.py`.

## Source-code ZIP versus running a model

The coursework source ZIP includes commented training/evaluation/inference source, configurations, scripts, the report authoring source, and visual figures. It **excludes** large model checkpoint `.pt` files and copyrighted image datasets. The source can be inspected and tested without those files, but pretrained inference requires an E7 checkpoint. The weights remain deployed on Lightning AI and are retained separately by the project owner.

## Historical alternative: Hugging Face CPU-backed Gradio Space

The earlier plan was to upload model weights and Gradio source directly to a CPU-backed Hugging Face Space. That plan could not be completed under the account's no-paid-compute constraints. This is **historical context only**, not the final deployment architecture. Do not follow older `outputs/huggingface_space` upload instructions expecting them to recreate the current public E7 link.
