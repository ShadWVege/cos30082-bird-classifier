"""Lightning AI Gradio Plugin Entrypoint.

Exposes the top-level `demo` object for Lightning Studio's Gradio Plugin.
Launches the full E7 ConvNeXt-Tiny application on the assigned plugin port (default: 7863).
"""
import os
import sys
from pathlib import Path

# Ensure deploy/e7_lightning root is in Python module search path
APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from app import build_app

# Lightning Gradio plugin expects a top-level `demo` gr.Blocks instance
demo = build_app()

if __name__ == "__main__":
    port = int(os.environ.get("LIGHTNING_APP_PORT", os.environ.get("PORT", "7863")))
    host = os.environ.get("GRADIO_SERVER_NAME", "0.0.0.0")
    print(f"Launching Lightning Gradio Plugin on {host}:{port}...")
    demo.launch(server_name=host, server_port=port)
