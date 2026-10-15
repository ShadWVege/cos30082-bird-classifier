"""Print environment/device information without touching the test images."""
import importlib.metadata
import json
import platform
import sys


def main():
    info = {"python": sys.version, "executable": sys.executable, "platform": platform.platform()}
    for name in ("torch", "torchvision", "Pillow", "numpy", "matplotlib", "gradio"):
        try:
            info[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            info[name] = "MISSING"
    try:
        import torch
        info["cuda_available"] = torch.cuda.is_available()
        info["torch_cuda_runtime"] = torch.version.cuda
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
            info["gpu_memory_gib"] = torch.cuda.get_device_properties(0).total_memory / 1024**3
    except ImportError:
        pass
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
