# Colab / Kaggle training

The project is ordinary Python, so the same entry points work locally and in notebooks. Use a GPU runtime. Transfer project source plus annotation/split files to the runtime. Place the provided training images in Google Drive or a private Kaggle dataset; preserve the official dataset and do not download another CUB version.

In a notebook cell, change into the project root, then install the small additional dependencies. Use the runtime's existing matched GPU torch/torchvision pair if it satisfies requirements; inspect `python -m src.doctor` before training.

```python
%cd /content/COS30082_BirdClassification
!python -m pip install -r requirements-dev.txt
!python -m src.doctor
```

Write a cloud data config with absolute image paths:

```python
import json
from pathlib import Path
config = json.loads(Path('configs/data.example.json').read_text())
config['train_root'] = '/content/drive/MyDrive/BirdData/Train'
config['test_root'] = '/content/drive/MyDrive/BirdData/Test'
Path('configs/data.cloud.json').write_text(json.dumps(config, indent=2))
```

Reuse `data/prepared/split.json`. Its image paths are relative to the supplied root; if you change the extraction layout, set the root so those relative paths still resolve. Do not independently generate a new split on each platform.

```python
!python -m src.train --config configs/baseline.json --data configs/data.cloud.json --out /content/drive/MyDrive/BirdRuns/baseline_001 --device cuda
```

Write run outputs to persistent storage so a disconnected notebook does not lose checkpoints. Repeat for regularized and resnet18; or use `src.run_experiments` with a persistent output directory. Windows defaults to zero workers for reliability; on cloud, a separate config can try two workers after testing. Document any change. Resume with the identical config, split and command plus `--resume`.

Do not run final test evaluation in a tuning loop. Copy all run summaries, best/last checkpoints, curves, environment records and frozen selection back to the local project after training. Absolute checkpoint paths in `selection.json` identify the originating environment; freeze the selection on the environment where evaluation/export will run, before accessing test data.
