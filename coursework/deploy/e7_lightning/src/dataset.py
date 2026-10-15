"""Shared image preprocessing and PyTorch Dataset for CUB-200 bird classification.

Learning Objectives & Design Rationale:
- ImageNet Normalization: Pretrained backbones (ResNet-18, ConvNeXt-Tiny) expect input
  RGB channels standardized to ImageNet statistics (mean=[0.485, 0.456, 0.406],
  std=[0.229, 0.224, 0.225]). Applying identical normalization across all splits
  guarantees representation alignment.
- Evaluation Determinism: Validation, held-out test, and production Gradio apps share
  the exact same deterministic transform pipeline (Resize to 256 followed by CenterCrop
  to 224x224), preventing data distortion while matching standard benchmark protocols.
- Training Regularization: Data augmentations (RandomResizedCrop, RandomHorizontalFlip,
  mild rotation, ColorJitter) simulate natural photographic variation while preserving
  bilateral symmetry and fine plumage markings.
"""
from pathlib import Path
from PIL import Image, ImageOps
from torch.utils.data import Dataset
from torchvision import transforms as T

# Standard ImageNet RGB channel mean and standard deviation.
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def image_transform(size=224, augment=False):
    """Construct torchvision transform pipeline for training or evaluation.

    Args:
        size (int): Target square crop size (default: 224).
        augment (bool): If True, applies stochastic training augmentations.
            If False, applies deterministic resize + center crop.

    Design Rationale:
    - Deterministic branch: Resizes the smaller edge to 256 (via round(size * 256 / 224))
      before center-cropping to 224x224. This preserves the original aspect ratio without
      geometric stretching, centering the subject bird in the field of view.
    - Augmentation branch: Uses a conservative scale range (0.7 to 1.0) and mild rotation
      (+-10 degrees) because birds possess delicate diagnostic features (beak shape, wing bars);
      excessive distortion could destroy discriminative fine-grained plumage details.
    """
    if augment:
        steps = [
            T.RandomResizedCrop(size, scale=(0.7, 1.0)),
            T.RandomHorizontalFlip(),
            T.RandomRotation(10),
            T.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
        ]
    else:
        steps = [
            T.Resize(round(size * 256 / 224)),
            T.CenterCrop(size),
        ]
    return T.Compose(steps + [T.ToTensor(), T.Normalize(MEAN, STD)])


class BirdDataset(Dataset):
    """PyTorch Dataset loading CUB-200 bird images from audit records.

    Each item is a dictionary containing:
        - 'relative_path': Relative file path from dataset root.
        - 'label': Zero-indexed integer class label (0 to 199).
        - 'name': Unique image identifier (e.g., 'Red_headed_Woodpecker_0012_2578571305.jpg').

    Learning Objective:
    Demonstrates proper EXIF handling and RGB conversion in computer vision pipelines.
    Smartphone and DSLR cameras often embed orientation tags in EXIF metadata; without
    `ImageOps.exif_transpose`, rotated images would corrupt both feature extraction
    and spatial Grad-CAM attribution.
    """

    def __init__(self, root, records, size=224, augment=False):
        self.root, self.records = Path(root), records
        self.transform = image_transform(size, augment)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        row = self.records[index]
        with Image.open(self.root / row["relative_path"]) as im:
            # Correct orientation from EXIF tags and ensure 3-channel RGB format
            image = ImageOps.exif_transpose(im).convert("RGB")
            tensor = self.transform(image)
        return tensor, row["label"], row["name"]

