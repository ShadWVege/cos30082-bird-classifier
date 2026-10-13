"""Shared preprocessing for training, evaluation and deployment."""
from pathlib import Path
from PIL import Image, ImageOps
from torch.utils.data import Dataset
from torchvision import transforms as T

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def image_transform(size=224, augment=False):
    # ImageNet normalization matches the pretrained ResNet18 V1 weights.
    # Validation, test and app all call this exact deterministic branch.
    if augment:
        steps = [T.RandomResizedCrop(size, scale=(0.7, 1.0)), T.RandomHorizontalFlip(),
                 T.RandomRotation(10), T.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1)]
    else:
        steps = [T.Resize(round(size * 256 / 224)), T.CenterCrop(size)]
    return T.Compose(steps + [T.ToTensor(), T.Normalize(MEAN, STD)])


class BirdDataset(Dataset):
    def __init__(self, root, records, size=224, augment=False):
        self.root, self.records = Path(root), records
        self.transform = image_transform(size, augment)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        row = self.records[index]
        with Image.open(self.root / row["relative_path"]) as im:
            image = ImageOps.exif_transpose(im).convert("RGB")
            tensor = self.transform(image)
        return tensor, row["label"], row["name"]
