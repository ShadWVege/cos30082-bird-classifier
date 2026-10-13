"""Two comparable custom CNNs and a staged pretrained ResNet18."""
import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18


class CustomCNN(nn.Module):
    def __init__(self, num_classes, regularized=False, dropout=0.4):
        super().__init__()
        layers, incoming = [], 3
        for width in (32, 64, 128, 256):
            layers.append(nn.Conv2d(incoming, width, 3, padding=1, bias=not regularized))
            if regularized:
                layers.append(nn.BatchNorm2d(width))
            layers.extend([nn.ReLU(inplace=False), nn.MaxPool2d(2)])
            incoming = width
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout) if regularized else nn.Identity()
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, images):
        features = self.pool(self.features(images)).flatten(1)
        # Return logits: CrossEntropyLoss applies log-softmax internally.
        return self.classifier(self.dropout(features))


def build_model(config, pretrained=True):
    kind, classes = config["model"], config["num_classes"]
    if kind in ("baseline", "regularized"):
        return CustomCNN(classes, kind == "regularized", config.get("dropout", 0.4))
    if kind == "resnet18":
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        model = resnet18(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, classes)
        return model
    raise ValueError(f"Unknown model: {kind}")


def set_stage(model, kind, stage, finetune_blocks=("layer4",)):
    """Only the classifier then active blocks + classifier receive gradient updates."""
    if kind != "resnet18":
        return
    active_prefixes = tuple(f"{b}." for b in finetune_blocks) if stage == "finetune" else ()
    for name, parameter in model.named_parameters():
        parameter.requires_grad = name.startswith("fc.") or (stage == "finetune" and name.startswith(active_prefixes))


def training_mode(model, kind, stage, finetune_blocks=("layer4",), freeze_bn=False):
    model.train()
    if kind == "resnet18":
        # requires_grad=False alone DOES NOT freeze BatchNorm running statistics.
        # Start all backbone modules in eval mode, then train only the active blocks.
        model.eval()
        model.fc.train()
        if stage == "finetune":
            for b in finetune_blocks:
                block = getattr(model, b, None)
                if block is not None:
                    block.train()
            if freeze_bn:
                for m in model.modules():
                    if isinstance(m, nn.modules.batchnorm._BatchNorm):
                        m.eval()
