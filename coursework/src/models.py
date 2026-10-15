"""Convolutional neural network architectures for CUB-200 fine-grained classification.

Supported Models:
- CustomCNN Baseline (E1): 4 Conv-ReLU-Pool stages, global average pooling (439,816 parameters).
- CustomCNN Regularized (E2): Added BatchNorm2d + Dropout 0.4 (440,296 parameters).
- CustomCNN Dropout Ablation (E4): Same as E2 but with Dropout 0.0 (440,296 parameters).
- ResNet-18 Transfer Learning (E3/E5/E6): ImageNet-1K pretrained backbone (11,279,112 parameters).
- ConvNeXt-Tiny Transfer Learning (E7): Modern depthwise-separable ConvNet (27,973,928 parameters).
"""
import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18


class CustomCNN(nn.Module):
    """Custom 4-stage convolutional neural network designed from first principles.

    Learning Objectives & Design Rationale:
    1. Hierarchical Feature Extraction: Four Conv-ReLU-Pool blocks progressively double
       channel capacity (32 -> 64 -> 128 -> 256) while halving spatial resolution
       (224x224 -> 112x112 -> 56x56 -> 28x28 -> 14x14).
    2. Kernel & Padding: 3x3 kernels with padding=1 preserve spatial boundaries within
       each block prior to explicit downsampling via MaxPool2d(2).
    3. Global Average Pooling (GAP): AdaptiveAvgPool2d(1) collapses the final 14x14x256
       feature volume into a 256-D vector. Flattening directly would require a massive
       linear layer (14*14*256 * 200 = 10,035,200 weights), causing extreme overfitting.
       GAP limits the linear classifier to just 256 * 200 + 200 = 51,400 weights.
    4. Conv Bias Omission & Parameter Counts: When BatchNorm is enabled (regularized=True),
       Conv2d bias is set to False (bias=not regularized) because BatchNorm's learnable
       affine shift (beta) mathematically subsumes any constant bias offset:
       BN(W*x + b) == BN(W*x).
       - Baseline E1: 387,936 (conv weights) + 480 (conv biases) + 51,400 (classifier) = 439,816.
       - Regularized E2/E4: 387,936 (conv weights) + 0 (conv biases) + 960 (BN gamma+beta)
         + 51,400 (classifier) = 440,296 parameters.
       - Historical Note: Early coursework drafts cited 440,776 by adding BN (960) directly
         to E1 without subtracting the 480 omitted conv biases (439,816 + 960 = 440,776).
         The actual trained checkpoints strictly instantiate 440,296 parameters.
    5. Inplace=False: Preserves input tensors for autograd integrity and hook inspection.
    """

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
        # Return unnormalized logits: PyTorch CrossEntropyLoss computes log-softmax
        # internally via the LogSumExp trick for maximal numerical stability.
        return self.classifier(self.dropout(features))


def build_model(config, pretrained=True):
    """Build candidate architectures for CUB-200 fine-grained visual classification.

    Design Rationale & Classifier Surgery:
    - Baseline & Regularized: CustomCNN with 4 stages, global pooling, and 200-way linear head.
    - ResNet-18 (E3): Replaces the 1,000-way ImageNet head (model.fc) with nn.Linear(512, 200).
      Total parameters: 11,279,112.
    - ConvNeXt-Tiny (E7): Replaces the 1,000-way linear layer inside model.classifier[2]
      with nn.Linear(768, 200), retaining classifier[0] (LayerNorm2d(768)) and classifier[1]
      (Flatten). Total parameters: 27,973,928.
    - EfficientNetV2-S: Replaces model.classifier[1] with nn.Linear(1280, 200), preserving
      the pre-classifier dropout regularizer.
    """
    kind, classes = config["model"], config["num_classes"]
    if kind in ("baseline", "regularized"):
        return CustomCNN(classes, kind == "regularized", config.get("dropout", 0.4))
    if kind == "resnet18":
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        model = resnet18(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, classes)
        return model
    if kind == "convnext_tiny":
        from torchvision.models import ConvNeXt_Tiny_Weights, convnext_tiny
        weights = ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
        model = convnext_tiny(weights=weights)
        in_features = model.classifier[2].in_features
        model.classifier[2] = nn.Linear(in_features, classes)
        return model
    if kind == "efficientnet_v2_s":
        from torchvision.models import EfficientNet_V2_S_Weights, efficientnet_v2_s
        weights = EfficientNet_V2_S_Weights.IMAGENET1K_V1 if pretrained else None
        model = efficientnet_v2_s(weights=weights)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, classes)
        return model
    if kind == "convnext_small":
        from torchvision.models import ConvNeXt_Small_Weights, convnext_small
        weights = ConvNeXt_Small_Weights.IMAGENET1K_V1 if pretrained else None
        model = convnext_small(weights=weights)
        in_features = model.classifier[2].in_features
        model.classifier[2] = nn.Linear(in_features, classes)
        return model
    raise ValueError(f"Unknown model: {kind}")


def set_stage(model, kind, stage, finetune_blocks=("layer4",)):
    """Configure gradient tracking across staged transfer learning.

    Design choice:
    - Stage 'head': Freeze all pretrained backbone representations; compute gradients only
      for the newly attached 200-way classification head to avoid corrupting pretrained features.
    - Stage 'finetune': Unfreeze only the final convolutional block (e.g., layer4 in ResNet18,
      features.7 in ConvNeXt, or features.7/6 in EfficientNetV2) alongside the classifier,
      preventing catastrophic over-parameterisation on small datasets.
    """
    if kind == "resnet18":
        active_prefixes = tuple(f"{b}." for b in finetune_blocks) if stage == "finetune" else ()
        for name, parameter in model.named_parameters():
            parameter.requires_grad = name.startswith("fc.") or (stage == "finetune" and name.startswith(active_prefixes))
    elif kind in ("convnext_tiny", "convnext_small"):
        active_blocks = finetune_blocks if finetune_blocks != ("layer4",) else ("features.7",)
        active_prefixes = tuple(f"{b}." for b in active_blocks) if stage == "finetune" else ()
        for name, parameter in model.named_parameters():
            parameter.requires_grad = name.startswith("classifier.") or (stage == "finetune" and name.startswith(active_prefixes))
    elif kind == "efficientnet_v2_s":
        active_blocks = finetune_blocks if finetune_blocks != ("layer4",) else ("features.7", "features.6")
        active_prefixes = tuple(f"{b}." for b in active_blocks) if stage == "finetune" else ()
        for name, parameter in model.named_parameters():
            parameter.requires_grad = name.startswith("classifier.") or (stage == "finetune" and name.startswith(active_prefixes))


def training_mode(model, kind, stage, finetune_blocks=("layer4",), freeze_bn=False):
    """Set modules to train/eval mode according to transfer stage.

    Design choice:
    - In ResNet18 and EfficientNetV2, requires_grad=False alone DOES NOT freeze BatchNorm
      running mean/variance updates. Starting backbones in eval() preserves ImageNet statistics.
    - In ConvNeXt, LayerNorm computes statistics dynamically across channels per instance,
      eliminating running-statistic drift. Placing features in eval() during head warmup
      additionally keeps stochastic depth (DropPath) disabled for training stability.
    """
    model.train()
    if kind == "resnet18":
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
    elif kind in ("convnext_tiny", "convnext_small"):
        model.eval()
        model.classifier.train()
        if stage == "finetune":
            active_blocks = finetune_blocks if finetune_blocks != ("layer4",) else ("features.7",)
            for name, module in model.named_modules():
                if any(name.startswith(b) for b in active_blocks):
                    module.train()
    elif kind == "efficientnet_v2_s":
        model.eval()
        model.classifier.train()
        if stage == "finetune":
            active_blocks = finetune_blocks if finetune_blocks != ("layer4",) else ("features.7", "features.6")
            for name, module in model.named_modules():
                if any(name.startswith(b) for b in active_blocks):
                    module.train()
            if freeze_bn:
                for m in model.modules():
                    if isinstance(m, nn.modules.batchnorm._BatchNorm):
                        m.eval()

