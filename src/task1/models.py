from __future__ import annotations

import torch
from torch import nn
from torchvision.models import (
    ResNet18_Weights,
    ResNet34_Weights,
    resnet18,
    resnet34,
)
from torchvision.models.resnet import BasicBlock, ResNet


class SEBlock(nn.Module):
    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, _, _ = x.shape
        scale = self.pool(x).view(batch, channels)
        scale = self.fc(scale).view(batch, channels, 1, 1)
        return x * scale


class SEBasicBlock(BasicBlock):
    def __init__(self, *args, reduction: int = 16, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.se = SEBlock(self.conv2.out_channels, reduction=reduction)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


def _standard_resnet(name: str, pretrained: bool, num_classes: int) -> nn.Module:
    if name == "resnet18":
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        model = resnet18(weights=weights)
    elif name == "resnet34":
        weights = ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
        model = resnet34(weights=weights)
    else:
        raise ValueError(f"Unsupported model: {name}")
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def _se_resnet(name: str, pretrained: bool, num_classes: int) -> nn.Module:
    if name == "resnet18":
        layers = [2, 2, 2, 2]
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    elif name == "resnet34":
        layers = [3, 4, 6, 3]
        weights = ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
    else:
        raise ValueError(f"SE attention currently supports resnet18/resnet34, got {name}")

    model = ResNet(SEBasicBlock, layers)
    if weights is not None:
        state = weights.get_state_dict(progress=True)
        model.load_state_dict(state, strict=False)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def build_model(config: dict) -> nn.Module:
    model_cfg = config["model"]
    name = model_cfg["name"]
    pretrained = bool(model_cfg.get("pretrained", True))
    num_classes = int(model_cfg.get("num_classes", 102))
    attention = model_cfg.get("attention", "none")

    if attention == "none":
        model = _standard_resnet(name, pretrained, num_classes)
    elif attention == "se":
        model = _se_resnet(name, pretrained, num_classes)
    else:
        raise ValueError(f"Unsupported attention type: {attention}")

    if model_cfg.get("freeze_backbone", False):
        for param in model.parameters():
            param.requires_grad = False
        for param in model.fc.parameters():
            param.requires_grad = True
    return model

