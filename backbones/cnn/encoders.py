"""
ResNet RGB Encoder
==================
ResNet18/ResNet50 backbone wrapper with pretrained weight loading.
"""

from __future__ import annotations

import os
from collections import OrderedDict
from typing import List, Optional, Tuple

import torch
import torch.nn as nn

try:
    from torchvision.models import resnet18, resnet50
except Exception as exc:  # pragma: no cover - torchvision required
    raise ImportError("torchvision is required for ResNet backbones") from exc


class RGBEncoder(nn.Module):
    """
    RGB Encoder using ResNet backbone.

    Returns 4 feature maps (1/4, 1/8, 1/16, 1/32).
    """

    def __init__(
        self,
        depth: int = 18,
        pretrained: bool = True,
        weights_path: Optional[str] = None,
    ) -> None:
        super().__init__()

        if depth == 18:
            backbone = resnet18(weights=None)
            self.out_channels = [64, 128, 256, 512]
        elif depth == 50:
            backbone = resnet50(weights=None)
            self.out_channels = [256, 512, 1024, 2048]
        else:
            raise ValueError(f"Unsupported ResNet depth: {depth}")

        self.depth = depth
        self.backbone = backbone
        self.stem = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
        )
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        if pretrained and weights_path:
            self.load_pretrained_weights(weights_path)

    def load_pretrained_weights(self, path: str) -> None:
        if not os.path.exists(path):
            print(f"Warning: Pretrained weights not found at {path}")
            return
        print(f"Loading pretrained weights from {path}...")
        checkpoint = torch.load(path, map_location="cpu")
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        elif isinstance(checkpoint, dict) and "model" in checkpoint:
            state_dict = checkpoint["model"]
        else:
            state_dict = checkpoint

        cleaned = OrderedDict()
        for key, value in state_dict.items():
            new_key = key
            if new_key.startswith("module."):
                new_key = new_key[len("module.") :]
            cleaned[new_key] = value

        missing, unexpected = self.backbone.load_state_dict(cleaned, strict=False)
        if missing:
            print(f"  Missing keys: {len(missing)}")
        if unexpected:
            print(f"  Unexpected keys: {len(unexpected)}")

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        x = self.stem(x)
        f1 = self.layer1(x)
        f2 = self.layer2(f1)
        f3 = self.layer3(f2)
        f4 = self.layer4(f3)
        return [f1, f2, f3, f4]


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    enc = RGBEncoder(depth=18, pretrained=False).to(device)
    dummy = torch.randn(2, 3, 512, 512, device=device)
    feats = enc(dummy)
    for i, feat in enumerate(feats):
        print(f"Stage {i}: {feat.shape}")
