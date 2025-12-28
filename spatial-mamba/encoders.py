"""
Spatial-Mamba RGB Encoder
=========================
RGBEncoder: Spatial-Mamba backbone wrapper with pretrained weight loading.
Uses the original Spatial-Mamba model without bypass or fallback paths.
"""

from __future__ import annotations

import os
import sys
import importlib.util
from collections import OrderedDict
from typing import List, Tuple, Optional

import torch
import torch.nn as nn


SPATIAL_MAMBA_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "Spatial-Mamba")
)
SPATIAL_MAMBA_CLASSIFICATION = os.path.join(SPATIAL_MAMBA_ROOT, "classification")
SPATIAL_MAMBA_MODELS = os.path.join(SPATIAL_MAMBA_CLASSIFICATION, "models")
if SPATIAL_MAMBA_CLASSIFICATION not in sys.path:
    sys.path.insert(0, SPATIAL_MAMBA_CLASSIFICATION)
if SPATIAL_MAMBA_MODELS not in sys.path:
    sys.path.insert(0, SPATIAL_MAMBA_MODELS)


class RGBEncoder(nn.Module):
    """
    RGB Encoder using the original Spatial-Mamba backbone.

    Variant selection is controlled by depths/dims and extra SSM settings.
    Returns multi-scale features from 4 stages.
    """

    def __init__(
        self,
        depths: Tuple[int, ...] = (2, 4, 21, 5),
        dims: Tuple[int, ...] = (64, 128, 256, 512),
        drop_path_rate: float = 0.3,
        pretrained: bool = True,
        weights_path: Optional[str] = None,
        model_variant: str = "small",
        d_state: int = 1,
        dt_init: str = "random",
        mlp_ratio: float = 4.0,
    ) -> None:
        super().__init__()

        self.depths = depths
        self.dims = dims
        self.out_channels = list(dims)
        self.model_variant = model_variant.lower()

        resolved_weights = None
        if pretrained and weights_path:
            if os.path.isfile(weights_path):
                resolved_weights = weights_path
            else:
                print(f"Warning: Pretrained weights path is not a file: {weights_path}")

        try:
            spatialmamba_path = os.path.join(SPATIAL_MAMBA_MODELS, "spatialmamba.py")
            spec = importlib.util.spec_from_file_location("spatialmamba", spatialmamba_path)
            if spec is None or spec.loader is None:
                raise ImportError(f"Failed to load Spatial-Mamba module from {spatialmamba_path}")
            spatialmamba = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(spatialmamba)
            Backbone_SpatialMamba = spatialmamba.Backbone_SpatialMamba
        except Exception as exc:
            raise ImportError(
                "Spatial-Mamba is required for the RGB encoder. "
                "Check that Spatial-Mamba/classification/models is available."
            ) from exc

        self.backbone = Backbone_SpatialMamba(
            out_indices=(0, 1, 2, 3),
            pretrained=None,
            depths=list(depths),
            dims=list(dims),
            drop_path_rate=drop_path_rate,
            d_state=d_state,
            dt_init=dt_init,
            mlp_ratio=mlp_ratio,
            patch_norm=True,
            norm_layer="ln",
        )

        if pretrained and resolved_weights:
            self.load_pretrained_weights(resolved_weights)

    def load_pretrained_weights(self, path: str) -> None:
        """Load pretrained weights into Spatial-Mamba backbone."""
        if not os.path.exists(path):
            print(f"Warning: Pretrained weights not found at {path}")
            return

        print(f"Loading pretrained weights from {path}...")
        if hasattr(self.backbone, "load_pretrained"):
            self.backbone.load_pretrained(path)
            return

        checkpoint = torch.load(path, map_location="cpu")
        if "model" in checkpoint:
            state_dict = checkpoint["model"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint

        cleaned_state_dict = OrderedDict()
        for key, value in state_dict.items():
            new_key = key
            if new_key.startswith("module."):
                new_key = new_key[len("module."):]
            if any(skip in new_key for skip in ("head.", "classifier.", "fc.")):
                continue
            cleaned_state_dict[new_key] = value

        missing, unexpected = self.backbone.load_state_dict(cleaned_state_dict, strict=False)
        if missing:
            print(f"  Missing keys: {len(missing)}")
        if unexpected:
            print(f"  Unexpected keys: {len(unexpected)}")

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Args:
            x: (N, 3, H, W) RGB input
        Returns:
            List of 4 feature maps at different scales.
        """
        features = self.backbone(x)
        if isinstance(features, (list, tuple)):
            return list(features)
        return [features]


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\nTesting RGBEncoder (Spatial-Mamba)...")
    rgb_enc = RGBEncoder(pretrained=False).to(device)
    x_rgb = torch.randn(2, 3, 512, 512).to(device)
    rgb_feats = rgb_enc(x_rgb)
    print(f"  Input: {x_rgb.shape}")
    for i, feat in enumerate(rgb_feats):
        print(f"  Stage {i}: {feat.shape}")
