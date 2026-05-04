"""
MambaVision RGB Encoder
=======================
RGBEncoder: MambaVision backbone wrapper with pretrained weight loading.
Uses the original MambaVision model without bypass or fallback paths.
"""

from __future__ import annotations

import os
import sys
from typing import List, Tuple, Optional

import torch
import torch.nn as nn


MAMBAVISION_PATH = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "MambaVision", "MambaVision")
)
if MAMBAVISION_PATH not in sys.path:
    sys.path.insert(0, MAMBAVISION_PATH)


class RGBEncoder(nn.Module):
    """
    RGB Encoder using the original MambaVision backbone.

    Variant selection is controlled by model_variant.
    Returns multi-scale features from 4 stages.
    """

    def __init__(
        self,
        depths: Tuple[int, ...] = (3, 3, 10, 5),
        dims: Tuple[int, ...] = (128, 256, 512, 1024),
        drop_path_rate: float = 0.3,
        pretrained: bool = True,
        weights_path: Optional[str] = None,
        model_variant: str = "base",
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
            from mambavision.models.mamba_vision import (
                mamba_vision_T,
                mamba_vision_T2,
                mamba_vision_S,
                mamba_vision_B,
                mamba_vision_L,
                mamba_vision_L2,
                window_partition,
                window_reverse,
            )
        except ImportError as exc:
            raise ImportError(
                "MambaVision is required for the RGB encoder. "
                "Check that Mamba-Segmentation/MambaVision/MambaVision is available."
            ) from exc

        factory = {
            "tiny": mamba_vision_T,
            "tiny2": mamba_vision_T2,
            "small": mamba_vision_S,
            "base": mamba_vision_B,
            "large": mamba_vision_L,
            "large2": mamba_vision_L2,
        }.get(self.model_variant, mamba_vision_B)

        self.backbone = factory(
            pretrained=False,
            depths=list(depths),
            dim=int(dims[0]),
            drop_path_rate=drop_path_rate,
        )
        self._window_partition = window_partition
        self._window_reverse = window_reverse

        if pretrained and resolved_weights:
            self.load_pretrained_weights(resolved_weights)

    def load_pretrained_weights(self, path: str) -> None:
        """Load pretrained weights into MambaVision backbone."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Pretrained weights not found at {path}")

        print(f"Loading pretrained weights from {path}...")
        try:
            import argparse

            torch.serialization.add_safe_globals([argparse.Namespace])
        except Exception:
            pass
        try:
            self.backbone._load_state_dict(path, strict=False)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load pretrained weights at {path}: {exc}"
            ) from exc

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Args:
            x: (N, 3, H, W) RGB input
        Returns:
            List of 4 feature maps at different scales.
        """
        x = self.backbone.patch_embed(x)
        features: List[torch.Tensor] = []
        for level in self.backbone.levels:
            _, _, H, W = x.shape
            if level.transformer_block:
                pad_r = (level.window_size - W % level.window_size) % level.window_size
                pad_b = (level.window_size - H % level.window_size) % level.window_size
                if pad_r > 0 or pad_b > 0:
                    x = torch.nn.functional.pad(x, (0, pad_r, 0, pad_b))
                    _, _, Hp, Wp = x.shape
                else:
                    Hp, Wp = H, W
                x = self._window_partition(x, level.window_size)

            for blk in level.blocks:
                x = blk(x)

            if level.transformer_block:
                x = self._window_reverse(x, level.window_size, Hp, Wp)
                if pad_r > 0 or pad_b > 0:
                    x = x[:, :, :H, :W].contiguous()

            features.append(x)

            if level.downsample is not None:
                x = level.downsample(x)

        return features


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\nTesting RGBEncoder (MambaVision)...")
    rgb_enc = RGBEncoder(pretrained=False).to(device)
    x_rgb = torch.randn(2, 3, 512, 512).to(device)
    rgb_feats = rgb_enc(x_rgb)
    print(f"  Input: {x_rgb.shape}")
    for i, feat in enumerate(rgb_feats):
        print(f"  Stage {i}: {feat.shape}")
