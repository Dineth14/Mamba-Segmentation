"""
VMamba RGB Encoder
==================
RGBEncoder: VMamba backbone wrapper with pretrained weight loading.
Uses the original VMamba configuration (no bypass forward types).
"""

from __future__ import annotations

import os
import sys
from collections import OrderedDict
from typing import List, Tuple, Optional

import torch
import torch.nn as nn


VMAMBA_PATH = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "VMamba", "VMamba")
)
if VMAMBA_PATH not in sys.path:
    sys.path.insert(0, VMAMBA_PATH)


class RGBEncoder(nn.Module):
    """
    RGB Encoder using the original VMamba backbone.

    Variant configuration is controlled by depths/dims/drop_path_rate.
    Returns multi-scale features from 4 stages.
    """

    def __init__(
        self,
        depths: Tuple[int, ...] = (2, 2, 27, 2),
        dims: Tuple[int, ...] = (96, 192, 384, 768),
        drop_path_rate: float = 0.3,
        pretrained: bool = True,
        weights_path: Optional[str] = None,
        ssm_d_state: int = 16,
        ssm_ratio: float = 2.0,
        ssm_dt_rank: str = "auto",
        ssm_act_layer: str = "silu",
        ssm_conv: int = 3,
        ssm_conv_bias: bool = True,
        ssm_drop_rate: float = 0.0,
        ssm_init: str = "v0",
        forward_type: str = "v0",
        mlp_ratio: float = 0.0,
        gmlp: bool = False,
        norm_layer: str = "ln",
        downsample_version: str = "v1",
        patchembed_version: str = "v1",
    ) -> None:
        super().__init__()

        self.depths = depths
        self.dims = dims
        self.out_channels = list(dims)

        resolved_weights = None
        if pretrained and weights_path:
            if os.path.isfile(weights_path):
                resolved_weights = weights_path
            else:
                print(f"Warning: Pretrained weights path is not a file: {weights_path}")

        try:
            from vmamba import Backbone_VSSM

            self.backbone = Backbone_VSSM(
                depths=list(depths),
                dims=list(dims),
                drop_path_rate=drop_path_rate,
                out_indices=(0, 1, 2, 3),
                ssm_d_state=ssm_d_state,
                ssm_ratio=ssm_ratio,
                ssm_dt_rank=ssm_dt_rank,
                ssm_act_layer=ssm_act_layer,
                ssm_conv=ssm_conv,
                ssm_conv_bias=ssm_conv_bias,
                ssm_drop_rate=ssm_drop_rate,
                ssm_init=ssm_init,
                forward_type=forward_type,
                mlp_ratio=mlp_ratio,
                gmlp=gmlp,
                patch_norm=True,
                norm_layer=norm_layer,
                downsample_version=downsample_version,
                patchembed_version=patchembed_version,
                pretrained=None,
            )
            self._weights_loaded = False
        except ImportError as exc:
            raise ImportError(
                "VMamba Backbone_VSSM is required for the RGB encoder. "
                "Check that Mamba-Segmentation/VMamba/VMamba is available."
            ) from exc

        if pretrained and resolved_weights:
            self.load_pretrained_weights(resolved_weights)

    def load_pretrained_weights(self, path: str) -> None:
        """
        Load pretrained ImageNet-1K weights.

        Handles key cleaning:
        - Removes common prefixes (backbone., model., module.)
        - Skips classifier/head weights
        """
        if not os.path.exists(path):
            print(f"Warning: Pretrained weights not found at {path}")
            return

        print(f"Loading pretrained weights from {path}...")
        checkpoint = torch.load(path, map_location="cpu")

        if isinstance(checkpoint, dict) and "model" in checkpoint:
            state_dict = checkpoint["model"]
        elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint

        cleaned_state_dict = OrderedDict()
        for key, value in state_dict.items():
            new_key = key
            for prefix in ("backbone.", "model.", "module."):
                if new_key.startswith(prefix):
                    new_key = new_key[len(prefix):]

            if any(skip in new_key for skip in ("head.", "classifier.", "fc.")):
                continue

            cleaned_state_dict[new_key] = value

        missing, unexpected = self.backbone.load_state_dict(cleaned_state_dict, strict=False)
        if missing:
            print(f"  Missing keys: {len(missing)}")
        if unexpected:
            print(f"  Unexpected keys: {len(unexpected)}")

        print(f"  Successfully loaded {len(cleaned_state_dict)} weight tensors")

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
    print("\nTesting RGBEncoder (VMamba)...")
    rgb_enc = RGBEncoder(pretrained=False).to(device)
    x_rgb = torch.randn(2, 3, 512, 512).to(device)
    rgb_feats = rgb_enc(x_rgb)
    print(f"  Input: {x_rgb.shape}")
    for i, feat in enumerate(rgb_feats):
        print(f"  Stage {i}: {feat.shape}")
