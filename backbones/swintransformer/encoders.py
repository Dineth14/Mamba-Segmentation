"""
Swin Transformer RGB Encoder
============================
Swin Tiny backbone wrapper with pretrained weight loading.
"""

from __future__ import annotations

import math
import os
import sys
from collections import OrderedDict
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import checkpoint


SWIN_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "Swin-Transformer")
)
if SWIN_ROOT not in sys.path:
    sys.path.insert(0, SWIN_ROOT)

from models.swin_transformer import SwinTransformer  # noqa: E402


class RGBEncoder(nn.Module):
    """
    RGB Encoder using Swin Transformer.

    Returns 4 feature maps (1/4, 1/8, 1/16, 1/32).
    """

    def __init__(
        self,
        img_size: int = 512,
        patch_size: int = 4,
        embed_dim: int = 96,
        depths: Tuple[int, ...] = (2, 2, 6, 2),
        num_heads: Tuple[int, ...] = (3, 6, 12, 24),
        window_size: int = 7,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop_path_rate: float = 0.2,
        ape: bool = False,
        patch_norm: bool = True,
        pretrained: bool = True,
        weights_path: Optional[str] = None,
    ) -> None:
        super().__init__()

        self.out_channels = [embed_dim * (2 ** i) for i in range(len(depths))]

        resolved_weights = None
        if pretrained and weights_path:
            if os.path.isfile(weights_path):
                resolved_weights = weights_path
            else:
                print(f"Warning: Pretrained weights path is not a file: {weights_path}")

        self.backbone = SwinTransformer(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=3,
            num_classes=0,
            embed_dim=embed_dim,
            depths=list(depths),
            num_heads=list(num_heads),
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            drop_path_rate=drop_path_rate,
            ape=ape,
            patch_norm=patch_norm,
        )

        if pretrained and resolved_weights:
            self.load_pretrained_weights(resolved_weights)

    def load_pretrained_weights(self, path: str) -> None:
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
            if new_key.startswith("module."):
                new_key = new_key[len("module.") :]
            if new_key.startswith("head."):
                continue
            if "relative_position_index" in new_key or "attn_mask" in new_key:
                continue
            cleaned_state_dict[new_key] = value

        model_state = self.backbone.state_dict()
        for key, value in list(cleaned_state_dict.items()):
            if "relative_position_bias_table" not in key:
                continue
            if key not in model_state:
                continue
            pretrained = value
            current = model_state[key]
            if pretrained.shape == current.shape:
                continue
            n_heads = pretrained.shape[1]
            size_pre = int(math.sqrt(pretrained.shape[0]))
            size_cur = int(math.sqrt(current.shape[0]))
            if size_pre * size_pre != pretrained.shape[0] or size_cur * size_cur != current.shape[0]:
                continue
            resized = pretrained.permute(1, 0).view(1, n_heads, size_pre, size_pre)
            resized = F.interpolate(resized, size=(size_cur, size_cur), mode="bicubic", align_corners=False)
            resized = resized.view(n_heads, size_cur * size_cur).permute(1, 0)
            cleaned_state_dict[key] = resized

        missing, unexpected = self.backbone.load_state_dict(cleaned_state_dict, strict=False)
        if missing:
            print(f"  Missing keys: {len(missing)}")
        if unexpected:
            print(f"  Unexpected keys: {len(unexpected)}")

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        x = self.backbone.patch_embed(x)
        if self.backbone.ape:
            x = x + self.backbone.absolute_pos_embed
        x = self.backbone.pos_drop(x)

        features: List[torch.Tensor] = []
        for layer in self.backbone.layers:
            for blk in layer.blocks:
                if layer.use_checkpoint:
                    x = checkpoint.checkpoint(blk, x)
                else:
                    x = blk(x)
            bsz, _, channels = x.shape
            height, width = layer.input_resolution
            feat = x.view(bsz, height, width, channels).permute(0, 3, 1, 2).contiguous()
            features.append(feat)
            if layer.downsample is not None:
                x = layer.downsample(x)
        return features


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    enc = RGBEncoder(pretrained=False, img_size=512).to(device)
    dummy = torch.randn(2, 3, 512, 512, device=device)
    feats = enc(dummy)
    for i, feat in enumerate(feats):
        print(f"Stage {i}: {feat.shape}")
