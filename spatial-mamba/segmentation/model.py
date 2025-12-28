"""
Spatial-Mamba segmentation model (UrbanMamba-style).
RGB-only encoder + lightweight U-Net decoder.
"""

from __future__ import annotations

import os
import sys
from typing import List, Tuple, Optional, Dict, Any

import torch
import torch.nn as nn


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from encoders import RGBEncoder  # noqa: E402
from light_decoder import LightUNetDecoder  # noqa: E402


class SpatialMambaSegmentation(nn.Module):
    """
    RGB-only Spatial-Mamba segmentation model.
    """

    def __init__(
        self,
        num_classes: int = 7,
        encoder_dims: Tuple[int, ...] = (64, 128, 256, 512),
        decoder_channels: int = 256,
        drop_path_rate: float = 0.3,
        encoder_depths: Tuple[int, ...] = (2, 4, 21, 5),
        pretrained_rgb: bool = True,
        weights_path: Optional[str] = None,
        encoder_variant: str = "small",
        d_state: int = 1,
        dt_init: str = "random",
        mlp_ratio: float = 4.0,
    ) -> None:
        super().__init__()

        self.num_classes = num_classes
        self.encoder_dims = encoder_dims

        self.rgb_encoder = RGBEncoder(
            depths=encoder_depths,
            dims=encoder_dims,
            drop_path_rate=drop_path_rate,
            pretrained=pretrained_rgb,
            weights_path=weights_path,
            model_variant=encoder_variant,
            d_state=d_state,
            dt_init=dt_init,
            mlp_ratio=mlp_ratio,
        )

        self.decoder = LightUNetDecoder(
            encoder_channels=encoder_dims,
            decoder_channels=decoder_channels,
            num_classes=num_classes,
            use_aux=False,
        )

    def forward(
        self,
        rgb: torch.Tensor,
        return_features: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        target_size = rgb.shape[2:]
        rgb_features = self.rgb_encoder(rgb)
        main_out, aux_out = self.decoder(rgb_features, target_size=target_size)

        if return_features:
            return main_out, aux_out, {
                "rgb_features": rgb_features,
            }

        return main_out, aux_out

    def get_param_groups(
        self,
        lr_backbone: float = 6e-5,
        lr_head: float = 3e-4,
        weight_decay: float = 0.05,
    ) -> List[Dict[str, Any]]:
        backbone_params = []
        backbone_params_no_decay = []
        head_params = []
        head_params_no_decay = []

        no_decay_keywords = ["bias", "bn", "norm", "ln"]

        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue

            is_backbone = "rgb_encoder.backbone" in name
            has_decay = not any(kw in name.lower() for kw in no_decay_keywords)

            if is_backbone:
                (backbone_params if has_decay else backbone_params_no_decay).append(param)
            else:
                (head_params if has_decay else head_params_no_decay).append(param)

        param_groups = [
            {"params": backbone_params, "lr": lr_backbone, "weight_decay": weight_decay, "name": "backbone"},
            {"params": backbone_params_no_decay, "lr": lr_backbone, "weight_decay": 0.0, "name": "backbone_no_decay"},
            {"params": head_params, "lr": lr_head, "weight_decay": weight_decay, "name": "head"},
            {"params": head_params_no_decay, "lr": lr_head, "weight_decay": 0.0, "name": "head_no_decay"},
        ]

        return [g for g in param_groups if len(g["params"]) > 0]

    def load_pretrained_rgb(self, weights_path: str) -> None:
        self.rgb_encoder.load_pretrained_weights(weights_path)
