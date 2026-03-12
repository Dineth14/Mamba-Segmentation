"""
ResNet Segmentation Model
=========================
RGB-only segmentation model combining ResNet encoder + LightUNetDecoder.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from config import Config
from encoders import RGBEncoder
from light_decoder import LightUNetDecoder


class ResNetSegmentation(nn.Module):
    """ResNet encoder + LightUNetDecoder segmentation model."""

    def __init__(
        self,
        num_classes: int = 7,
        encoder_depth: int = 18,
        decoder_channels: int = 256,
        pretrained_rgb: bool = True,
        weights_path: Optional[str] = None,
    ) -> None:
        super().__init__()

        self.num_classes = num_classes
        self.rgb_encoder = RGBEncoder(
            depth=encoder_depth,
            pretrained=pretrained_rgb,
            weights_path=weights_path,
        )

        self.decoder = LightUNetDecoder(
            encoder_channels=self.rgb_encoder.out_channels,
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
            return main_out, aux_out, {"rgb_features": rgb_features}
        return main_out, aux_out

    def get_param_groups(
        self,
        lr_backbone: float = 6e-5,
        lr_head: float = 3e-4,
        weight_decay: float = 0.05,
    ) -> List[Dict[str, Any]]:
        backbone_params = []
        head_params = []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if "rgb_encoder" in name:
                backbone_params.append(param)
            else:
                head_params.append(param)
        return [
            {"params": backbone_params, "lr": lr_backbone, "weight_decay": weight_decay, "name": "backbone"},
            {"params": head_params, "lr": lr_head, "weight_decay": weight_decay, "name": "head"},
        ]


def build_model(cfg: Config) -> ResNetSegmentation:
    return ResNetSegmentation(
        num_classes=cfg.NUM_CLASSES,
        encoder_depth=cfg.RESNET_DEPTH,
        decoder_channels=cfg.DECODER_CHANNELS,
        pretrained_rgb=cfg.USE_PRETRAINED,
        weights_path=cfg.WEIGHTS_PATH,
    )
