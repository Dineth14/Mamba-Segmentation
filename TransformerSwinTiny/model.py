"""
Swin Tiny Segmentation Model
============================
RGB-only segmentation model combining Swin Tiny encoder + LightUNetDecoder.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from config import Config
from encoders import RGBEncoder
from light_decoder import LightUNetDecoder


class SwinTinySegmentation(nn.Module):
    def __init__(
        self,
        num_classes: int = 7,
        encoder_embed_dim: int = 96,
        encoder_depths: Tuple[int, ...] = (2, 2, 6, 2),
        encoder_heads: Tuple[int, ...] = (3, 6, 12, 24),
        window_size: int = 7,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop_path_rate: float = 0.2,
        ape: bool = False,
        patch_norm: bool = True,
        pretrained_rgb: bool = True,
        weights_path: Optional[str] = None,
        img_size: int = 512,
        decoder_channels: int = 256,
    ) -> None:
        super().__init__()

        self.rgb_encoder = RGBEncoder(
            img_size=img_size,
            patch_size=4,
            embed_dim=encoder_embed_dim,
            depths=encoder_depths,
            num_heads=encoder_heads,
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            drop_path_rate=drop_path_rate,
            ape=ape,
            patch_norm=patch_norm,
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
        features = self.rgb_encoder(rgb)
        main_out, aux_out = self.decoder(features, target_size=target_size)
        if return_features:
            return main_out, aux_out, {"rgb_features": features}
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


def build_model(cfg: Config) -> SwinTinySegmentation:
    return SwinTinySegmentation(
        num_classes=cfg.NUM_CLASSES,
        encoder_embed_dim=cfg.SWIN_EMBED_DIM,
        encoder_depths=cfg.SWIN_DEPTHS,
        encoder_heads=cfg.SWIN_NUM_HEADS,
        window_size=cfg.SWIN_WINDOW_SIZE,
        mlp_ratio=cfg.SWIN_MLP_RATIO,
        qkv_bias=cfg.SWIN_QKV_BIAS,
        drop_path_rate=cfg.SWIN_DROP_PATH,
        ape=cfg.SWIN_APE,
        patch_norm=cfg.SWIN_PATCH_NORM,
        pretrained_rgb=cfg.USE_PRETRAINED,
        weights_path=cfg.WEIGHTS_PATH,
        img_size=cfg.CROP_SIZE,
        decoder_channels=cfg.DECODER_CHANNELS,
    )
