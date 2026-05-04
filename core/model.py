"""
core/model.py
=============
Unified SegmentationModel that supports all backbone families via dynamic import.

Usage:
    from core.model import SegmentationModel
    model = SegmentationModel(cfg)   # cfg is a dict from core/config_loader.py
"""

from __future__ import annotations

import importlib
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from core.light_decoder import LightUNetDecoder


class SegmentationModel(nn.Module):
    """
    RGB-only semantic segmentation model.

    Combines any registered encoder backbone with the lightweight U-Net decoder.
    The backbone is selected at construction time by ``cfg["backbone"]``:

      backbone       module
      ----------     ---------------------------------
      mambavision    backbones.mambavision.encoders
      vmamba         backbones.vmamba.encoders
      visionmamba    backbones.visionmamba.encoders
      spatialmamba   backbones.spatialmamba.encoders
      cnn            backbones.cnn.encoders
      swintransformer backbones.swintransformer.encoders
    """

    def __init__(self, cfg: Dict[str, Any]) -> None:
        super().__init__()

        backbone = cfg["backbone"].lower()
        enc_mod = importlib.import_module(f"backbones.{backbone}.encoders")

        encoder_kwargs = cfg.get("encoder_kwargs", {})
        self.rgb_encoder = enc_mod.RGBEncoder(**encoder_kwargs)

        self.decoder = LightUNetDecoder(
            encoder_channels=self.rgb_encoder.out_channels,
            decoder_channels=cfg.get("decoder_channels", 256),
            num_classes=cfg["num_classes"],
            use_aux=False,
        )

        self.num_classes = cfg["num_classes"]

    # ------------------------------------------------------------------
    def forward(
        self,
        rgb: torch.Tensor,
        return_features: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            rgb: (N, 3, H, W)
        Returns:
            main_out: (N, num_classes, H, W)
            aux_out:  (N, num_classes, H, W) or None
        """
        target_size = rgb.shape[2:]
        features = self.rgb_encoder(rgb)
        main_out, aux_out = self.decoder(features, target_size=target_size)

        if return_features:
            return main_out, aux_out, {"features": features}
        return main_out, aux_out

    # ------------------------------------------------------------------
    def get_param_groups(
        self,
        lr_backbone: float = 6e-5,
        lr_head: float = 3e-4,
        weight_decay: float = 0.05,
    ) -> List[Dict[str, Any]]:
        """Return parameter groups with differential learning rates."""
        no_decay_keywords = ["bias", "bn", "norm", "ln"]
        backbone_params: List[nn.Parameter] = []
        backbone_no_decay: List[nn.Parameter] = []
        head_params: List[nn.Parameter] = []
        head_no_decay: List[nn.Parameter] = []

        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            is_backbone = "rgb_encoder.backbone" in name
            has_decay = not any(kw in name.lower() for kw in no_decay_keywords)
            if is_backbone:
                (backbone_params if has_decay else backbone_no_decay).append(param)
            else:
                (head_params if has_decay else head_no_decay).append(param)

        groups = [
            {"params": backbone_params, "lr": lr_backbone, "weight_decay": weight_decay, "name": "backbone"},
            {"params": backbone_no_decay, "lr": lr_backbone, "weight_decay": 0.0, "name": "backbone_no_decay"},
            {"params": head_params, "lr": lr_head, "weight_decay": weight_decay, "name": "head"},
            {"params": head_no_decay, "lr": lr_head, "weight_decay": 0.0, "name": "head_no_decay"},
        ]
        return [g for g in groups if len(g["params"]) > 0]

    # ------------------------------------------------------------------
    @torch.no_grad()
    def inference(self, rgb: torch.Tensor, flip_tta: bool = False) -> torch.Tensor:
        """Inference with optional horizontal flip TTA."""
        self.eval()
        main_out, _ = self.forward(rgb)
        if flip_tta:
            rgb_flip = torch.flip(rgb, dims=[-1])
            main_flip, _ = self.forward(rgb_flip)
            main_flip = torch.flip(main_flip, dims=[-1])
            main_out = (main_out + main_flip) / 2
        return main_out
