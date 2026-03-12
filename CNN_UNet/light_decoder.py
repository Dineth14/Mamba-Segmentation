"""
Lightweight U-Net style decoder for multi-scale features.

Uses Conv-BN-ReLU blocks and a single main head.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple


def _init_weights(module: nn.Module) -> None:
    """Kaiming init for convs and sensible defaults for BatchNorm2d."""
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)


class ConvBlock(nn.Module):
    """Two stacked 3x3 Conv-BN-ReLU blocks with optional dropout."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout: float = 0.0,
        norm_layer: nn.Module = nn.BatchNorm2d
    ):
        super().__init__()

        layers = [
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            norm_layer(out_channels),
            nn.ReLU(inplace=True),
        ]

        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))

        layers.extend([
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            norm_layer(out_channels),
            nn.ReLU(inplace=True),
        ])

        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UpBlock(nn.Module):
    """Upsample 2x, concatenate skip, then refine with ConvBlock."""

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        dropout: float = 0.0,
        norm_layer: nn.Module = nn.BatchNorm2d
    ):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv = ConvBlock(
            in_channels + skip_channels,
            out_channels,
            dropout=dropout,
            norm_layer=norm_layer
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)

        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class LightUNetDecoder(nn.Module):
    """
    Lightweight convolutional decoder operating on fused feature pyramid.

    Expected feature ordering:
        f0: stride 4  (N, C0, H/4,  W/4)
        f1: stride 8  (N, C1, H/8,  W/8)
        f2: stride 16 (N, C2, H/16, W/16)
        f3: stride 32 (N, C3, H/32, W/32)

    Returns:
        main logits at full resolution
        no auxiliary logits
    """

    def __init__(
        self,
        encoder_channels: Tuple[int, ...] = (96, 192, 384, 768),
        decoder_channels: int = 256,
        num_classes: int = 7,
        use_aux: bool = False,
        dropout: float = 0.1,
        norm_layer: nn.Module = nn.BatchNorm2d
    ):
        super().__init__()
        assert len(encoder_channels) == 4, "LightUNetDecoder expects 4 encoder feature maps."

        c0, c1, c2, c3 = encoder_channels  # strides 4, 8, 16, 32
        base_ch = decoder_channels
        mid_ch = max(decoder_channels // 2, 1)

        # Bottleneck (replace PPM with simple ConvBlock)
        self.bottleneck = ConvBlock(c3, base_ch, dropout=dropout, norm_layer=norm_layer)

        # Decoder stages
        self.up3 = UpBlock(base_ch, c2, base_ch, dropout=dropout, norm_layer=norm_layer)      # 1/32 -> 1/16
        self.up2 = UpBlock(base_ch, c1, mid_ch, dropout=dropout, norm_layer=norm_layer)       # 1/16 -> 1/8
        self.up1 = UpBlock(mid_ch, c0, mid_ch, dropout=dropout, norm_layer=norm_layer)        # 1/8  -> 1/4

        # Heads
        self.main_head = nn.Conv2d(mid_ch, num_classes, 1)
        self.use_aux = use_aux
        if use_aux:
            self.aux_head = nn.Conv2d(mid_ch, num_classes, 1)
        else:
            self.aux_head = None

        _init_weights(self)

    def forward(
        self,
        features: List[torch.Tensor],
        target_size: Optional[Tuple[int, int]] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if len(features) != 4:
            raise ValueError(f"Expected 4 feature maps, got {len(features)}.")

        f0, f1, f2, f3 = features

        x = self.bottleneck(f3)
        x2 = self.up3(x, f2)
        x1 = self.up2(x2, f1)
        x0 = self.up1(x1, f0)

        # Heads
        main_out = self.main_head(x0)
        aux_out = self.aux_head(x1) if self.use_aux else None

        # Upsample logits to input resolution
        if target_size is None:
            target_size = (f0.shape[2] * 4, f0.shape[3] * 4)

        main_out = F.interpolate(main_out, size=target_size, mode='bilinear', align_corners=False)
        if aux_out is not None:
            aux_out = F.interpolate(aux_out, size=target_size, mode='bilinear', align_corners=False)

        return main_out, aux_out


if __name__ == "__main__":
    # Shape sanity test with dummy tensors
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    decoder = LightUNetDecoder(
        encoder_channels=(96, 192, 384, 768),
        decoder_channels=256,
        num_classes=7,
        use_aux=False,
        dropout=0.1
    ).to(device)

    B, H, W = 2, 640, 640
    fused_feats = [
        torch.randn(B, 96, H // 4, W // 4, device=device),
        torch.randn(B, 192, H // 8, W // 8, device=device),
        torch.randn(B, 384, H // 16, W // 16, device=device),
        torch.randn(B, 768, H // 32, W // 32, device=device),
    ]

    main, aux = decoder(fused_feats, target_size=(H, W))

    assert main.shape == (B, 7, H, W), f"Main output shape mismatch: {main.shape}"
    if aux is not None:
        assert aux.shape == (B, 7, H, W), f"Aux output shape mismatch: {aux.shape}"

    print("LightUNetDecoder sanity check passed.")
