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
    Simplified Decoder for Vision Mamba with 1 block.
    
    Accepts a single feature map.
    """

    def __init__(
        self,
        encoder_channels: Tuple[int, ...] = (192,),
        decoder_channels: int = 256,
        num_classes: int = 7,
        use_aux: bool = False,
        dropout: float = 0.1,
        norm_layer: nn.Module = nn.BatchNorm2d
    ):
        super().__init__()
        
        # We expect encoder_channels to be a tuple, but we only use the last/only one.
        in_ch = encoder_channels[-1] if encoder_channels else 192
        
        self.block = ConvBlock(in_ch, decoder_channels, dropout=dropout, norm_layer=norm_layer)
        self.head = nn.Conv2d(decoder_channels, num_classes, 1)
        
        _init_weights(self)

    def forward(
        self,
        features: List[torch.Tensor],
        target_size: Optional[Tuple[int, int]] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        
        # We expect a list with at least one feature map
        if not features:
            raise ValueError("No features provided to decoder")
            
        x = features[-1] # Take the last feature map (highest semantic level / only one)
        
        x = self.block(x)
        out = self.head(x)
        
        if target_size is not None:
            out = F.interpolate(out, size=target_size, mode='bilinear', align_corners=False)
            
        return out, None


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
