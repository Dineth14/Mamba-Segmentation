"""
UrbanMamba Custom Building Blocks
=================================
Shared layers used by the RGB-only UrbanMamba model.
"""

import torch
import torch.nn as nn
from typing import Tuple


class LayerNorm2d(nn.Module):
    """
    LayerNorm for 2D feature maps (N, C, H, W).
    Automatically permutes to (N, H, W, C) for normalization, then back.
    
    This is crucial for bridging CNN-style features with Transformer/Mamba style normalization.
    """
    
    def __init__(self, num_channels: int, eps: float = 1e-6):
        super().__init__()
        self.norm = nn.LayerNorm(num_channels, eps=eps)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (N, C, H, W) tensor
        Returns:
            (N, C, H, W) normalized tensor
        """
        # (N, C, H, W) -> (N, H, W, C)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        # (N, H, W, C) -> (N, C, H, W)
        x = x.permute(0, 3, 1, 2)
        return x


class ConvBNReLU(nn.Module):
    """Standard Conv-BN-ReLU block."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = False,
        activation: bool = True
    ):
        super().__init__()
        
        layers = [
            nn.Conv2d(
                in_channels, out_channels, kernel_size,
                stride=stride, padding=padding, dilation=dilation,
                groups=groups, bias=bias
            ),
            nn.BatchNorm2d(out_channels)
        ]
        
        if activation:
            layers.append(nn.ReLU(inplace=True))
        
        self.block = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DepthwiseSeparableConv(nn.Module):
    """Depthwise Separable Convolution for efficiency."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        dilation: int = 1
    ):
        super().__init__()
        
        self.depthwise = nn.Sequential(
            nn.Conv2d(
                in_channels, in_channels, kernel_size,
                stride=stride, padding=padding, dilation=dilation,
                groups=in_channels, bias=False
            ),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        
        self.pointwise = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


if __name__ == "__main__":
    # Test building blocks
    print("Testing LayerNorm2d...")
    ln = LayerNorm2d(64)
    x = torch.randn(2, 64, 32, 32)
    out = ln(x)
    print(f"  Input: {x.shape}, Output: {out.shape}")

    print("\nAll tests passed!")
