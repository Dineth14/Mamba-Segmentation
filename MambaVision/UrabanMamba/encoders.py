"""
UrbanMamba RGB Encoder (MambaVision)
===================================
RGBEncoder: MambaVision backbone wrapper with pretrained weight loading.
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Dict, Any

from parts import ConvBNReLU

try:
    from mambavision.models.mamba_vision import window_partition, window_reverse
except Exception:
    window_partition = None
    window_reverse = None


def _window_partition_fallback(x: torch.Tensor, window_size: int) -> torch.Tensor:
    """Local fallback for window partition if MambaVision import fails."""
    B, C, H, W = x.shape
    x = x.view(B, C, H // window_size, window_size, W // window_size, window_size)
    x = x.permute(0, 2, 4, 3, 5, 1).contiguous()
    return x.view(-1, window_size * window_size, C)


def _window_reverse_fallback(windows: torch.Tensor, window_size: int, H: int, W: int) -> torch.Tensor:
    """Local fallback for window reverse if MambaVision import fails."""
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 5, 1, 3, 2, 4).contiguous()
    return x.view(B, windows.shape[2], H, W)


MAMBA_VISION_PATH = "/storage2/ChangeDetection/NSST-mamba/mamba_vision/MambaVision"
if MAMBA_VISION_PATH not in sys.path:
    sys.path.insert(0, MAMBA_VISION_PATH)


class RGBEncoder(nn.Module):
    """
    RGB Encoder using MambaVision backbone.
    
    Variant configuration is controlled by model_variant.
    Returns multi-scale features from 4 stages.
    """
    
    def __init__(
        self,
        depths: Tuple[int, ...] = (2, 2, 27, 2),
        dims: Tuple[int, ...] = (96, 192, 384, 768),
        drop_path_rate: float = 0.3,
        pretrained: bool = True,
        weights_path: Optional[str] = None,
        model_variant: str = "small"
    ):
        super().__init__()
        
        self.depths = depths
        self.dims = dims
        self.out_channels = list(dims)
        self.model_variant = model_variant.lower()
        resolved_weights = None
        self._weights_load_failed = False
        if pretrained and weights_path:
            if os.path.isfile(weights_path):
                resolved_weights = weights_path
            else:
                print(f"Warning: Pretrained weights path is not a file: {weights_path}")
        
        # Import and create MambaVision backbone (for feature extraction)
        try:
            from mambavision.models.mamba_vision import (
                mamba_vision_T,
                mamba_vision_T2,
                mamba_vision_S,
                mamba_vision_B,
                mamba_vision_B_21k,
                mamba_vision_L,
                mamba_vision_L_21k,
                mamba_vision_L2,
                mamba_vision_L2_512_21k,
                mamba_vision_L3_256_21k,
                mamba_vision_L3_512_21k,
            )
            
            factory = {
                "tiny": mamba_vision_T,
                "tiny2": mamba_vision_T2,
                "small": mamba_vision_S,
                "base": mamba_vision_B,
                "base_21k": mamba_vision_B_21k,
                "large": mamba_vision_L,
                "large_21k": mamba_vision_L_21k,
                "large2": mamba_vision_L2,
                "large2_512_21k": mamba_vision_L2_512_21k,
                "large3_256_21k": mamba_vision_L3_256_21k,
                "large3_512_21k": mamba_vision_L3_512_21k,
            }.get(self.model_variant, mamba_vision_S)
            
            self.backbone = factory(
                pretrained=False,
                depths=list(depths),
                dim=int(dims[0]),
                drop_path_rate=drop_path_rate,
            )
            if resolved_weights is not None and hasattr(self.backbone, "_load_state_dict"):
                try:
                    if os.path.getsize(resolved_weights) < 1024:
                        raise RuntimeError(f"Weight file is too small: {resolved_weights}")
                    self.backbone._load_state_dict(resolved_weights, strict=False)
                    self._weights_loaded = True
                except Exception as exc:
                    print(f"Warning: Failed to load MambaVision weights ({resolved_weights}): {exc}")
                    self._weights_load_failed = True
                    self._weights_loaded = False
            else:
                self._weights_loaded = False
            
        except ImportError as e:
            print(f"Warning: Could not import MambaVision: {e}")
            print("Falling back to ResNet-style encoder...")
            self.backbone = self._create_fallback_encoder(dims)
            self._weights_loaded = False
        
        # Load weights if not already loaded via Backbone_VSSM
        if pretrained and resolved_weights and not self._weights_loaded and not self._weights_load_failed:
            self.load_pretrained_weights(resolved_weights)
    
    def _create_fallback_encoder(self, dims: Tuple[int, ...]) -> nn.Module:
        """Create a fallback CNN encoder if MambaVision is not available."""
        return FallbackRGBEncoder(dims)
    
    def load_pretrained_weights(self, path: str) -> None:
        """Load pretrained weights into MambaVision backbone."""
        if not os.path.exists(path):
            print(f"Warning: Pretrained weights not found at {path}")
            return

        print(f"Loading pretrained weights from {path}...")
        if hasattr(self.backbone, "_load_state_dict"):
            self.backbone._load_state_dict(path, strict=False)
            return

        checkpoint = torch.load(path, map_location="cpu")
        if "model" in checkpoint:
            state_dict = checkpoint["model"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
        missing, unexpected = self.backbone.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"  Missing keys: {len(missing)}")
        if unexpected:
            print(f"  Unexpected keys: {len(unexpected)}")
    
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Args:
            x: (N, 3, H, W) RGB input
        Returns:
            List of 4 feature maps at different scales:
            - Stage 0: (N, 96, H/4, W/4)
            - Stage 1: (N, 192, H/8, W/8)
            - Stage 2: (N, 384, H/16, W/16)
            - Stage 3: (N, 768, H/32, W/32)
        """
        if hasattr(self.backbone, "patch_embed") and hasattr(self.backbone, "levels"):
            x = self.backbone.patch_embed(x)
            features: List[torch.Tensor] = []
            for level in self.backbone.levels:
                if getattr(level, "transformer_block", False):
                    if window_partition is None or window_reverse is None:
                        window_partition_fn = _window_partition_fallback
                        window_reverse_fn = _window_reverse_fallback
                    else:
                        window_partition_fn = window_partition
                        window_reverse_fn = window_reverse
                    H, W = x.shape[2:]
                    window_size = level.window_size
                    pad_r = (window_size - W % window_size) % window_size
                    pad_b = (window_size - H % window_size) % window_size
                    if pad_r > 0 or pad_b > 0:
                        x = F.pad(x, (0, pad_r, 0, pad_b))
                    Hp, Wp = x.shape[2], x.shape[3]
                    x = window_partition_fn(x, window_size)

                for blk in level.blocks:
                    x = blk(x)

                if getattr(level, "transformer_block", False):
                    x = window_reverse_fn(x, window_size, Hp, Wp)
                    if pad_r > 0 or pad_b > 0:
                        x = x[:, :, :H, :W].contiguous()

                features.append(x)

                if level.downsample is not None:
                    x = level.downsample(x)
            return features
        features = self.backbone(x)
        if isinstance(features, (list, tuple)):
            return list(features)
        return [features]


class FallbackRGBEncoder(nn.Module):
    """Fallback CNN encoder when MambaVision is not available."""
    
    def __init__(self, dims: Tuple[int, ...] = (96, 192, 384, 768)):
        super().__init__()
        
        self.dims = dims
        
        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(3, dims[0], 7, stride=4, padding=3, bias=False),
            nn.BatchNorm2d(dims[0]),
            nn.ReLU(inplace=True)
        )
        
        # Stages
        self.stage1 = self._make_stage(dims[0], dims[0], 2, stride=1)
        self.down1 = nn.Conv2d(dims[0], dims[1], 3, stride=2, padding=1, bias=False)
        
        self.stage2 = self._make_stage(dims[1], dims[1], 2, stride=1)
        self.down2 = nn.Conv2d(dims[1], dims[2], 3, stride=2, padding=1, bias=False)
        
        self.stage3 = self._make_stage(dims[2], dims[2], 6, stride=1)
        self.down3 = nn.Conv2d(dims[2], dims[3], 3, stride=2, padding=1, bias=False)
        
        self.stage4 = self._make_stage(dims[3], dims[3], 2, stride=1)
    
    def _make_stage(self, in_ch: int, out_ch: int, num_blocks: int, stride: int = 1) -> nn.Sequential:
        layers = []
        for i in range(num_blocks):
            layers.append(ConvBNReLU(in_ch if i == 0 else out_ch, out_ch, 3, 1, 1))
        return nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        features = []
        
        x = self.stem(x)
        x = self.stage1(x)
        features.append(x)
        
        x = self.down1(x)
        x = self.stage2(x)
        features.append(x)
        
        x = self.down2(x)
        x = self.stage3(x)
        features.append(x)
        
        x = self.down3(x)
        x = self.stage4(x)
        features.append(x)
        
        return features


if __name__ == "__main__":
    # Test encoders
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("\nTesting RGBEncoder (fallback mode)...")
    rgb_enc = RGBEncoder(pretrained=False).to(device)
    x_rgb = torch.randn(2, 3, 512, 512).to(device)
    rgb_feats = rgb_enc(x_rgb)
    print(f"  Input: {x_rgb.shape}")
    for i, f in enumerate(rgb_feats):
        print(f"  Stage {i}: {f.shape}")
    
    print("\nAll tests passed!")
