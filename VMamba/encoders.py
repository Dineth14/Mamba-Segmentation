"""
UrbanMamba RGB Encoder
======================
RGBEncoder: VMamba backbone wrapper with pretrained weight loading.
"""

import os
import sys
import torch
import torch.nn as nn
from typing import List, Tuple, Optional, Dict, Any
from collections import OrderedDict

# Add VMamba to path
VMAMBA_PATH = "/storage2/ChangeDetection/NSST-mamba/NSST_Mamba_v2/VMamba"
if VMAMBA_PATH not in sys.path:
    sys.path.insert(0, VMAMBA_PATH)

from parts import ConvBNReLU


class RGBEncoder(nn.Module):
    """
    RGB Encoder using VMamba backbone.
    
    Variant configuration is controlled by depths/dims/drop_path_rate.
    Returns multi-scale features from 4 stages.
    """
    
    def __init__(
        self,
        depths: Tuple[int, ...] = (2, 2, 27, 2),
        dims: Tuple[int, ...] = (96, 192, 384, 768),
        drop_path_rate: float = 0.3,
        pretrained: bool = True,
        weights_path: Optional[str] = None
    ):
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
        
        # Import and create VMamba Backbone (for feature extraction)
        try:
            from vmamba import Backbone_VSSM
            
            # Use Backbone_VSSM which returns multi-scale features
            # and doesn't have a classifier
            self.backbone = Backbone_VSSM(
                depths=list(depths),
                dims=list(dims),
                drop_path_rate=drop_path_rate,
                forward_type="v05",  # No custom CUDA kernels required
                downsample_version="v1",  # Match pretrained weights structure
                out_indices=(0, 1, 2, 3),  # Extract features from all 4 stages
                pretrained=resolved_weights,
            )
            self._weights_loaded = pretrained and resolved_weights is not None
            
        except ImportError as e:
            print(f"Warning: Could not import VMamba Backbone_VSSM: {e}")
            print("Falling back to ResNet-style encoder...")
            self.backbone = self._create_fallback_encoder(dims)
            self._weights_loaded = False
        
        # Load weights if not already loaded via Backbone_VSSM
        if pretrained and resolved_weights and not self._weights_loaded:
            self.load_pretrained_weights(weights_path)
    
    def _create_fallback_encoder(self, dims: Tuple[int, ...]) -> nn.Module:
        """Create a fallback CNN encoder if VMamba is not available."""
        return FallbackRGBEncoder(dims)
    
    def load_pretrained_weights(self, path: str) -> None:
        """
        Load pretrained ImageNet-1K weights.
        
        Handles key cleaning:
        - Removes 'backbone.' prefix
        - Removes 'head.' prefix
        - Removes 'model.' prefix
        """
        if not os.path.exists(path):
            print(f"Warning: Pretrained weights not found at {path}")
            return
        
        print(f"Loading pretrained weights from {path}...")
        
        checkpoint = torch.load(path, map_location='cpu')
        
        # Handle different checkpoint formats
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        
        # Clean keys
        cleaned_state_dict = OrderedDict()
        for key, value in state_dict.items():
            new_key = key
            
            # Remove common prefixes
            for prefix in ['backbone.', 'model.', 'module.']:
                if new_key.startswith(prefix):
                    new_key = new_key[len(prefix):]
            
            # Skip classifier/head weights
            if any(skip in new_key for skip in ['head.', 'classifier.', 'fc.']):
                continue
            
            cleaned_state_dict[new_key] = value
        
        # Load with strict=False to handle missing/extra keys
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
            List of 4 feature maps at different scales:
            - Stage 0: (N, 96, H/4, W/4)
            - Stage 1: (N, 192, H/8, W/8)
            - Stage 2: (N, 384, H/16, W/16)
            - Stage 3: (N, 768, H/32, W/32)
        """
        # Backbone_VSSM directly returns list of features from each stage
        features = self.backbone(x)
        if isinstance(features, (list, tuple)):
            return list(features)
        else:
            # Fallback: wrap single tensor in list
            return [features]


class FallbackRGBEncoder(nn.Module):
    """Fallback CNN encoder when VMamba is not available."""
    
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
