"""
Vision Mamba RGB Encoder
========================
RGBEncoder: Vision Mamba (Vim) backbone wrapper for semantic segmentation.
Supports all Vision Mamba variants (tiny, small, base) with proper weight loading.
"""

from __future__ import annotations

import os
import sys
from collections import OrderedDict
from typing import List, Tuple, Optional, Dict

import torch
import torch.nn as nn

# Add Vim path
VIM_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Vim")
if VIM_PATH not in sys.path:
    sys.path.insert(0, VIM_PATH)

from vim.models_mamba import VisionMamba


# Vision Mamba variant configurations
VIM_CONFIGS = {
    "tiny": {
        "embed_dim": 192,
        "depth": 24,
        "default_block_depth": 1,
    },
    "small": {
        "embed_dim": 384,
        "depth": 24,
        "default_block_depth": 1,
    },
    "base": {
        "embed_dim": 768,
        "depth": 24,
        "default_block_depth": 1,
    },
}


class RGBEncoder(nn.Module):
    """
    RGB Encoder using Vision Mamba (Vim) backbone for semantic segmentation.
    
    Features:
    - Supports multiple variants: tiny (192), small (384), base (768)
    - Configurable depth for encoder blocks
    - Proper weight loading from checkpoint files
    - Returns spatial feature maps (B, C, H, W) suitable for segmentation
    """

    def __init__(
        self,
        variant: str = "tiny",
        depths: Tuple[int, ...] = (1,),
        dims: Tuple[int, ...] = (192,),
        drop_path_rate: float = 0.0,
        pretrained: bool = False,
        weights_path: Optional[str] = None,
        patch_size: int = 16,
        use_rms_norm: bool = True,
        fused_add_norm: bool = True,
        residual_in_fp32: bool = True,
        if_rope: bool = False,
        bimamba_type: str = "v2",
        **kwargs
    ) -> None:
        super().__init__()
        
        self.variant = variant.lower()
        self.patch_size = patch_size
        
        # Validate variant
        if self.variant not in VIM_CONFIGS:
            raise ValueError(f"Unknown variant '{self.variant}'. Available: {list(VIM_CONFIGS.keys())}")
        
        # Use provided dims or defaults from variant
        self.embed_dim = dims[0] if dims else VIM_CONFIGS[self.variant]["embed_dim"]
        self.depth = depths[0] if depths else VIM_CONFIGS[self.variant]["depth"]
        
        # Output channels for multi-scale feature extraction
        self.out_channels = [self.embed_dim]
        
        print(f"[RGBEncoder] Initializing Vision Mamba-{self.variant}")
        print(f"  embed_dim: {self.embed_dim}, depth: {self.depth}, patch_size: {patch_size}")

        # Create Vision Mamba backbone
        self.backbone = VisionMamba(
            img_size=224,
            patch_size=patch_size,
            embed_dim=self.embed_dim,
            depth=self.depth,
            rms_norm=use_rms_norm,
            residual_in_fp32=residual_in_fp32,
            fused_add_norm=fused_add_norm,
            final_pool_type='none',  # Keep spatial structure
            if_abs_pos_embed=True,
            if_rope=if_rope,
            if_rope_residual=False,
            bimamba_type=bimamba_type,
            if_cls_token=False,  # No class token for segmentation
            if_divide_out=True,
            use_middle_cls_token=False,
        )
        
        if pretrained and weights_path:
            self.load_pretrained_weights(weights_path)

    def load_pretrained_weights(self, path: str) -> None:
        """
        Load pretrained weights from checkpoint file.
        
        Args:
            path: Path to checkpoint file
        """
        if not os.path.exists(path):
            print(f"[RGBEncoder] Warning: Pretrained weights not found at {path}")
            return

        print(f"[RGBEncoder] Loading pretrained weights from {path}...")
        try:
            checkpoint = torch.load(path, map_location="cpu")
            
            # Extract state dict from checkpoint
            if isinstance(checkpoint, dict):
                if "model" in checkpoint:
                    state_dict = checkpoint["model"]
                elif "state_dict" in checkpoint:
                    state_dict = checkpoint["state_dict"]
                else:
                    state_dict = checkpoint
            else:
                state_dict = checkpoint
            
            # Load state dict
            missing_keys, unexpected_keys = self.backbone.load_state_dict(
                state_dict, strict=False
            )
            
            if missing_keys:
                print(f"[RGBEncoder]   Missing keys: {len(missing_keys)}")
            if unexpected_keys:
                print(f"[RGBEncoder]   Unexpected keys: {len(unexpected_keys)}")
            
            print(f"[RGBEncoder] ✓ Weights loaded successfully")
            
        except Exception as e:
            print(f"[RGBEncoder] Error loading weights: {e}")



    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Forward pass to extract spatial features for segmentation.
        
        Args:
            x: Input tensor (B, 3, H, W) in range [0, 1] or ImageNet normalized
            
        Returns:
            List containing spatial feature map: [(B, C, H', W')] where H' = H/patch_size, W' = W/patch_size
        """
        B, C, H, W = x.shape
        
        # Patch embedding
        x = self.backbone.patch_embed(x)  # (B, L, embed_dim)
        
        # Positional embedding
        if self.backbone.if_abs_pos_embed:
            # Vim pos_embed includes space for CLS token if if_cls_token=True
            # Since we set if_cls_token=False, we use the full pos_embed
            x = x + self.backbone.pos_embed
            x = self.backbone.pos_drop(x)
        
        # Apply Vision Mamba blocks
        residual = None
        for layer in self.backbone.layers:
            x, residual = layer(x, residual)
        
        # Normalize
        if not self.backbone.fused_add_norm:
            if residual is not None:
                residual = residual + self.backbone.drop_path(x)
            x = self.backbone.norm_f(residual.to(dtype=self.backbone.norm_f.weight.dtype))
        else:
            # Use fused norm if available
            fused_add_norm_fn = self.backbone.norm_f
            x = fused_add_norm_fn(x, residual=residual, prenorm=False)
        
        # Reshape to spatial format (B, C, H', W')
        H_patch = H // self.patch_size
        W_patch = W // self.patch_size
        
        # x is (B, H*W/patch_size^2, C)
        feat = x.transpose(1, 2).reshape(B, self.embed_dim, H_patch, W_patch)
        
        return [feat]

    @property
    def num_layers(self) -> int:
        """Return number of Mamba blocks."""
        return len(self.backbone.layers)

    def get_config(self) -> Dict:
        """Return model configuration."""
        return {
            "variant": self.variant,
            "embed_dim": self.embed_dim,
            "depth": self.depth,
            "patch_size": self.patch_size,
            "out_channels": self.out_channels,
        }


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("\n" + "="*70)
    print("Testing Vision Mamba RGBEncoder (all variants)")
    print("="*70)
    
    # Test all variants
    for variant in ["tiny", "small", "base"]:
        try:
            print(f"\n[Test] Variant: {variant}")
            encoder = RGBEncoder(
                variant=variant,
                depths=(1,),
                pretrained=False,  # Set True if weights are available
            ).to(device)
            
            x = torch.randn(2, 3, 224, 224).to(device)
            with torch.no_grad():
                feats = encoder(x)
            
            print(f"  Input shape:  {x.shape}")
            for i, feat in enumerate(feats):
                print(f"  Output {i} shape: {feat.shape}")
            print(f"  Config: {encoder.get_config()}")
            print(f"  ✓ {variant} encoder working correctly")
            
        except Exception as e:
            print(f"  ✗ Error: {e}")
    
    print("\n" + "="*70)
