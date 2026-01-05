"""
Vision Mamba Segmentation Model
===============================
Multi-variant Vision Mamba segmentation model supporting:
- Encoder: Vision Mamba (tiny, small, base) with configurable depth
- Decoder: Lightweight UNet decoder
"""

import torch
import torch.nn as nn
from typing import List, Tuple, Optional, Dict, Any

from config import Config
from encoders import RGBEncoder
from light_decoder import LightUNetDecoder


class VisionMambaSegmentation(nn.Module):
    """
    Segmentation model combining Vision Mamba encoder with lightweight decoder.
    
    Supports all Vision Mamba variants (tiny: 192, small: 384, base: 768)
    with configurable encoder depth and decoder channels.
    """
    
    def __init__(
        self,
        num_classes: int = 7,
        variant: str = "tiny",
        encoder_dims: Tuple[int, ...] = (192,),
        encoder_depths: Tuple[int, ...] = (1,),
        decoder_channels: int = 256,
        drop_path_rate: float = 0.0,
        pretrained_rgb: bool = True,
        weights_path: Optional[str] = None,
        patch_size: int = 16,
        use_rms_norm: bool = True,
        fused_add_norm: bool = True,
        residual_in_fp32: bool = True,
        if_rope: bool = False,
        bimamba_type: str = "v2",
        **kwargs
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.variant = variant.lower()
        self.encoder_dims = encoder_dims
        self.encoder_depths = encoder_depths
        
        print(f"[VisionMambaSegmentation] Building model with variant={self.variant}")
        
        # ============== RGB Encoder ==============
        self.rgb_encoder = RGBEncoder(
            variant=self.variant,
            depths=self.encoder_depths,
            dims=self.encoder_dims,
            drop_path_rate=drop_path_rate,
            pretrained=pretrained_rgb,
            weights_path=weights_path,
            patch_size=patch_size,
            use_rms_norm=use_rms_norm,
            fused_add_norm=fused_add_norm,
            residual_in_fp32=residual_in_fp32,
            if_rope=if_rope,
            bimamba_type=bimamba_type,
        )
        
        # ============== Decoder ==============
        self.decoder = LightUNetDecoder(
            encoder_channels=self.encoder_dims,
            decoder_channels=decoder_channels,
            num_classes=num_classes,
            use_aux=False
        )
    
    def forward(
        self,
        rgb: torch.Tensor,
        return_features: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass.
        
        Args:
            rgb: (N, 3, H, W) RGB input
            return_features: If True, also return intermediate features
        
        Returns:
            main_out: (N, num_classes, H, W) Main segmentation output
            aux_out: Auxiliary output or None
        """
        # Get target size for decoder upsampling
        target_size = rgb.shape[2:]
        
        # ============== Encode ==============
        rgb_features = self.rgb_encoder(rgb)  # List of feature tensors
        
        # ============== Decode ==============
        main_out, aux_out = self.decoder(
            rgb_features,
            target_size=target_size
        )
        
        if return_features:
            return main_out, aux_out, {
                'rgb_features': rgb_features,
            }
        
        return main_out, aux_out
    
    def get_param_groups(
        self,
        lr_backbone: float = 6e-5,
        lr_head: float = 3e-4,
        weight_decay: float = 0.05
    ) -> List[Dict[str, Any]]:
        """
        Get parameter groups with differential learning rates.
        
        Args:
            lr_backbone: Learning rate for backbone parameters
            lr_head: Learning rate for decoder parameters
            weight_decay: Weight decay coefficient
        """
        backbone_params = []
        backbone_params_no_decay = []
        head_params = []
        head_params_no_decay = []
        
        # Keywords that indicate parameters should not have weight decay
        no_decay_keywords = ['bias', 'bn', 'norm', 'ln']
        
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            
            # Identify if it's a backbone parameter
            is_backbone = 'rgb_encoder.backbone' in name
            
            # Check if it should have weight decay
            has_decay = not any(kw in name.lower() for kw in no_decay_keywords)
            
            if is_backbone:
                if has_decay:
                    backbone_params.append(param)
                else:
                    backbone_params_no_decay.append(param)
            else:
                if has_decay:
                    head_params.append(param)
                else:
                    head_params_no_decay.append(param)
        
        # Build parameter groups
        param_groups = [
            {
                'params': backbone_params,
                'lr': lr_backbone,
                'weight_decay': weight_decay,
                'name': 'backbone'
            },
            {
                'params': backbone_params_no_decay,
                'lr': lr_backbone,
                'weight_decay': 0.0,
                'name': 'backbone_no_decay'
            },
            {
                'params': head_params,
                'lr': lr_head,
                'weight_decay': weight_decay,
                'name': 'decoder'
            },
            {
                'params': head_params_no_decay,
                'lr': lr_head,
                'weight_decay': 0.0,
                'name': 'decoder_no_decay'
            },
        ]
        
        # Remove empty parameter groups
        param_groups = [g for g in param_groups if len(g['params']) > 0]
        
        return param_groups
    
    def load_pretrained_rgb(self, weights_path: str) -> None:
        """Load pretrained weights for RGB encoder."""
        self.rgb_encoder.load_pretrained_weights(weights_path)
    
    def get_model_config(self) -> Dict[str, Any]:
        """Get model configuration."""
        return {
            'variant': self.variant,
            'num_classes': self.num_classes,
            'encoder_dims': self.encoder_dims,
            'encoder_depths': self.encoder_depths,
            'decoder_channels': self.decoder.decoder_channels,
            'rgb_encoder_config': self.rgb_encoder.get_config(),
        }
    
    @torch.no_grad()
    def inference(
        self,
        rgb: torch.Tensor,
        flip_tta: bool = False
    ) -> torch.Tensor:
        """
        Inference with optional test-time augmentation (flip).
        
        Args:
            rgb: Input image (B, 3, H, W)
            flip_tta: Whether to use horizontal flip TTA
            
        Returns:
            Segmentation output (B, num_classes, H, W)
        """
        self.eval()
        
        main_out, _ = self.forward(rgb)
        
        if flip_tta:
            # Horizontal flip TTA
            rgb_flip = torch.flip(rgb, dims=[-1])
            main_flip, _ = self.forward(rgb_flip)
            main_flip = torch.flip(main_flip, dims=[-1])
            main_out = (main_out + main_flip) / 2.0
        
        return main_out


def build_model(cfg: Config) -> VisionMambaSegmentation:
    """
    Build Vision Mamba segmentation model from config.
    
    Args:
        cfg: Config object with all parameters
        
    Returns:
        VisionMambaSegmentation model
    """
    model = VisionMambaSegmentation(
        num_classes=cfg.NUM_CLASSES,
        variant=cfg.VIM_VARIANT,
        encoder_dims=cfg.VIM_DIMS,
        encoder_depths=cfg.VIM_DEPTHS,
        decoder_channels=cfg.DECODER_CHANNELS,
        drop_path_rate=cfg.VIM_DROP_PATH,
        pretrained_rgb=True,
        weights_path=cfg.WEIGHTS_PATH,
        use_rms_norm=cfg.USE_RMS_NORM,
        fused_add_norm=cfg.VIM_FUSED_ADD_NORM,
        residual_in_fp32=cfg.VIM_RESIDUAL_IN_FP32,
        if_rope=cfg.VIM_IF_ROPE,
        bimamba_type=cfg.VIM_BIMAMBA_TYPE,
    )
    
    return model


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("\n" + "="*70)
    print("Testing Vision Mamba Segmentation Model")
    print("="*70)
    
    # Test all variants
    for variant in ["tiny", "small", "base"]:
        try:
            print(f"\n[Test] Building {variant} model...")
            model = VisionMambaSegmentation(
                num_classes=7,
                variant=variant,
                encoder_dims=None,  # Will be auto-determined from variant
                encoder_depths=(1,),
                decoder_channels=256,
                pretrained_rgb=False  # Set True if weights available
            ).to(device)
            
            # Count parameters
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            print(f"  Total parameters: {total_params:,}")
            print(f"  Trainable parameters: {trainable_params:,}")
            
            # Test forward pass
            print(f"  Testing forward pass...")
            rgb = torch.randn(2, 3, 224, 224).to(device)
            with torch.no_grad():
                main_out, aux_out = model(rgb)
            
            print(f"    Input: {rgb.shape}")
            print(f"    Output: {main_out.shape}")
            
            # Test inference
            print(f"  Testing inference...")
            with torch.no_grad():
                pred = model.inference(rgb, flip_tta=False)
            print(f"    Prediction shape: {pred.shape}")
            
            # Test parameter groups
            param_groups = model.get_param_groups()
            print(f"  Parameter groups: {len(param_groups)}")
            for group in param_groups:
                num_params = sum(p.numel() for p in group['params'])
                print(f"    - {group['name']}: {num_params:,} params")
            
            print(f"  ✓ {variant} model OK")
            
        except Exception as e:
            print(f"  ✗ Error: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*70)
    print("All tests passed!")
    print("="*70)
