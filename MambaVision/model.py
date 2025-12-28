"""
UrbanMamba Model Assembly
=========================
RGB-only NSSTMamba segmentation model combining:
- RGBEncoder (MambaVision variants)
- Lightweight convolutional decoder (U-Net style) with optional auxiliary head
"""

import torch
import torch.nn as nn
from typing import List, Tuple, Optional, Dict, Any

from config import Config
from encoders import RGBEncoder
from light_decoder import LightUNetDecoder


class Mambavision(nn.Module):
    """
    UrbanMamba Semantic Segmentation Model.
    
    RGB-only architecture combining:
    - MambaVision backbone for RGB spatial features
    - Lightweight U-Net decoder with optional deep supervision
    """
    
    def __init__(
        self,
        num_classes: int = 7,
        encoder_dims: Tuple[int, ...] = (96, 192, 384, 768),
        decoder_channels: int = 256,
        drop_path_rate: float = 0.3,
        encoder_depths: Tuple[int, ...] = (2, 2, 27, 2),
        pretrained_rgb: bool = True,
        weights_path: Optional[str] = None,
        encoder_variant: str = "small"
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.encoder_dims = encoder_dims
        
        # ============== Encoders ==============
        # RGB Encoder (MambaVision variant)
        self.rgb_encoder = RGBEncoder(
            depths=encoder_depths,
            dims=encoder_dims,
            drop_path_rate=drop_path_rate,
            pretrained=pretrained_rgb,
            weights_path=weights_path,
            model_variant=encoder_variant
        )
        
        # ============== Decoder ==============
        self.decoder = LightUNetDecoder(
            encoder_channels=encoder_dims,
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
            aux: (N, num_classes, H, W) Optional auxiliary output from stride-8 features
        """
        # Get target size for output
        target_size = rgb.shape[2:]
        
        # ============== Encode ==============
        # RGB features: [1/4, 1/8, 1/16, 1/32]
        rgb_features = self.rgb_encoder(rgb)
        fused_features = rgb_features
        
        # ============== Decode ==============
        # Pass fused fused pyramid through lightweight decoder
        main_out, aux_out = self.decoder(
            fused_features,
            target_size=target_size
        )
        
        if return_features:
            return main_out, aux_out, {
                'rgb_features': rgb_features,
                'fused_features': fused_features
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
            lr_backbone: Learning rate for MambaVision backbone
            lr_head: Learning rate for other components
            weight_decay: Weight decay
        
        Returns:
            List of parameter group dicts for optimizer
        """
        # Backbone parameters (MambaVision)
        backbone_params = []
        backbone_params_no_decay = []
        
        # Head parameters (Freq encoder, fusion, decoder)
        head_params = []
        head_params_no_decay = []
        
        # Parameters that shouldn't have weight decay
        no_decay_keywords = ['bias', 'bn', 'norm', 'ln']
        
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            
            # Check if it's a backbone parameter
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
        
        param_groups = [
            {'params': backbone_params, 'lr': lr_backbone, 'weight_decay': weight_decay, 'name': 'backbone'},
            {'params': backbone_params_no_decay, 'lr': lr_backbone, 'weight_decay': 0.0, 'name': 'backbone_no_decay'},
            {'params': head_params, 'lr': lr_head, 'weight_decay': weight_decay, 'name': 'head'},
            {'params': head_params_no_decay, 'lr': lr_head, 'weight_decay': 0.0, 'name': 'head_no_decay'},
        ]
        
        # Remove empty groups
        param_groups = [g for g in param_groups if len(g['params']) > 0]
        
        return param_groups
    
    def load_pretrained_rgb(self, weights_path: str) -> None:
        """Load pretrained weights for RGB encoder."""
        self.rgb_encoder.load_pretrained_weights(weights_path)
    
    @torch.no_grad()
    def inference(
        self,
        rgb: torch.Tensor,
        flip_tta: bool = False
    ) -> torch.Tensor:
        """
        Inference with optional test-time augmentation.
        
        Args:
            rgb: (N, 3, H, W) RGB input
            flip_tta: If True, use horizontal flip TTA
        
        Returns:
            (N, num_classes, H, W) Averaged predictions
        """
        self.eval()
        
        # Forward pass
        main_out, _ = self.forward(rgb)
        
        if flip_tta:
            # Horizontal flip
            rgb_flip = torch.flip(rgb, dims=[-1])
            main_flip, _ = self.forward(rgb_flip)
            main_flip = torch.flip(main_flip, dims=[-1])
            
            # Average
            main_out = (main_out + main_flip) / 2
        
        return main_out


def build_model(cfg: Config) -> Mambavision:
    """Build UrbanMamba model from config."""
    model = Mambavision(
        num_classes=cfg.NUM_CLASSES,
        encoder_dims=cfg.MAMBAVISION_DIMS,
        decoder_channels=cfg.DECODER_CHANNELS,
        drop_path_rate=cfg.MAMBAVISION_DROP_PATH,
        encoder_depths=cfg.MAMBAVISION_DEPTHS,
        pretrained_rgb=True,
        weights_path=cfg.WEIGHTS_PATH,
        encoder_variant=cfg.MAMBAVISION_VARIANT
    )
    return model


if __name__ == "__main__":
    # Test model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Testing on device: {device}")
    
    # Create model
    print("\nBuilding Mambavision model...")
    model = Mambavision(
        num_classes=7,
        encoder_dims=(96, 192, 384, 768),
        decoder_channels=256,
        pretrained_rgb=False  # Skip pretrained for test
    ).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Test forward pass
    print("\nTesting forward pass...")
    rgb = torch.randn(2, 3, 640, 640).to(device)
    with torch.cuda.amp.autocast(enabled=True):
        main_out, aux = model(rgb)
    
    print(f"RGB input: {rgb.shape}")
    print(f"Main output: {main_out.shape}")
    if aux is not None:
        print(f"Aux output: {aux.shape}")
    
    # Test parameter groups
    print("\nTesting parameter groups...")
    param_groups = model.get_param_groups(lr_backbone=6e-5, lr_head=3e-4)
    for group in param_groups:
        num_params = sum(p.numel() for p in group['params'])
        print(f"  {group['name']}: {num_params:,} params, lr={group['lr']}, wd={group['weight_decay']}")
    
    # Memory usage
    if device.type == 'cuda':
        print(f"\nGPU Memory: {torch.cuda.memory_allocated() / 1024**3:.2f} GB allocated")
        print(f"GPU Memory: {torch.cuda.memory_reserved() / 1024**3:.2f} GB reserved")
    
    print("\nAll tests passed!")
