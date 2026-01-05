"""
Vision Mamba Configuration
==========================
Production-grade configuration for Vision Mamba semantic segmentation.
Supports all Vision Mamba variants (tiny, small, base) with proper weight loading.
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional
import os


@dataclass
class Config:
    """Master configuration for Vision Mamba training with multi-variant support."""
    
    # ============== Paths ==============
    DATA_ROOT: str = "/storage2/ChangeDetection/Datasets/Loveda"
    WEIGHTS_PATH: str = "auto"
    OUTPUT_DIR: str = "/storage2/ChangeDetection/NSST-mamba/Mamba-Segmentation/Comparison_Experiments/VisionMamba_tiny_1block/"
    RESUME_PATH: str = ""
    
    # Vision Mamba Weights Directory
    VIM_WEIGHTS_DIR: str = "/storage2/ChangeDetection/NSST-mamba/Mamba-Segmentation/VisionMamba/weights"
    
    # ============== Model ==============
    NUM_CLASSES: int = 7
    IGNORE_INDEX: int = 255
    
    # Vision Mamba Variant: "tiny" | "small" | "base"
    # Available weights:
    # - tiny: vim_t_midclstok_76p1acc.pth (76.1% acc) | vim_t_midclstok_ft_78p3acc.pth (78.3% acc, fine-tuned)
    # - small: vim_s_midclstok_80p5acc.pth (80.5% acc) | vim_s_midclstok_ft_81p6acc.pth (81.6% acc, fine-tuned)
    # - base: vim_b_midclstok_81p9acc.pth (81.9% acc)
    VIM_VARIANT: str = "tiny"
    
    # Use fine-tuned weights if available (for tiny and small)
    USE_FINETUNED_WEIGHTS: bool = True
    
    # Model depth configuration
    # For Vision Mamba, this typically refers to the number of Mamba blocks
    # Set to (1,) for single-block encoder, (24,) for full depth
    VIM_DEPTHS: Tuple[int, ...] = (1,)  
    
    # Embedding dimensions per variant
    VIM_DIMS_MAP: Dict[str, Tuple[int, ...]] = field(default_factory=lambda: {
        "tiny": (192,),
        "small": (384,),
        "base": (768,),
    })
    # Will be resolved in post_init
    VIM_DIMS: Tuple[int, ...] = (192,)
    
    # Weight mapping: variant -> (base_weights, finetuned_weights)
    VIM_WEIGHTS_MAP: Dict[str, Dict[str, str]] = field(default_factory=lambda: {
        "tiny": {
            "base": "vim_t_midclstok_76p1acc.pth",
            "finetuned": "vim_t_midclstok_ft_78p3acc.pth",
        },
        "small": {
            "base": "vim_s_midclstok_80p5acc.pth",
            "finetuned": "vim_s_midclstok_ft_81p6acc.pth",
        },
        "base": {
            "base": "vim_b_midclstok_81p9acc.pth",
            "finetuned": "vim_b_midclstok_81p9acc.pth",  # No fine-tuned version available
        },
    }) 
    
    # Drop path rate for regularization
    VIM_DROP_PATH: float = 0.0  # No drop path for single block usually
    
    # Decoder configuration
    DECODER_CHANNELS: int = 256
    
    # Vision Mamba specific settings
    USE_RMS_NORM: bool = True  # Vim uses RMSNorm
    VIM_FUSED_ADD_NORM: bool = True  # Use fused add+norm for efficiency
    VIM_RESIDUAL_IN_FP32: bool = True  # Keep residuals in FP32
    VIM_IF_ROPE: bool = False  # Vision Mamba uses absolute positional embeddings
    VIM_BIMAMBA_TYPE: str = "v2"  # BiMamba v2 architecture
    
    # ============== Train/Val Paths ==============
    TRAIN_IMG_DIR: List[str] = field(default_factory=lambda: [
        "Train/Train/Urban/images_png",
        "Train/Train/Rural/images_png"
    ])
    TRAIN_MASK_DIR: List[str] = field(default_factory=lambda: [
        "Train/Train/Urban/masks_png",
        "Train/Train/Rural/masks_png"
    ])
    
    VAL_IMG_DIR: List[str] = field(default_factory=lambda: [
        "Val/Val/Urban/images_png",
        "Val/Val/Rural/images_png"
    ])
    VAL_MASK_DIR: List[str] = field(default_factory=lambda: [
        "Val/Val/Urban/masks_png",
        "Val/Val/Rural/masks_png"
    ])
    
    # ============== Training ==============
    BATCH_SIZE: int = 4 
    CROP_SIZE: int = 512
    MAX_ITERS: int = 50000 
    VAL_INTERVAL: int = 2500 
    NUM_WORKERS: int = 8 
    PREFETCH_FACTOR: int = 4 
    PIN_MEMORY: bool = True
    PERSISTENT_WORKERS: bool = True
    
    # ============== Optimization ==============
    LR_BACKBONE: float = 6e-5 
    LR_HEAD: float = 3e-4 
    WEIGHT_DECAY: float = 0.05
    POLY_POWER: float = 0.9
    
    # Mixed Precision
    USE_AMP: bool = True
    ALLOW_TF32: bool = True
    CUDNN_BENCHMARK: bool = True
    MATMUL_PRECISION: str = "high"
    
    # Gradient clipping
    GRAD_CLIP: float = 1.0
    
    # Focal Loss
    FOCAL_GAMMA: float = 2.0
    
    # ============== Data Augmentation ==============
    HORIZONTAL_FLIP_PROB: float = 0.5
    VERTICAL_FLIP_PROB: float = 0.5
    ROTATE_90_PROB: float = 0.5
    COLOR_JITTER: bool = True
    
    # ============== Normalization ==============
    # ImageNet stats for RGB
    RGB_MEAN: Tuple[float, ...] = (0.485, 0.456, 0.406)
    RGB_STD: Tuple[float, ...] = (0.229, 0.224, 0.225)
    
    # ============== Class Names ==============
    CLASS_NAMES: Tuple[str, ...] = (
        "Background", "Building", "Road", "Water", "Barren", "Forest", "Agricultural"
    )
    
    GPU_ID: int = 0
    
    # ============== Logging ==============
    LOG_INTERVAL: int = 250
    
    def __post_init__(self):
        """Resolve paths, dimensions, and weights."""
        variant = (self.VIM_VARIANT or "tiny").lower()
        
        # Validate variant
        valid_variants = list(self.VIM_DIMS_MAP.keys())
        if variant not in valid_variants:
            raise ValueError(f"Invalid variant '{variant}'. Must be one of {valid_variants}")
        
        # Resolve embedding dimensions
        self.VIM_DIMS = self.VIM_DIMS_MAP[variant]
        
        # Resolve weights path
        if not self.WEIGHTS_PATH or self.WEIGHTS_PATH.lower() == "auto":
            weight_type = "finetuned" if self.USE_FINETUNED_WEIGHTS else "base"
            weight_name = self.VIM_WEIGHTS_MAP.get(variant, {}).get(weight_type)
            
            if weight_name:
                self.WEIGHTS_PATH = os.path.join(self.VIM_WEIGHTS_DIR, weight_name)
                if not os.path.exists(self.WEIGHTS_PATH):
                    # Fallback to base weights if finetuned doesn't exist
                    if weight_type == "finetuned":
                        weight_name = self.VIM_WEIGHTS_MAP[variant]["base"]
                        self.WEIGHTS_PATH = os.path.join(self.VIM_WEIGHTS_DIR, weight_name)
            else:
                self.WEIGHTS_PATH = ""
        
        # Create output directories
        os.makedirs(self.OUTPUT_DIR, exist_ok=True)
        os.makedirs(os.path.join(self.OUTPUT_DIR, "checkpoints"), exist_ok=True)
        os.makedirs(os.path.join(self.OUTPUT_DIR, "logs"), exist_ok=True)
        os.makedirs(os.path.join(self.OUTPUT_DIR, "tensorboard"), exist_ok=True)
        os.makedirs(os.path.join(self.OUTPUT_DIR, "val_preds"), exist_ok=True)
    
    def get_full_path(self, relative_path: str) -> str:
        """Get full path for a data directory."""
        return os.path.join(self.DATA_ROOT, relative_path)
    
    def get_train_paths(self):
        """Get training image and mask directories."""
        return (
            [self.get_full_path(p) for p in self.TRAIN_IMG_DIR],
            [self.get_full_path(p) for p in self.TRAIN_MASK_DIR]
        )
    
    def get_val_paths(self):
        """Get validation image and mask directories."""
        return (
            [self.get_full_path(p) for p in self.VAL_IMG_DIR],
            [self.get_full_path(p) for p in self.VAL_MASK_DIR]
        )
    
    def get_model_config(self) -> Dict:
        """Get model configuration as a dictionary."""
        return {
            "num_classes": self.NUM_CLASSES,
            "variant": self.VIM_VARIANT,
            "depths": self.VIM_DEPTHS,
            "dims": self.VIM_DIMS,
            "drop_path_rate": self.VIM_DROP_PATH,
            "decoder_channels": self.DECODER_CHANNELS,
            "use_rms_norm": self.USE_RMS_NORM,
            "fused_add_norm": self.VIM_FUSED_ADD_NORM,
            "residual_in_fp32": self.VIM_RESIDUAL_IN_FP32,
            "if_rope": self.VIM_IF_ROPE,
            "bimamba_type": self.VIM_BIMAMBA_TYPE,
        }


# Global config instance
cfg = Config()


if __name__ == "__main__":
    print("=" * 70)
    print("Vision Mamba Segmentation Configuration")
    print("=" * 70)
    print(f"Variant:              {cfg.VIM_VARIANT}")
    print(f"Available Variants:   tiny (192), small (384), base (768)")
    print(f"Depths:               {cfg.VIM_DEPTHS}")
    print(f"Dims:                 {cfg.VIM_DIMS}")
    print(f"Drop Path Rate:       {cfg.VIM_DROP_PATH}")
    print(f"Use Finetuned:        {cfg.USE_FINETUNED_WEIGHTS}")
    print(f"Weights Path:         {cfg.WEIGHTS_PATH}")
    print(f"Weights Exist:        {os.path.exists(cfg.WEIGHTS_PATH) if cfg.WEIGHTS_PATH else False}")
    print(f"Output Dir:           {cfg.OUTPUT_DIR}")
    print(f"Data Root:            {cfg.DATA_ROOT}")
    print(f"Classes:              {cfg.NUM_CLASSES}")
    print("=" * 70)
    print("\nAvailable Weights in {0}:".format(cfg.VIM_WEIGHTS_DIR))
    if os.path.exists(cfg.VIM_WEIGHTS_DIR):
        for w in sorted(os.listdir(cfg.VIM_WEIGHTS_DIR)):
            if w.endswith('.pth'):
                weight_path = os.path.join(cfg.VIM_WEIGHTS_DIR, w)
                size_mb = os.path.getsize(weight_path) / (1024**2)
                print(f"  - {w} ({size_mb:.1f} MB)")
    print("=" * 70)
