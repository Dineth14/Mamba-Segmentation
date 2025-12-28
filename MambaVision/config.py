"""
UrbanMamba Configuration
========================
Production-grade configuration for semantic segmentation on LOVEDA dataset.
RGB-only MambaVision encoder with lightweight U-Net decoder.
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Dict
import os


@dataclass
class Config:
    """Master configuration for UrbanMamba training."""
    
    # ============== Paths ==============
    DATA_ROOT: str = "/storage2/ChangeDetection/Datasets/Loveda"
    WEIGHTS_PATH: str = "/storage2/ChangeDetection/NSST-mamba/mamba-segmentations/mamba_vision/UrbanMamba/weights/1k"
    OUTPUT_DIR: str = "/storage2/ChangeDetection/NSST-mamba/Mamba-Segmentation/Comparison_Experiments/mambavision_tiny2_256"
    RESUME_PATH: str = ""
    MAMBAVISION_WEIGHTS_DIR: str = "/storage2/ChangeDetection/NSST-mamba/mamba-segmentations/mamba_vision/UrbanMamba/weights/1k"
    MAMBAVISION_WEIGHTS_MAP: Dict[str, str] = field(default_factory=lambda: {
        "tiny": "mambavision_tiny_1k.pth.tar",
        "tiny2": "mambavision_tiny2_1k.pth.tar",
        "small": "mambavision_small_1k.pth.tar",
        "base": "mambavision_base_1k.pth.tar",
        "large": "mambavision_large_1k.pth.tar",
        "large2": "mambavision_large2_1k.pth.tar",
     
    })
    MAMBAVISION_DEPTHS_MAP: Dict[str, Tuple[int, ...]] = field(default_factory=lambda: {
        "tiny": (1, 3, 8, 4),
        "tiny2": (1, 3, 11, 4),
        "small": (3, 3, 7, 5),
        "base": (3, 3, 10, 5),
        "large": (3, 3, 10, 5),
        "large2": (3, 3, 12, 5),

    })
    MAMBAVISION_DIMS_MAP: Dict[str, Tuple[int, ...]] = field(default_factory=lambda: {
        "tiny": (80, 160, 320, 640),
        "tiny2": (80, 160, 320, 640),
        "small": (96, 192, 384, 768),
        "base": (128, 256, 512, 1024),
        "large": (196, 392, 784, 1568),
        "large2": (196, 392, 784, 1568),
        
    })
    MAMBAVISION_DROP_PATH_MAP: Dict[str, float] = field(default_factory=lambda: {
        "tiny": 0.2,
        "tiny2": 0.2,
        "small": 0.2,
        "base": 0.3,
        "large": 0.3,
        "large2": 0.3,
     
    })
    
    # Train/Val paths (relative to DATA_ROOT)
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
    CROP_SIZE: int = 256  # Random crop size during training
    MAX_ITERS: int = 100000  # Total training iterations
    VAL_INTERVAL: int = 5000  # Validate every N iterations
    NUM_WORKERS: int = 8  # DataLoader workers
    
    # ============== Model ==============
    NUM_CLASSES: int = 7  # LOVEDA: Background, Building, Road, Water, Barren, Forest, Agricultural
    IGNORE_INDEX: int = 255  # Label to ignore in loss computation
    
    # MambaVision variant selection: "tiny" | "tiny2" | "small" | "base" |
    # "large" | "large2" 
    MAMBAVISION_VARIANT: str = "tiny2"
    # Auto-populated from maps based on MAMBAVISION_VARIANT
    MAMBAVISION_DEPTHS: Tuple[int, ...] = (1, 3, 11, 4)
    MAMBAVISION_DIMS: Tuple[int, ...] = (80, 160, 320, 640)
    MAMBAVISION_DROP_PATH: float = 0.2
    
    # Decoder
    DECODER_CHANNELS: int = 256
    
    # ============== Optimization ==============
    LR_BACKBONE: float = 6e-5  # MambaVision backbone learning rate
    LR_HEAD: float = 3e-4  # Decoder and other components learning rate
    WEIGHT_DECAY: float = 0.05
    POLY_POWER: float = 0.9  # Polynomial LR decay power
    
    # Mixed Precision
    USE_AMP: bool = True
    
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
        "Background",
        "Building", 
        "Road",
        "Water",
        "Barren",
        "Forest",
        "Agricultural"
    )
    
    GPU_ID: int = 1 # CUDA device index
    
    # ============== Logging ==============
    LOG_INTERVAL: int = 250
    
    def __post_init__(self):
        """Create output directories and resolve MambaVision variant."""
        variant = (self.MAMBAVISION_VARIANT or "base").lower()
        if variant not in self.MAMBAVISION_DEPTHS_MAP:
            raise ValueError(f"Unsupported MAMBAVISION_VARIANT '{self.MAMBAVISION_VARIANT}'")
        self.MAMBAVISION_DEPTHS = self.MAMBAVISION_DEPTHS_MAP.get(variant, self.MAMBAVISION_DEPTHS)
        self.MAMBAVISION_DIMS = self.MAMBAVISION_DIMS_MAP.get(variant, self.MAMBAVISION_DIMS)
        self.MAMBAVISION_DROP_PATH = self.MAMBAVISION_DROP_PATH_MAP.get(variant, self.MAMBAVISION_DROP_PATH)
        if self.WEIGHTS_PATH and os.path.isdir(self.WEIGHTS_PATH):
            self.MAMBAVISION_WEIGHTS_DIR = self.WEIGHTS_PATH
            self.WEIGHTS_PATH = ""
        if not self.WEIGHTS_PATH or self.WEIGHTS_PATH.lower() == "auto":
            weight_name = self.MAMBAVISION_WEIGHTS_MAP.get(variant, "")
            self.WEIGHTS_PATH = os.path.join(self.MAMBAVISION_WEIGHTS_DIR, weight_name) if weight_name else ""
        os.makedirs(self.OUTPUT_DIR, exist_ok=True)
        os.makedirs(os.path.join(self.OUTPUT_DIR, "checkpoints"), exist_ok=True)
        os.makedirs(os.path.join(self.OUTPUT_DIR, "logs"), exist_ok=True)
        os.makedirs(os.path.join(self.OUTPUT_DIR, "tensorboard"), exist_ok=True)
        os.makedirs(os.path.join(self.OUTPUT_DIR, "val_preds"), exist_ok=True)
    
    def get_full_path(self, relative_path: str) -> str:
        """Get full path from relative path."""
        return os.path.join(self.DATA_ROOT, relative_path)
    
    def get_train_paths(self):
        """Get full training paths."""
        return (
            [self.get_full_path(p) for p in self.TRAIN_IMG_DIR],
            [self.get_full_path(p) for p in self.TRAIN_MASK_DIR]
        )
    
    def get_val_paths(self):
        """Get full validation paths."""
        return (
            [self.get_full_path(p) for p in self.VAL_IMG_DIR],
            [self.get_full_path(p) for p in self.VAL_MASK_DIR]
        )


# Global config instance
cfg = Config()


if __name__ == "__main__":
    # Print configuration summary
    print("=" * 60)
    print("UrbanMamba (MambaVision) Configuration")
    print("=" * 60)
    print(f"Data Root: {cfg.DATA_ROOT}")
    print(f"Output Dir: {cfg.OUTPUT_DIR}")
    print(f"Resume Path: {cfg.RESUME_PATH}")
    print(f"Batch Size: {cfg.BATCH_SIZE}")
    print(f"Crop Size: {cfg.CROP_SIZE}")
    print(f"Max Iterations: {cfg.MAX_ITERS}")
    print(f"Num Classes: {cfg.NUM_CLASSES}")
    print(f"MambaVision Variant: {cfg.MAMBAVISION_VARIANT}")
    print(f"MambaVision Weights: {cfg.WEIGHTS_PATH}")
    print(f"MambaVision Depths: {cfg.MAMBAVISION_DEPTHS}")
    print(f"MambaVision Dims: {cfg.MAMBAVISION_DIMS}")
    print(f"MambaVision DropPath: {cfg.MAMBAVISION_DROP_PATH}")
    print(f"LR Backbone: {cfg.LR_BACKBONE}")
    print(f"LR Head: {cfg.LR_HEAD}")
    print(f"Use AMP: {cfg.USE_AMP}")
    print("=" * 60)
