"""
UrbanMamba Configuration
========================
Production-grade configuration for semantic segmentation on LOVEDA dataset.
RGB-only VMamba encoder with lightweight U-Net decoder.
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Dict
import os


@dataclass
class Config:
    """Master configuration for UrbanMamba training."""
    
    # ============== Paths ==============
    DATA_ROOT: str = "/storage2/ChangeDetection/Datasets/Loveda"
    WEIGHTS_PATH: str = "auto"
    OUTPUT_DIR: str = "/storage2/ChangeDetection/NSST-mamba/Mamba-Segmentation/Comparison_Experiments/Vmamb_base_512/"
    RESUME_PATH: str = ""
    VMAMBA_WEIGHT_SET: str = "imagenet1k"  # "imagenet1k" | "ade20k" | "vanilla_ade20k"
    VMAMBA_WEIGHTS_DIR: str = ""
    VMAMBA_WEIGHTS_DIR_MAP: Dict[str, str] = field(default_factory=lambda: {
        "imagenet1k": "/storage2/ChangeDetection/NSST-mamba/Mamba-Segmentation/VMamba/Vmamba_weights/ImageNet-1K",
        "ade20k": "/storage2/ChangeDetection/NSST-mamba/Mamba-Segmentation/VMamba/Vmamba_weights/ADE20K_weights",
        "vanilla_ade20k": "/storage2/ChangeDetection/NSST-mamba/Mamba-Segmentation/VMamba/Vmamba_weights/vanilla-vmamba-ADE20K",
    })
    VMAMBA_WEIGHTS_FILE_MAP: Dict[str, Dict[str, str]] = field(default_factory=lambda: {
        "imagenet1k": {
            "tiny": "vssmtiny_dp01_ckpt_epoch_292.pth",
            "small": "vssmsmall_dp03_ckpt_epoch_238.pth",
            "base": "vssmbase_dp06_ckpt_epoch_241.pth",
        },
        "ade20k": {
            "tiny": "upernet_vssm_4xb4-160k_ade20k-512x512_tiny_s_iter_160000.pth",
            "small": "upernet_vssm_4xb4-160k_ade20k-512x512_small_iter_144000.pth",
            "base": "upernet_vssm_4xb4-160k_ade20k-512x512_base_iter_160000.pth",
        },
        "vanilla_ade20k": {
            "tiny": "vssmtiny_upernet_4xb4-160k_ade20k-512x512_iter_160000.pth",
            "small": "vssmsmall_upernet_4xb4-160k_ade20k-512x512_iter_160000.pth",
            "base": "vssmbase_upernet_4xb4-160k_ade20k-512x512_iter_128000.pth",
        },
    })
    VMAMBA_DEPTHS_MAP: Dict[str, Tuple[int, ...]] = field(default_factory=lambda: {
        "tiny": (2, 2, 9, 2),
        "small": (2, 2, 27, 2),
        "base": (2, 2, 27, 2),
    })
    VMAMBA_DIMS_MAP: Dict[str, Tuple[int, ...]] = field(default_factory=lambda: {
        "tiny": (96, 192, 384, 768),
        "small": (96, 192, 384, 768),
        "base": (128, 256, 512, 1024),
    })
    VMAMBA_DROP_PATH_MAP: Dict[str, float] = field(default_factory=lambda: {
        "tiny": 0.2,
        "small": 0.3,
        "base": 0.6,
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
    CROP_SIZE: int = 512  # Random crop size during training
    MAX_ITERS: int = 50000  # Total training iterations
    VAL_INTERVAL: int = 2500  # Validate every N iterations
    NUM_WORKERS: int = 8  # DataLoader workers
    PREFETCH_FACTOR: int = 4  # Batches prefetched per worker
    PIN_MEMORY: bool = True
    PERSISTENT_WORKERS: bool = True
    
    # ============== Model ==============
    NUM_CLASSES: int = 7  # LOVEDA: Background, Building, Road, Water, Barren, Forest, Agricultural
    IGNORE_INDEX: int = 255  # Label to ignore in loss computation
    
    # VMamba variant selection: "tiny" | "small" | "base"
    VMAMBA_VARIANT: str = "base"
    # Auto-populated from maps based on VMAMBA_VARIANT
    VMAMBA_DEPTHS: Tuple[int, ...] = (2, 2, 27, 2)
    VMAMBA_DIMS: Tuple[int, ...] = (128, 256, 512, 1024)
    VMAMBA_DROP_PATH: float = 0.6
    # VMamba backbone config (auto-resolved from weight set + variant)
    VMAMBA_SSM_D_STATE: int = 16
    VMAMBA_SSM_RATIO: float = 2.0
    VMAMBA_SSM_DT_RANK: str = "auto"
    VMAMBA_SSM_ACT_LAYER: str = "silu"
    VMAMBA_SSM_CONV: int = 3
    VMAMBA_SSM_CONV_BIAS: bool = True
    VMAMBA_SSM_DROP_RATE: float = 0.0
    VMAMBA_SSM_INIT: str = "v0"
    VMAMBA_FORWARD_TYPE: str = "v0"
    VMAMBA_MLP_RATIO: float = 0.0
    VMAMBA_GMLP: bool = False
    VMAMBA_NORM_LAYER: str = "ln"
    VMAMBA_DOWNSAMPLE_VERSION: str = "v1"
    VMAMBA_PATCHEMBED_VERSION: str = "v1"
    
    # Decoder
    DECODER_CHANNELS: int = 256
    
    # ============== Optimization ==============
    LR_BACKBONE: float = 6e-5  # VMamba backbone learning rate
    LR_HEAD: float = 3e-4  # Decoder and other components learning rate
    WEIGHT_DECAY: float = 0.05
    POLY_POWER: float = 0.9  # Polynomial LR decay power
    
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
        "Background",
        "Building", 
        "Road",
        "Water",
        "Barren",
        "Forest",
        "Agricultural"
    )
    
    GPU_ID: int = 0 # CUDA device index
    
    # ============== Logging ==============
    LOG_INTERVAL: int = 250
    
    def __post_init__(self):
        """Create output directories and resolve VMamba variant."""
        variant = (self.VMAMBA_VARIANT or "base").lower()
        if variant not in ("tiny", "small", "base"):
            raise ValueError(f"Unsupported VMAMBA_VARIANT '{self.VMAMBA_VARIANT}'")
        self.VMAMBA_DEPTHS = self.VMAMBA_DEPTHS_MAP.get(variant, self.VMAMBA_DEPTHS)
        self.VMAMBA_DIMS = self.VMAMBA_DIMS_MAP.get(variant, self.VMAMBA_DIMS)
        self.VMAMBA_DROP_PATH = self.VMAMBA_DROP_PATH_MAP.get(variant, self.VMAMBA_DROP_PATH)
        weight_set = (self.VMAMBA_WEIGHT_SET or "imagenet1k").lower()
        if weight_set not in self.VMAMBA_WEIGHTS_DIR_MAP:
            raise ValueError(f"Unsupported VMAMBA_WEIGHT_SET '{self.VMAMBA_WEIGHT_SET}'")
        self.VMAMBA_WEIGHTS_DIR = self.VMAMBA_WEIGHTS_DIR_MAP[weight_set]
        if self.WEIGHTS_PATH and os.path.isdir(self.WEIGHTS_PATH):
            self.VMAMBA_WEIGHTS_DIR = self.WEIGHTS_PATH
            self.WEIGHTS_PATH = ""
        if not self.WEIGHTS_PATH or self.WEIGHTS_PATH.lower() == "auto":
            weight_map = self.VMAMBA_WEIGHTS_FILE_MAP.get(weight_set, {})
            weight_name = weight_map.get(variant, "")
            self.WEIGHTS_PATH = os.path.join(self.VMAMBA_WEIGHTS_DIR, weight_name) if weight_name else ""
        backbone_cfg = {
            "ssm_d_state": 16,
            "ssm_ratio": 2.0,
            "ssm_dt_rank": "auto",
            "ssm_act_layer": "silu",
            "ssm_conv": 3,
            "ssm_conv_bias": True,
            "ssm_drop_rate": 0.0,
            "ssm_init": "v0",
            "forward_type": "v0",
            "mlp_ratio": 0.0,
            "gmlp": False,
            "norm_layer": "ln",
            "downsample_version": "v1",
            "patchembed_version": "v1",
        }
        if weight_set == "ade20k":
            backbone_cfg.update({
                "ssm_d_state": 1,
                "ssm_ratio": 2.0,
                "ssm_conv_bias": False,
                "forward_type": "v05_noz",
                "mlp_ratio": 4.0,
                "norm_layer": "ln2d",
                "downsample_version": "v3",
                "patchembed_version": "v2",
            })
            if variant == "tiny":
                backbone_cfg["ssm_ratio"] = 1.0
        elif weight_set == "vanilla_ade20k":
            backbone_cfg.update({
                "ssm_d_state": 16,
                "ssm_ratio": 2.0,
                "ssm_conv_bias": True,
                "forward_type": "v0",
                "mlp_ratio": 0.0,
                "norm_layer": "ln",
                "downsample_version": "v1",
                "patchembed_version": "v1",
            })
        self.VMAMBA_SSM_D_STATE = backbone_cfg["ssm_d_state"]
        self.VMAMBA_SSM_RATIO = backbone_cfg["ssm_ratio"]
        self.VMAMBA_SSM_DT_RANK = backbone_cfg["ssm_dt_rank"]
        self.VMAMBA_SSM_ACT_LAYER = backbone_cfg["ssm_act_layer"]
        self.VMAMBA_SSM_CONV = backbone_cfg["ssm_conv"]
        self.VMAMBA_SSM_CONV_BIAS = backbone_cfg["ssm_conv_bias"]
        self.VMAMBA_SSM_DROP_RATE = backbone_cfg["ssm_drop_rate"]
        self.VMAMBA_SSM_INIT = backbone_cfg["ssm_init"]
        self.VMAMBA_FORWARD_TYPE = backbone_cfg["forward_type"]
        self.VMAMBA_MLP_RATIO = backbone_cfg["mlp_ratio"]
        self.VMAMBA_GMLP = backbone_cfg["gmlp"]
        self.VMAMBA_NORM_LAYER = backbone_cfg["norm_layer"]
        self.VMAMBA_DOWNSAMPLE_VERSION = backbone_cfg["downsample_version"]
        self.VMAMBA_PATCHEMBED_VERSION = backbone_cfg["patchembed_version"]
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
    print("UrbanMamba Configuration")
    print("=" * 60)
    print(f"Data Root: {cfg.DATA_ROOT}")
    print(f"Output Dir: {cfg.OUTPUT_DIR}")
    print(f"Resume Path: {cfg.RESUME_PATH}")
    print(f"Batch Size: {cfg.BATCH_SIZE}")
    print(f"Crop Size: {cfg.CROP_SIZE}")
    print(f"Max Iterations: {cfg.MAX_ITERS}")
    print(f"Num Classes: {cfg.NUM_CLASSES}")
    print(f"VMamba Variant: {cfg.VMAMBA_VARIANT}")
    print(f"VMamba Weights: {cfg.WEIGHTS_PATH}")
    print(f"VMamba Depths: {cfg.VMAMBA_DEPTHS}")
    print(f"VMamba Dims: {cfg.VMAMBA_DIMS}")
    print(f"VMamba DropPath: {cfg.VMAMBA_DROP_PATH}")
    print(f"LR Backbone: {cfg.LR_BACKBONE}")
    print(f"LR Head: {cfg.LR_HEAD}")
    print(f"Use AMP: {cfg.USE_AMP}")
    print(f"Allow TF32: {cfg.ALLOW_TF32}")
    print(f"CUDNN Benchmark: {cfg.CUDNN_BENCHMARK}")
    print(f"Matmul Precision: {cfg.MATMUL_PRECISION}")
    print("=" * 60)
