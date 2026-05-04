"""
VisionMamba ICPRS Configuration
===============================
Production-grade configuration for Vision Mamba semantic segmentation on ICPRS.
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Dict
import os

_HERE = os.path.dirname(os.path.abspath(__file__))


@dataclass
class Config:
    """Master configuration for Vision Mamba training on ICPRS."""

    # ============== Paths ==============
    # Set POTSDAM_ROOT env var to point to your Potsdam or Vaihingen dataset path
    DATA_ROOT: str = os.environ.get("POTSDAM_ROOT", "data/Potsdam")  # set POTSDAM_ROOT env var or edit this path
    DATASET: str = "icprs"
    WEIGHTS_PATH: str = "auto"
    OUTPUT_DIR: str = os.path.join(_HERE, "..", "Comparison_Experiments_ICPRS_potsdam", "visionmamba_base_512")
    RESUME_PATH: str = ""

    # Vision Mamba Weights Directory
    VIM_WEIGHTS_DIR: str = os.path.join(_HERE, "weights")

    # ============== Model ==============
    NUM_CLASSES: int = 6
    IGNORE_INDEX: int = 255

    # Vision Mamba Variant: "tiny" | "small" | "base"
    VIM_VARIANT: str = "base"

    # Use fine-tuned weights if available (for tiny and small)
    USE_FINETUNED_WEIGHTS: bool = True

    # Model depth configuration
    VIM_DEPTHS: Tuple[int, ...] = (1,)

    # Embedding dimensions per variant
    VIM_DIMS_MAP: Dict[str, Tuple[int, ...]] = field(default_factory=lambda: {
        "tiny": (192,),
        "small": (384,),
        "base": (768,),
    })
    VIM_DIMS: Tuple[int, ...] = (192,)

    # Weight mapping: variant -> (base_weights, finetuned_weights)
    VIM_WEIGHTS_MAP: Dict[str, Dict[str, str]] = field(default_factory=lambda: {
        "tiny": {
            "finetuned": "vim_t_midclstok_ft_78p3acc.pth",
        },
        "small": {
            "finetuned": "vim_s_midclstok_ft_81p6acc.pth",
        },
        "base": {
            "base": "vim_b_midclstok_81p9acc.pth",
            "finetuned": "vim_b_midclstok_81p9acc.pth",
        },
    })

    # Drop path rate for regularization
    VIM_DROP_PATH: float = 0.0

    # Decoder configuration
    DECODER_CHANNELS: int = 256

    # Vision Mamba specific settings
    USE_RMS_NORM: bool = True
    VIM_FUSED_ADD_NORM: bool = True
    VIM_RESIDUAL_IN_FP32: bool = True
    VIM_IF_ROPE: bool = False
    VIM_BIMAMBA_TYPE: str = "v2"

    # ============== Train/Val Paths ==============
    TRAIN_IMG_DIR: List[str] = field(default_factory=lambda: ["Images"])
    TRAIN_MASK_DIR: List[str] = field(default_factory=lambda: ["Labels"])
    VAL_IMG_DIR: List[str] = field(default_factory=lambda: ["Images"])
    VAL_MASK_DIR: List[str] = field(default_factory=lambda: ["Labels"])

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
    RGB_MEAN: Tuple[float, ...] = (0.485, 0.456, 0.406)
    RGB_STD: Tuple[float, ...] = (0.229, 0.224, 0.225)

    # ============== Class Names ==============
    CLASS_NAMES: Tuple[str, ...] = (
        "Impervious",
        "Building",
        "LowVeg",
        "Tree",
        "Car",
        "Clutter",
    )
    ICPRS_CLASS_NAMES: Tuple[str, ...] = (
        "Impervious",
        "Building",
        "LowVeg",
        "Tree",
        "Car",
        "Clutter",
    )

    # ICPRS split settings (used when DATASET == "icprs")
    ICPRS_VAL_SPLIT: float = 0.2
    ICPRS_SEED: int = 42

    GPU_ID: int = 1

    # ============== Logging ==============
    LOG_INTERVAL: int = 250

    def __post_init__(self):
        """Resolve paths, dimensions, and weights."""
        variant = (self.VIM_VARIANT or "base").lower()
        valid_variants = list(self.VIM_DIMS_MAP.keys())
        if variant not in valid_variants:
            raise ValueError(f"Invalid variant '{variant}'. Must be one of {valid_variants}")
        self.VIM_DIMS = self.VIM_DIMS_MAP[variant]

        if not self.WEIGHTS_PATH or self.WEIGHTS_PATH.lower() == "auto":
            weight_type = "finetuned" if self.USE_FINETUNED_WEIGHTS else "base"
            weight_name = self.VIM_WEIGHTS_MAP.get(variant, {}).get(weight_type)
            if weight_name:
                self.WEIGHTS_PATH = os.path.join(self.VIM_WEIGHTS_DIR, weight_name)
                if not os.path.exists(self.WEIGHTS_PATH):
                    if weight_type == "finetuned":
                        weight_name = self.VIM_WEIGHTS_MAP[variant].get("base", "")
                        self.WEIGHTS_PATH = (
                            os.path.join(self.VIM_WEIGHTS_DIR, weight_name) if weight_name else ""
                        )
            else:
                self.WEIGHTS_PATH = ""

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
            [self.get_full_path(p) for p in self.TRAIN_MASK_DIR],
        )

    def get_val_paths(self):
        """Get validation image and mask directories."""
        return (
            [self.get_full_path(p) for p in self.VAL_IMG_DIR],
            [self.get_full_path(p) for p in self.VAL_MASK_DIR],
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
    print("=" * 60)
    print("VisionMamba ICPRS Configuration")
    print("=" * 60)
    print(f"Data Root: {cfg.DATA_ROOT}")
    print(f"Output Dir: {cfg.OUTPUT_DIR}")
    print(f"Resume Path: {cfg.RESUME_PATH}")
    print(f"Batch Size: {cfg.BATCH_SIZE}")
    print(f"Crop Size: {cfg.CROP_SIZE}")
    print(f"Max Iterations: {cfg.MAX_ITERS}")
    print(f"Num Classes: {cfg.NUM_CLASSES}")
    print(f"Vision Mamba Variant: {cfg.VIM_VARIANT}")
    print(f"Weights Path: {cfg.WEIGHTS_PATH}")
    print(f"Decoder Channels: {cfg.DECODER_CHANNELS}")
    print("=" * 60)
