"""
Spatial-Mamba ICPRS Configuration
=================================
Production-grade configuration for semantic segmentation on ICPRS (ISPRS).
RGB-only Spatial-Mamba encoder with lightweight U-Net decoder.
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Dict
import os

_HERE = os.path.dirname(os.path.abspath(__file__))


@dataclass
class Config:
    """Master configuration for Spatial-Mamba training on ICPRS."""

    # ============== Paths ==============
    # Set POTSDAM_ROOT to your Vaihingen path to use Vaihingen instead
    DATA_ROOT: str = os.environ.get("POTSDAM_ROOT", "data/Potsdam")  # set POTSDAM_ROOT env var or edit this path
    DATASET: str = "icprs"
    WEIGHTS_PATH: str = "auto"
    OUTPUT_DIR: str = os.path.join(_HERE, "..", "Comparison_Experiments_ICPRS_potsdam", "spatialmamba_base_512")
    RESUME_PATH: str = ""
    SPATIALMAMBA_WEIGHTS_DIR: str = os.path.join(_HERE, "weights", "imageNet1K")
    SPATIALMAMBA_WEIGHTS_MAP: Dict[str, str] = field(default_factory=lambda: {
        "tiny": "spatialmamba_tiny_224_1k.pth",
        "small": "spatialmamba_small_224_1k.pth",
        "base": "spatialmamba_base_224_1k.pth",
    })
    SPATIALMAMBA_DEPTHS_MAP: Dict[str, Tuple[int, ...]] = field(default_factory=lambda: {
        "tiny": (2, 4, 8, 4),
        "small": (2, 4, 21, 5),
        "base": (2, 4, 21, 5),
    })
    SPATIALMAMBA_DIMS_MAP: Dict[str, Tuple[int, ...]] = field(default_factory=lambda: {
        "tiny": (64, 128, 256, 512),
        "small": (64, 128, 256, 512),
        "base": (96, 192, 384, 768),
    })
    SPATIALMAMBA_DROP_PATH_MAP: Dict[str, float] = field(default_factory=lambda: {
        "tiny": 0.2,
        "small": 0.3,
        "base": 0.5,
    })
    SPATIALMAMBA_D_STATE: int = 1
    SPATIALMAMBA_DT_INIT: str = "random"
    SPATIALMAMBA_MLP_RATIO: float = 4.0

    # Train/Val paths (relative to DATA_ROOT)
    TRAIN_IMG_DIR: List[str] = field(default_factory=lambda: ["Images"])
    TRAIN_MASK_DIR: List[str] = field(default_factory=lambda: ["Labels"])
    VAL_IMG_DIR: List[str] = field(default_factory=lambda: ["Images"])
    VAL_MASK_DIR: List[str] = field(default_factory=lambda: ["Labels"])

    # ============== Training ==============
    BATCH_SIZE: int = 8
    CROP_SIZE: int = 512 # Random crop size during training
    MAX_ITERS: int = 25000  # Total training iterations
    VAL_INTERVAL: int = 2500  # Validate every N iterations
    NUM_WORKERS: int = 16  # DataLoader workers
    PREFETCH_FACTOR: int = 4  # Batches prefetched per worker
    PIN_MEMORY: bool = True
    PERSISTENT_WORKERS: bool = True

    # ============== Model ==============
    NUM_CLASSES: int = 6
    IGNORE_INDEX: int = 255  # Label to ignore in loss computation

    # Spatial-Mamba variant selection: "tiny" | "small" | "base"
    SPATIALMAMBA_VARIANT: str = "base"
    # Auto-populated from maps based on SPATIALMAMBA_VARIANT
    SPATIALMAMBA_DEPTHS: Tuple[int, ...] = (2, 4, 21, 5)
    SPATIALMAMBA_DIMS: Tuple[int, ...] = (96, 192, 384, 768)
    SPATIALMAMBA_DROP_PATH: float = 0.5
    # Decoder
    DECODER_CHANNELS: int = 256

    # ============== Optimization ==============
    LR_BACKBONE: float = 3e-5  # Spatial-Mamba backbone learning rate
    LR_HEAD: float = 1e-4  # Decoder and other components learning rate
    WEIGHT_DECAY: float = 0.05
    BOUNDARY_START_ITER: int = 10000
    BOUNDARY_WEIGHT: float = 0.5
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

    GPU_ID: int = 1  # CUDA device index

    # ============== Logging ==============
    LOG_INTERVAL: int = 250

    def __post_init__(self) -> None:
        """Create output directories and resolve Spatial-Mamba variant."""
        variant = (self.SPATIALMAMBA_VARIANT or "base").lower()
        if variant not in ("tiny", "small", "base"):
            raise ValueError(f"Unsupported SPATIALMAMBA_VARIANT '{self.SPATIALMAMBA_VARIANT}'")
        self.SPATIALMAMBA_DEPTHS = self.SPATIALMAMBA_DEPTHS_MAP.get(variant, self.SPATIALMAMBA_DEPTHS)
        self.SPATIALMAMBA_DIMS = self.SPATIALMAMBA_DIMS_MAP.get(variant, self.SPATIALMAMBA_DIMS)
        self.SPATIALMAMBA_DROP_PATH = self.SPATIALMAMBA_DROP_PATH_MAP.get(variant, self.SPATIALMAMBA_DROP_PATH)
        if self.WEIGHTS_PATH and os.path.isdir(self.WEIGHTS_PATH):
            self.SPATIALMAMBA_WEIGHTS_DIR = self.WEIGHTS_PATH
            self.WEIGHTS_PATH = ""
        if not self.WEIGHTS_PATH or self.WEIGHTS_PATH.lower() == "auto":
            weight_name = self.SPATIALMAMBA_WEIGHTS_MAP.get(variant, "")
            self.WEIGHTS_PATH = (
                os.path.join(self.SPATIALMAMBA_WEIGHTS_DIR, weight_name) if weight_name else ""
            )
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
            [self.get_full_path(p) for p in self.TRAIN_MASK_DIR],
        )

    def get_val_paths(self):
        """Get full validation paths."""
        return (
            [self.get_full_path(p) for p in self.VAL_IMG_DIR],
            [self.get_full_path(p) for p in self.VAL_MASK_DIR],
        )


# Global config instance
cfg = Config()


if __name__ == "__main__":
    print("=" * 60)
    print("UrbanMamba (Spatial-Mamba) ICPRS Configuration")
    print("=" * 60)
    print(f"Data Root: {cfg.DATA_ROOT}")
    print(f"Output Dir: {cfg.OUTPUT_DIR}")
    print(f"Resume Path: {cfg.RESUME_PATH}")
    print(f"Batch Size: {cfg.BATCH_SIZE}")
    print(f"Crop Size: {cfg.CROP_SIZE}")
    print(f"Max Iterations: {cfg.MAX_ITERS}")
    print(f"Num Classes: {cfg.NUM_CLASSES}")
    print(f"Spatial-Mamba Variant: {cfg.SPATIALMAMBA_VARIANT}")
    print(f"Spatial-Mamba Weights: {cfg.WEIGHTS_PATH}")
    print(f"Spatial-Mamba Depths: {cfg.SPATIALMAMBA_DEPTHS}")
    print(f"Spatial-Mamba Dims: {cfg.SPATIALMAMBA_DIMS}")
    print(f"Spatial-Mamba DropPath: {cfg.SPATIALMAMBA_DROP_PATH}")
    print(f"LR Backbone: {cfg.LR_BACKBONE}")
    print(f"LR Head: {cfg.LR_HEAD}")
    print(f"Boundary start iter: {cfg.BOUNDARY_START_ITER}")
    print(f"Boundary weight: {cfg.BOUNDARY_WEIGHT}")
    print(f"Use AMP: {cfg.USE_AMP}")
    print(f"Allow TF32: {cfg.ALLOW_TF32}")
    print(f"CUDNN Benchmark: {cfg.CUDNN_BENCHMARK}")
    print(f"Matmul Precision: {cfg.MATMUL_PRECISION}")
    print("=" * 60)
