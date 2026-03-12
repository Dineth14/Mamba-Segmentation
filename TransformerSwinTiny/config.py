"""
Transformer Swin-Tiny Configuration
===================================
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple
import os


@dataclass
class Config:
    DATA_ROOT: str = "/storage2/ChangeDetection/Datasets/Loveda"
    DATASET: str = "loveda"
    WEIGHTS_PATH: str = "auto"
    OUTPUT_DIR: str = (
        "/storage2/ChangeDetection/NSST-mamba/Mamba-Segmentation/"
        "Comparison_Experiments/transformer_swintiny_512"
    )
    RESUME_PATH: str = ""

    SWIN_WEIGHTS_DIR: str = (
        "/storage2/ChangeDetection/NSST-mamba/Mamba-Segmentation/"
        "Swin-Transformer/weights/imagenet1k"
    )
    SWIN_WEIGHTS_FILE_MAP: Dict[str, str] = field(default_factory=lambda: {
        "tiny": "swin_tiny_patch4_window7_224.pth",
    })

    MODEL_NAME: str = "TransformerSwinTiny"
    SWIN_VARIANT: str = "tiny"
    USE_PRETRAINED: bool = True
    DECODER_CHANNELS: int = 256

    NUM_CLASSES: int = 7
    IGNORE_INDEX: int = 255

    SWIN_PATCH_SIZE: int = 4
    SWIN_WINDOW_SIZE: int = 8
    SWIN_MLP_RATIO: float = 4.0
    SWIN_QKV_BIAS: bool = True
    SWIN_APE: bool = False
    SWIN_PATCH_NORM: bool = True

    SWIN_EMBED_DIM_MAP: Dict[str, int] = field(default_factory=lambda: {
        "tiny": 96,
    })
    SWIN_DEPTHS_MAP: Dict[str, Tuple[int, ...]] = field(default_factory=lambda: {
        "tiny": (2, 2, 6, 2),
    })
    SWIN_NUM_HEADS_MAP: Dict[str, Tuple[int, ...]] = field(default_factory=lambda: {
        "tiny": (3, 6, 12, 24),
    })
    SWIN_DROP_PATH_MAP: Dict[str, float] = field(default_factory=lambda: {
        "tiny": 0.2,
    })

    SWIN_EMBED_DIM: int = 96
    SWIN_DEPTHS: Tuple[int, ...] = (2, 2, 6, 2)
    SWIN_NUM_HEADS: Tuple[int, ...] = (3, 6, 12, 24)
    SWIN_DROP_PATH: float = 0.2

    TRAIN_IMG_DIR: List[str] = field(default_factory=lambda: [
        "Train/Train/Urban/images_png",
        "Train/Train/Rural/images_png",
    ])
    TRAIN_MASK_DIR: List[str] = field(default_factory=lambda: [
        "Train/Train/Urban/masks_png",
        "Train/Train/Rural/masks_png",
    ])
    VAL_IMG_DIR: List[str] = field(default_factory=lambda: [
        "Val/Val/Urban/images_png",
        "Val/Val/Rural/images_png",
    ])
    VAL_MASK_DIR: List[str] = field(default_factory=lambda: [
        "Val/Val/Urban/masks_png",
        "Val/Val/Rural/masks_png",
    ])

    BATCH_SIZE: int = 4
    CROP_SIZE: int = 512
    MAX_ITERS: int = 50000
    VAL_INTERVAL: int = 2500
    NUM_WORKERS: int = 8
    PIN_MEMORY: bool = True
    PERSISTENT_WORKERS: bool = True

    LR_BACKBONE: float = 6e-5
    LR_HEAD: float = 3e-4
    WEIGHT_DECAY: float = 0.05
    POLY_POWER: float = 0.9

    USE_AMP: bool = True
    GRAD_CLIP: float = 1.0

    FOCAL_GAMMA: float = 2.0
    BOUNDARY_WEIGHT: float = 0.5
    BOUNDARY_WARMUP_ITERS: int = 10000

    RGB_MEAN: Tuple[float, ...] = (0.485, 0.456, 0.406)
    RGB_STD: Tuple[float, ...] = (0.229, 0.224, 0.225)

    CLASS_NAMES: Tuple[str, ...] = (
        "Background",
        "Building",
        "Road",
        "Water",
        "Barren",
        "Forest",
        "Agricultural",
    )
    ICPRS_CLASS_NAMES: Tuple[str, ...] = (
        "Impervious",
        "Building",
        "LowVeg",
        "Tree",
        "Car",
        "Clutter",
    )

    ICPRS_VAL_SPLIT: float = 0.2
    ICPRS_SEED: int = 42

    GPU_ID: int = 0
    LOG_INTERVAL: int = 250

    def __post_init__(self) -> None:
        variant = (self.SWIN_VARIANT or "tiny").lower()
        if variant not in self.SWIN_EMBED_DIM_MAP:
            raise ValueError(f"Unsupported SWIN_VARIANT '{self.SWIN_VARIANT}'")
        self.SWIN_EMBED_DIM = self.SWIN_EMBED_DIM_MAP[variant]
        self.SWIN_DEPTHS = self.SWIN_DEPTHS_MAP[variant]
        self.SWIN_NUM_HEADS = self.SWIN_NUM_HEADS_MAP[variant]
        self.SWIN_DROP_PATH = self.SWIN_DROP_PATH_MAP[variant]
        if not self.WEIGHTS_PATH or self.WEIGHTS_PATH.lower() == "auto":
            weight_name = self.SWIN_WEIGHTS_FILE_MAP.get(variant, "")
            self.WEIGHTS_PATH = os.path.join(self.SWIN_WEIGHTS_DIR, weight_name) if weight_name else ""
        os.makedirs(self.OUTPUT_DIR, exist_ok=True)
        os.makedirs(os.path.join(self.OUTPUT_DIR, "checkpoints"), exist_ok=True)
        os.makedirs(os.path.join(self.OUTPUT_DIR, "logs"), exist_ok=True)
        os.makedirs(os.path.join(self.OUTPUT_DIR, "tensorboard"), exist_ok=True)
        os.makedirs(os.path.join(self.OUTPUT_DIR, "val_preds"), exist_ok=True)

    def get_full_path(self, relative_path: str) -> str:
        return os.path.join(self.DATA_ROOT, relative_path)

    def get_train_paths(self):
        return (
            [self.get_full_path(p) for p in self.TRAIN_IMG_DIR],
            [self.get_full_path(p) for p in self.TRAIN_MASK_DIR],
        )

    def get_val_paths(self):
        return (
            [self.get_full_path(p) for p in self.VAL_IMG_DIR],
            [self.get_full_path(p) for p in self.VAL_MASK_DIR],
        )


cfg = Config()
