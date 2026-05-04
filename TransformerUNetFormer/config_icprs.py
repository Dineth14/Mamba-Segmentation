"""
TransformerUNetFormer (ResNet18) ICPRS Configuration
===================================================
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple
import os

_HERE = os.path.dirname(os.path.abspath(__file__))


@dataclass
class Config:
    # ============== Paths ==============
    DATA_ROOT: str = os.environ.get("POTSDAM_ROOT", "data/Potsdam")  # set POTSDAM_ROOT env var or edit this path
    DATASET: str = "icprs"
    WEIGHTS_PATH: str = "auto"
    OUTPUT_DIR: str = os.path.join(_HERE, "..", "Comparison_Experiments_ICPRS_potsdam", "transformer_unetformer_r18_512")
    RESUME_PATH: str = ""

    WEIGHTS_DIR: str = os.path.join(_HERE, "..", "weights", "imagenet")
    WEIGHTS_MAP: Dict[str, str] = field(default_factory=lambda: {
        "resnet18": "resnet18-f37072fd.pth",
    })

    # ============== Model ==============
    MODEL_NAME: str = "TransformerUNetFormer"
    BACKBONE_NAME: str = "resnet18"
    RESNET_DEPTH: int = 18
    USE_PRETRAINED: bool = True
    DECODER_CHANNELS: int = 256

    NUM_CLASSES: int = 6
    IGNORE_INDEX: int = 255

    # ============== Data Paths ==============
    TRAIN_IMG_DIR: List[str] = field(default_factory=lambda: ["Images"])
    TRAIN_MASK_DIR: List[str] = field(default_factory=lambda: ["Labels"])
    VAL_IMG_DIR: List[str] = field(default_factory=lambda: ["Images"])
    VAL_MASK_DIR: List[str] = field(default_factory=lambda: ["Labels"])

    # ICPRS split settings
    ICPRS_TRAIN_STRIDE: int = 512
    ICPRS_VAL_STRIDE: int = 512
    ICPRS_VAL_SPLIT: float = 0.2
    ICPRS_TEST_SPLIT: float = 0.0
    ICPRS_TRAIN_MODE: str = "random_crop"
    ICPRS_CACHE_TILES: bool = False
    ICPRS_SEED: int = 42

    # ============== Training ==============
    BATCH_SIZE: int = 4
    CROP_SIZE: int = 512
    MAX_ITERS: int = 25000
    VAL_INTERVAL: int = 2500
    NUM_WORKERS: int = 8
    PIN_MEMORY: bool = True
    PERSISTENT_WORKERS: bool = True

    # ============== Optimization ==============
    LR_BACKBONE: float = 6e-5
    LR_HEAD: float = 3e-4
    WEIGHT_DECAY: float = 0.05
    POLY_POWER: float = 0.9

    USE_AMP: bool = True
    GRAD_CLIP: float = 1.0

    FOCAL_GAMMA: float = 2.0
    BOUNDARY_WEIGHT: float = 0.5
    BOUNDARY_WARMUP_ITERS: int = 10000

    # ============== Normalization ==============
    RGB_MEAN: Tuple[float, ...] = (0.485, 0.456, 0.406)
    RGB_STD: Tuple[float, ...] = (0.229, 0.224, 0.225)

    CLASS_NAMES: Tuple[str, ...] = (
        "Impervious",
        "Building",
        "LowVeg",
        "Tree",
        "Car",
        "Clutter",
    )

    GPU_ID: int = 1
    LOG_INTERVAL: int = 250

    def __post_init__(self) -> None:
        if not self.WEIGHTS_PATH or self.WEIGHTS_PATH.lower() == "auto":
            weight_name = self.WEIGHTS_MAP.get(self.BACKBONE_NAME, "")
            self.WEIGHTS_PATH = os.path.join(self.WEIGHTS_DIR, weight_name) if weight_name else ""
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
