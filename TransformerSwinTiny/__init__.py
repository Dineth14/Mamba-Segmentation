"""
TransformerSwinTiny — Swin-Tiny encoder + LightUNetDecoder segmentation.
"""

from .config import Config
from .model import SwinTinySegmentation, build_model
from .dataset import LovedaDataset
from .losses import TriBraidLoss, LovaszSoftmaxV1, FocalLoss, BoundaryLoss
from .utils import (
    SegmentationEvaluator,
    PolynomialDecay,
    AverageMeter,
    create_optimizer_with_differential_lr,
    format_metrics_table,
    save_checkpoint,
    load_checkpoint,
)

__all__ = [
    "Config",
    "SwinTinySegmentation",
    "build_model",
    "LovedaDataset",
    "TriBraidLoss",
    "LovaszSoftmaxV1",
    "FocalLoss",
    "BoundaryLoss",
    "SegmentationEvaluator",
    "PolynomialDecay",
    "AverageMeter",
    "create_optimizer_with_differential_lr",
    "format_metrics_table",
    "save_checkpoint",
    "load_checkpoint",
]

__version__ = "1.0.0"
