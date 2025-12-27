"""
UrbanMamba — Production-grade Semantic Segmentation System

This package implements the UrbanMamba RGB-only architecture for remote sensing
image segmentation on the LOVEDA dataset.
"""

from .config import Config
from .model import NSSTMamba
from .dataset import LovedaDataset
from .losses import TriBraidLoss, LovaszSoftmaxV1, FocalLoss, BoundaryLoss
from .utils import (
    SegmentationEvaluator,
    PolynomialDecay,
    AverageMeter,
    create_optimizer_with_differential_lr,
    format_metrics_table,
    save_checkpoint,
    load_checkpoint
)

__all__ = [
    'Config',
    'NSSTMamba',
    'LovedaDataset',
    'TriBraidLoss',
    'LovaszSoftmaxV1',
    'FocalLoss',
    'BoundaryLoss',
    'SegmentationEvaluator',
    'PolynomialDecay',
    'AverageMeter',
    'create_optimizer_with_differential_lr',
    'format_metrics_table',
    'save_checkpoint',
    'load_checkpoint',
]

__version__ = '2.0.0'
