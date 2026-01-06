"""
utils.py — Evaluation utilities with streaming confusion matrix.

Provides SegmentationEvaluator for calculating:
- mIoU, mF1, Overall Accuracy
- Per-class IoU and F1 scores
"""

import torch
from typing import Dict, Tuple
from config import Config


class SegmentationEvaluator:
    """
    Streaming confusion matrix evaluator for semantic segmentation.
    Operates entirely on GPU to avoid CPU transfer bottlenecks.
    """
    
    def __init__(self, num_classes: int, device: torch.device):
        """
        Initialize evaluator with GPU-based confusion matrix.
        
        Args:
            num_classes: Number of valid classes (excluding ignore)
            device: Torch device for GPU operations
        """
        self.num_classes = num_classes
        self.device = device
        # Confusion matrix: rows = ground truth, cols = predictions
        self.confusion_matrix = torch.zeros(
            (num_classes, num_classes), 
            dtype=torch.int64, 
            device=device
        )
    
    def reset(self) -> None:
        """Reset confusion matrix to zeros."""
        self.confusion_matrix.zero_()
    
    @torch.no_grad()
    def update(self, pred: torch.Tensor, target: torch.Tensor) -> None:
        """
        Update confusion matrix with batch predictions.
        
        Args:
            pred: Predicted class indices [N, H, W], dtype=long
            target: Ground truth class indices [N, H, W], dtype=long
                    Values == 255 are treated as ignore_index
        """
        # Flatten spatial dimensions
        pred = pred.view(-1)
        target = target.view(-1)
        
        # Create mask for valid pixels (not ignore_index=255)
        valid_mask = target != 255
        
        # Filter to valid pixels only
        pred = pred[valid_mask]
        target = target[valid_mask]
        
        # Compute linear indices for scatter_add
        # index = target * num_classes + pred (row-major for confusion matrix)
        indices = target * self.num_classes + pred
        
        # Update confusion matrix using scatter_add for efficiency
        flat_cm = self.confusion_matrix.view(-1)
        ones = torch.ones_like(indices, dtype=torch.int64, device=self.device)
        flat_cm.scatter_add_(0, indices, ones)
    
    def get_metrics(self) -> Dict[str, float]:
        """
        Compute all metrics from accumulated confusion matrix.
        
        Returns:
            Dictionary containing:
            - 'mIoU': Mean Intersection over Union
            - 'mF1': Mean F1 score
            - 'OA': Overall Accuracy
            - 'per_class_iou': List of per-class IoU values
            - 'per_class_f1': List of per-class F1 values
        """
        # Convert to float for division operations
        cm = self.confusion_matrix.float()
        
        # Per-class true positives (diagonal elements)
        tp = cm.diag()
        
        # Per-class false positives (column sums - diagonal)
        fp = cm.sum(dim=0) - tp
        
        # Per-class false negatives (row sums - diagonal)
        fn = cm.sum(dim=1) - tp
        
        # IoU = TP / (TP + FP + FN)
        iou_denom = tp + fp + fn
        iou = torch.where(
            iou_denom > 0,
            tp / iou_denom,
            torch.zeros_like(tp)
        )
        
        # Precision = TP / (TP + FP)
        precision_denom = tp + fp
        precision = torch.where(
            precision_denom > 0,
            tp / precision_denom,
            torch.zeros_like(tp)
        )
        
        # Recall = TP / (TP + FN)
        recall_denom = tp + fn
        recall = torch.where(
            recall_denom > 0,
            tp / recall_denom,
            torch.zeros_like(tp)
        )
        
        # F1 = 2 * Precision * Recall / (Precision + Recall)
        f1_denom = precision + recall
        f1 = torch.where(
            f1_denom > 0,
            2 * precision * recall / f1_denom,
            torch.zeros_like(tp)
        )
        
        # Overall Accuracy = sum(TP) / sum(all pixels)
        total_pixels = cm.sum()
        overall_accuracy = tp.sum() / total_pixels if total_pixels > 0 else 0.0
        
        # Mean metrics (only over classes with samples)
        valid_classes = iou_denom > 0
        mean_iou = iou[valid_classes].mean() if valid_classes.any() else torch.tensor(0.0)
        mean_f1 = f1[valid_classes].mean() if valid_classes.any() else torch.tensor(0.0)
        
        return {
            'mIoU': mean_iou.item(),
            'mF1': mean_f1.item(),
            'OA': overall_accuracy.item() if isinstance(overall_accuracy, torch.Tensor) else overall_accuracy,
            'per_class_iou': iou.cpu().tolist(),
            'per_class_f1': f1.cpu().tolist()
        }


def format_metrics_table(
    metrics: Dict[str, float],
    class_names: list = None
) -> str:
    """
    Format metrics as a pretty table for logging.
    
    Args:
        metrics: Dictionary from SegmentationEvaluator.get_metrics()
        class_names: Optional list of class names for display
        
    Returns:
        Formatted string table
    """
    if class_names is None:
        class_names = Config.CLASS_NAMES
    
    per_class_iou = metrics['per_class_iou']
    per_class_f1 = metrics['per_class_f1']
    
    # Build table
    lines = []
    lines.append("=" * 55)
    lines.append(f"{'Class':<20} {'IoU':>10} {'F1':>10}")
    lines.append("-" * 55)
    
    for i, name in enumerate(class_names):
        if i < len(per_class_iou):
            iou_val = per_class_iou[i] * 100
            f1_val = per_class_f1[i] * 100
            lines.append(f"{name:<20} {iou_val:>10.2f} {f1_val:>10.2f}")
    
    lines.append("-" * 55)
    lines.append(f"{'Mean':<20} {metrics['mIoU']*100:>10.2f} {metrics['mF1']*100:>10.2f}")
    lines.append(f"{'Overall Accuracy':<20} {metrics['OA']*100:>10.2f}")
    lines.append("=" * 55)
    
    return "\n".join(lines)


class AverageMeter:
    """Computes and stores the average and current value."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count > 0 else 0


class PolynomialDecay:
    """
    Polynomial decay learning rate scheduler.
    
    LR(iter) = base_lr * (1 - iter / max_iters)^power
    """
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        max_iters: int,
        power: float = 0.9,
        min_lr: float = 1e-6
    ):
        """
        Initialize polynomial decay scheduler.
        
        Args:
            optimizer: PyTorch optimizer
            max_iters: Maximum number of iterations
            power: Polynomial power (default 0.9)
            min_lr: Minimum learning rate floor
        """
        self.optimizer = optimizer
        self.max_iters = max_iters
        self.power = power
        self.min_lr = min_lr
        
        # Store initial learning rates for each param group
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
    
    def step(self, current_iter: int) -> None:
        """
        Update learning rate based on current iteration.
        
        Args:
            current_iter: Current training iteration (0-indexed)
        """
        # Compute decay factor
        factor = (1 - current_iter / self.max_iters) ** self.power
        factor = max(factor, 0.0)  # Ensure non-negative
        
        # Update each param group
        for i, group in enumerate(self.optimizer.param_groups):
            new_lr = max(self.base_lrs[i] * factor, self.min_lr)
            group['lr'] = new_lr
    
    def get_lr(self) -> list:
        """Get current learning rates for all param groups."""
        return [group['lr'] for group in self.optimizer.param_groups]


def create_optimizer_with_differential_lr(
    model: torch.nn.Module,
    lr_backbone: float,
    lr_head: float,
    weight_decay: float = 0.01
) -> torch.optim.Optimizer:
    """
    Create AdamW optimizer with differential learning rates.
    
    Args:
        model: VisionMambaSegmentation model
        lr_backbone: Learning rate for encoder backbones
        lr_head: Learning rate for decoder/fusion heads
        weight_decay: Weight decay factor
        
    Returns:
        Configured AdamW optimizer with parameter groups
    """
    # Separate parameters into backbone and head groups
    backbone_params = []
    head_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        
        # Classify parameters based on module name
        if 'rgb_encoder' in name:
            backbone_params.append(param)
        else:
            # Decoder and heads are treated as head
            head_params.append(param)
    
    # Create parameter groups with different learning rates
    param_groups = [
        {'params': backbone_params, 'lr': lr_backbone, 'name': 'backbone'},
        {'params': head_params, 'lr': lr_head, 'name': 'head'}
    ]
    
    optimizer = torch.optim.AdamW(
        param_groups,
        weight_decay=weight_decay,
        betas=(0.9, 0.999)
    )
    
    return optimizer


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    metrics: Dict[str, float],
    filepath: str
) -> None:
    """
    Save training checkpoint.
    
    Args:
        model: Model to save
        optimizer: Optimizer state
        iteration: Current iteration
        metrics: Validation metrics
        filepath: Output file path
    """
    checkpoint = {
        'iteration': iteration,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics
    }
    torch.save(checkpoint, filepath)


def load_checkpoint(
    filepath: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer = None,
    device: torch.device = None,
    strict: bool = None
) -> Tuple[int, Dict[str, float]]:
    """
    Load training checkpoint.
    
    Args:
        filepath: Checkpoint file path
        model: Model to load weights into
        optimizer: Optional optimizer to restore state
        device: Device to load checkpoint to
        
    Returns:
        Tuple of (iteration, metrics)
    """
    checkpoint = torch.load(filepath, map_location=device)
    
    if strict is None:
        strict = True
    model.load_state_dict(checkpoint['model_state_dict'], strict=strict)
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    iteration = checkpoint.get('iteration', 0)
    metrics = checkpoint.get('metrics', {})
    
    return iteration, metrics
