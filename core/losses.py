"""
losses.py — TriBraidLoss for semantic segmentation.

Combines three complementary loss functions:
- Lovász-Softmax: Region-based IoU optimization
- Focal Loss: Hard example mining with gamma=2.0
- Boundary Loss: Edge-aware supervision

Formula: L_total = L_Lovasz + L_Focal + 0.5 * L_Boundary
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


# =============================================================================
# Lovász-Softmax Loss (IoU-based, differentiable)
# =============================================================================

def lovasz_grad(gt_sorted: torch.Tensor) -> torch.Tensor:
    """
    Compute the Lovász gradient from sorted ground truth labels.
    
    The Lovász extension provides a way to optimize set functions
    directly, making IoU differentiable.
    """
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1.0 - intersection / union
    
    if p > 1:
        # Compute gradient at each position
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    
    return jaccard


def lovasz_softmax_flat(
    probas: torch.Tensor,
    labels: torch.Tensor,
    classes: str = 'present'
) -> torch.Tensor:
    """
    Multi-class Lovász-Softmax loss for flattened predictions.
    
    Args:
        probas: [P, C] class probabilities at each prediction point
        labels: [P] ground truth labels at each point
        classes: 'present' to average over classes present in batch
        
    Returns:
        Scalar loss value
    """
    if probas.numel() == 0:
        return probas.sum() * 0.0
    
    C = probas.shape[1]
    losses = []
    
    # Determine which classes to compute loss for
    if classes == 'present':
        class_to_sum = torch.unique(labels)
    else:
        class_to_sum = torch.arange(C, device=labels.device)
    
    for c in class_to_sum:
        fg = (labels == c).float()  # Foreground indicator for class c
        if fg.sum() == 0 and classes == 'present':
            continue
        
        # Get probabilities for class c
        class_pred = probas[:, c]
        
        # Compute 1 - probability (errors)
        errors = (fg - class_pred).abs()
        
        # Sort errors in descending order
        errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
        fg_sorted = fg[perm]
        
        # Compute Lovász extension
        losses.append(torch.dot(errors_sorted, lovasz_grad(fg_sorted)))
    
    if len(losses) == 0:
        return probas.sum() * 0.0
    
    return torch.stack(losses).mean()


class LovaszSoftmaxV1(nn.Module):
    """
    Lovász-Softmax loss for multi-class semantic segmentation.
    
    This loss directly optimizes IoU through the Lovász extension,
    making it particularly effective for imbalanced datasets.
    """
    
    def __init__(self, classes: str = 'present', ignore_index: int = 255):
        """
        Args:
            classes: 'present' to average over classes in batch,
                    'all' to average over all classes
            ignore_index: Label index to ignore (default 255)
        """
        super().__init__()
        self.classes = classes
        self.ignore_index = ignore_index
    
    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute Lovász-Softmax loss.
        
        Args:
            logits: [N, C, H, W] raw predictions (before softmax)
            labels: [N, H, W] ground truth labels
            
        Returns:
            Scalar loss value
        """
        # Convert logits to probabilities
        probas = F.softmax(logits, dim=1)
        
        # Flatten spatial dimensions
        N, C, H, W = probas.shape
        probas = probas.permute(0, 2, 3, 1).contiguous().view(-1, C)  # [N*H*W, C]
        labels = labels.view(-1)  # [N*H*W]
        
        # Filter out ignore index
        valid_mask = labels != self.ignore_index
        probas = probas[valid_mask]
        labels = labels[valid_mask]
        
        return lovasz_softmax_flat(probas, labels, self.classes)


# =============================================================================
# Focal Loss (Hard example mining)
# =============================================================================

class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance through hard example mining.
    
    FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)
    
    The focusing parameter gamma reduces the relative loss for
    well-classified examples, putting more focus on hard, misclassified ones.
    """
    
    def __init__(
        self,
        gamma: float = 2.0,
        alpha: Optional[torch.Tensor] = None,
        ignore_index: int = 255,
        reduction: str = 'mean'
    ):
        """
        Args:
            gamma: Focusing parameter (default 2.0)
            alpha: Optional class weights tensor [C]
            ignore_index: Label index to ignore
            reduction: 'mean', 'sum', or 'none'
        """
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.ignore_index = ignore_index
        self.reduction = reduction
    
    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute Focal Loss.
        
        Args:
            logits: [N, C, H, W] raw predictions
            labels: [N, H, W] ground truth labels
            
        Returns:
            Scalar or per-pixel loss based on reduction
        """
        N, C, H, W = logits.shape
        
        # Get log probabilities
        log_p = F.log_softmax(logits, dim=1)  # [N, C, H, W]
        
        # Create valid mask
        valid_mask = labels != self.ignore_index  # [N, H, W]
        
        # Clone labels for safe indexing (replace ignore with 0)
        labels_safe = labels.clone()
        labels_safe[~valid_mask] = 0
        
        # Gather log probabilities for correct class
        # Use advanced indexing: log_p[n, labels[n,h,w], h, w]
        labels_expanded = labels_safe.unsqueeze(1)  # [N, 1, H, W]
        log_pt = log_p.gather(1, labels_expanded).squeeze(1)  # [N, H, W]
        
        # Compute pt for focal weight
        pt = torch.exp(log_pt)
        
        # Focal weight: (1 - pt)^gamma
        focal_weight = (1 - pt) ** self.gamma
        
        # Compute focal loss
        focal_loss = -focal_weight * log_pt  # [N, H, W]
        
        # Apply class weights if provided
        if self.alpha is not None:
            alpha = self.alpha.to(logits.device)
            alpha_t = alpha[labels_safe]  # [N, H, W]
            focal_loss = alpha_t * focal_loss
        
        # Mask out ignore pixels
        focal_loss = focal_loss * valid_mask.float()
        
        if self.reduction == 'mean':
            # Mean over valid pixels only
            num_valid = valid_mask.sum().clamp(min=1)
            return focal_loss.sum() / num_valid
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


# =============================================================================
# Boundary Loss (Edge-aware supervision)
# =============================================================================

class BoundaryLoss(nn.Module):
    """
    Boundary Loss for edge-aware semantic segmentation.
    
    Computes binary cross-entropy between predicted boundaries
    and ground truth boundaries extracted via Laplacian-like operation.
    """
    
    def __init__(
        self,
        theta0: float = 3.0,
        theta: float = 5.0,
        ignore_index: int = 255
    ):
        """
        Args:
            theta0: Boundary threshold for GT
            theta: Dilation factor for boundary region
            ignore_index: Label index to ignore
        """
        super().__init__()
        self.theta0 = theta0
        self.theta = theta
        self.ignore_index = ignore_index
        
        # Laplacian kernel for boundary detection
        laplacian_kernel = torch.tensor([
            [-1, -1, -1],
            [-1,  8, -1],
            [-1, -1, -1]
        ], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer('laplacian_kernel', laplacian_kernel)
    
    def compute_boundaries(self, labels: torch.Tensor) -> torch.Tensor:
        """
        Extract boundaries from label map using Laplacian.
        
        Args:
            labels: [N, H, W] ground truth labels
            
        Returns:
            [N, 1, H, W] boundary probability map
        """
        # Create valid mask (0 for ignore, 1 for valid)
        valid_mask = (labels != self.ignore_index).float()
        
        # Convert labels to float, treating ignore as separate class
        labels_float = labels.float() * valid_mask
        labels_float = labels_float.unsqueeze(1)  # [N, 1, H, W]
        
        # Apply Laplacian filter
        boundary = F.conv2d(
            labels_float,
            self.laplacian_kernel.to(labels.device),
            padding=1
        )
        
        # Boundary exists where Laplacian response is non-zero
        boundary = (boundary.abs() > 0.1).float()
        
        # Mask out boundaries near ignore regions
        valid_mask_4d = valid_mask.unsqueeze(1)
        boundary = boundary * valid_mask_4d
        
        return boundary
    
    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute Boundary Loss.
        
        Args:
            logits: [N, C, H, W] raw predictions
            labels: [N, H, W] ground truth labels
            
        Returns:
            Scalar loss value
        """
        N, C, H, W = logits.shape
        
        # Get predicted probabilities
        probs = F.softmax(logits, dim=1)
        
        # Compute predicted boundaries (max probability gradient)
        # Use Laplacian on max class probability
        max_probs, _ = probs.max(dim=1, keepdim=True)  # [N, 1, H, W]
        
        pred_boundary = F.conv2d(
            max_probs,
            self.laplacian_kernel.to(logits.device),
            padding=1
        )
        pred_boundary_logits = pred_boundary.abs()  # Keep as logits for BCE with logits
        
        # Compute ground truth boundaries
        gt_boundary = self.compute_boundaries(labels)  # [N, 1, H, W]
        
        # Valid mask for loss computation
        valid_mask = (labels != self.ignore_index).unsqueeze(1).float()
        
        # Binary cross-entropy with logits (AMP-safe)
        bce = F.binary_cross_entropy_with_logits(
            pred_boundary_logits * valid_mask,
            gt_boundary * valid_mask,
            reduction='none'
        )
        
        # Mean over valid pixels
        num_valid = valid_mask.sum().clamp(min=1)
        return (bce * valid_mask).sum() / num_valid


# =============================================================================
# TriBraidLoss (Combined loss)
# =============================================================================

class TriBraidLoss(nn.Module):
    """
    TriBraidLoss: Combined loss for semantic segmentation.
    
    Formula: L_total = L_Lovasz + L_Focal + 0.5 * L_Boundary
    """
    
    def __init__(
        self,
        ignore_index: int = 255,
        focal_gamma: float = 2.0,
        boundary_weight: float = 0.5,
        class_weights: Optional[torch.Tensor] = None
    ):
        """
        Args:
            ignore_index: Label index to ignore
            focal_gamma: Gamma for focal loss
            boundary_weight: Weight for boundary loss term
            class_weights: Optional class weights for focal loss
        """
        super().__init__()
        
        self.ignore_index = ignore_index
        self.boundary_weight = boundary_weight
        
        # Initialize component losses
        self.lovasz = LovaszSoftmaxV1(
            classes='present',
            ignore_index=ignore_index
        )
        
        self.focal = FocalLoss(
            gamma=focal_gamma,
            alpha=class_weights,
            ignore_index=ignore_index,
            reduction='mean'
        )
        
        self.boundary = BoundaryLoss(
            ignore_index=ignore_index
        )

    def _unpack_outputs(
        self,
        outputs: Tuple[torch.Tensor, ...]
    ) -> torch.Tensor:
        """Support tuples/lists/dicts with main output."""
        if isinstance(outputs, dict):
            main_out = outputs.get('main') or outputs.get('out') or outputs.get('logits')
            if main_out is None:
                raise ValueError("Outputs dict must contain 'main' logits.")
        elif isinstance(outputs, (list, tuple)):
            if len(outputs) == 0:
                raise ValueError("Outputs tuple must contain at least main logits.")
            main_out = outputs[0]
        elif torch.is_tensor(outputs):
            main_out = outputs
        else:
            raise TypeError(f"Unsupported outputs type: {type(outputs)}")

        return main_out

    def forward(
        self,
        outputs: Tuple[torch.Tensor, ...],
        labels: torch.Tensor
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute TriBraidLoss.
        
        Args:
            outputs: Tuple/list/dict with main logits
            labels: [N, H, W] ground truth labels
            
        Returns:
            Tuple of (total_loss, loss_dict with component values)
        """
        main_out = self._unpack_outputs(outputs)

        loss_lovasz_main = self.lovasz(main_out, labels)
        loss_focal_main = self.focal(main_out, labels)
        loss_boundary_main = self.boundary(main_out, labels)
        loss_main = loss_lovasz_main + loss_focal_main + self.boundary_weight * loss_boundary_main

        total_loss = loss_main

        loss_dict = {
            'total': total_loss.item(),
            'main': loss_main.item(),
            'lovasz': loss_lovasz_main.item(),
            'focal': loss_focal_main.item(),
            'boundary': loss_boundary_main.item()
        }

        return total_loss, loss_dict
