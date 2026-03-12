"""
train.py — Iteration-based training loop for TransformerUNetFormer.

Features:
- AdamW with differential learning rates (backbone vs head)
- Polynomial decay scheduler
- Mixed precision training (AMP)
- Validation every VAL_INTERVAL iterations
- Checkpoint saving with best model tracking
- TensorBoard logging
"""

from __future__ import annotations

import argparse
import os
import logging
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# Local imports
from model import build_model
from dataset import build_dataloaders
from dataset_isprs import build_potsdam_loaders
from losses import TriBraidLoss
from utils import (
    SegmentationEvaluator,
    AverageMeter,
    PolynomialDecay,
    create_optimizer_with_differential_lr,
    format_metrics_table,
    save_checkpoint,
    load_checkpoint
)


def _format_flops(flops: float) -> str:
    """Format FLOPs with readable units."""
    if flops >= 1e12:
        return f"{flops / 1e12:.2f} TFLOPs"
    if flops >= 1e9:
        return f"{flops / 1e9:.2f} GFLOPs"
    if flops >= 1e6:
        return f"{flops / 1e6:.2f} MFLOPs"
    return f"{flops:.0f} FLOPs"


def _clear_thop_hooks(model: nn.Module) -> None:
    """Remove any THOP hooks/attributes to avoid interfering with training."""
    for module in model.modules():
        for hook_attr in ("_forward_hooks", "_forward_pre_hooks"):
            hooks = getattr(module, hook_attr, None)
            if not hooks:
                continue
            remove_keys = [
                key
                for key, hook in list(hooks.items())
                if getattr(hook, "__module__", "").startswith("thop")
            ]
            for key in remove_keys:
                hooks.pop(key, None)
        if hasattr(module, "total_ops"):
            try:
                delattr(module, "total_ops")
            except Exception:
                pass
        if hasattr(module, "total_params"):
            try:
                delattr(module, "total_params")
            except Exception:
                pass


def _compute_flops(model: nn.Module, input_size: int) -> float:
    """Compute FLOPs using thop if available."""
    try:
        from thop import profile  # type: ignore
    except Exception:
        return -1.0

    model.eval()
    dummy = torch.randn(1, 3, input_size, input_size, device=next(model.parameters()).device)
    try:
        flops, _ = profile(model, inputs=(dummy,), verbose=False)
        return float(flops)
    except Exception:
        return -1.0
    finally:
        _clear_thop_hooks(model)


def _configure_runtime() -> None:
    """Best-effort multiprocessing defaults to avoid DataLoader worker crashes."""
    try:
        torch.multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    try:
        torch.multiprocessing.set_sharing_strategy("file_system")
    except RuntimeError:
        pass


def setup_logging(output_dir: str) -> logging.Logger:
    """Setup logging to file and console."""
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(output_dir, f"train_{timestamp}.log")
    
    # Create logger
    logger = logging.getLogger("TransformerUNetFormer")
    logger.setLevel(logging.INFO)
    
    # File handler
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger


def create_dataloaders(config: Config) -> tuple:
    """Create training and validation dataloaders using build_dataloaders."""
    dataset_name = getattr(config, "DATASET", "loveda").lower()
    if dataset_name == "icprs":
        train_loader, val_loader, _ = build_potsdam_loaders(
            root=config.DATA_ROOT,
            patch_size=config.CROP_SIZE,
            train_stride=getattr(config, "ICPRS_TRAIN_STRIDE", config.CROP_SIZE),
            val_stride=getattr(config, "ICPRS_VAL_STRIDE", config.CROP_SIZE),
            val_split=getattr(config, "ICPRS_VAL_SPLIT", 0.2),
            test_split=getattr(config, "ICPRS_TEST_SPLIT", 0.0),
            batch_size=config.BATCH_SIZE,
            num_workers=config.NUM_WORKERS,
            pin_memory=config.PIN_MEMORY,
            persistent_workers=config.PERSISTENT_WORKERS,
            normalize_mean=tuple(config.RGB_MEAN),
            normalize_std=tuple(config.RGB_STD),
            ignore_index=config.IGNORE_INDEX,
            train_mode=getattr(config, "ICPRS_TRAIN_MODE", "random_crop"),
            augment=True,
            cache_tiles=getattr(config, "ICPRS_CACHE_TILES", False),
            seed=getattr(config, "ICPRS_SEED", 0),
        )
        return train_loader, val_loader
    return build_dataloaders(config)


def infinite_dataloader(dataloader):
    """Create infinite iterator from dataloader."""
    while True:
        for batch in dataloader:
            yield batch


@torch.no_grad()
def validate(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    evaluator: SegmentationEvaluator,
    device: torch.device,
    logger: logging.Logger,
    use_amp: bool,
    class_names: list = None
) -> dict:
    """
    Run validation and compute metrics.
    
    Args:
        model: Model in eval mode
        val_loader: Validation dataloader
        criterion: Loss function
        evaluator: Segmentation evaluator
        device: Torch device
        logger: Logger instance
        
    Returns:
        Dictionary of metrics
    """
    model.eval()
    evaluator.reset()
    
    loss_meter = AverageMeter()

    val_pbar = tqdm(
        val_loader,
        desc="Val",
        leave=False,
        ncols=100
    )
    for batch_idx, batch in enumerate(val_pbar):
        rgb = batch['rgb'].to(device, non_blocking=True)
        mask = batch['mask'].to(device, non_blocking=True)
        
        # Forward pass (no amp during validation for stability)
        with autocast(enabled=use_amp):
            outputs = model(rgb)
            if isinstance(outputs, dict):
                main_out = outputs.get('main') or outputs.get('out') or outputs.get('logits')
            elif isinstance(outputs, (list, tuple)):
                main_out = outputs[0]
            else:
                main_out = outputs
            
            # Resize main output to match mask if needed
            if main_out.shape[-2:] != mask.shape[-2:]:
                main_out_resized = F.interpolate(
                    main_out,
                    size=mask.shape[-2:],
                    mode='bilinear',
                    align_corners=False
                )
            else:
                main_out_resized = main_out
        
        # Get predictions
        pred = main_out_resized.argmax(dim=1)
        
        # Update evaluator
        evaluator.update(pred, mask)
        
        # Compute loss for logging
        loss, _ = criterion(outputs, mask)
        loss_meter.update(loss.item())
        val_pbar.set_postfix(loss=f"{loss_meter.avg:.4f}")
    
    # Get metrics
    metrics = evaluator.get_metrics()
    metrics['val_loss'] = loss_meter.avg
    
    # Log formatted table
    logger.info("\n" + format_metrics_table(metrics, class_names=class_names))
    
    model.train()
    return metrics


def train(config: Config, resume_path: str = None):
    """Main training function."""
    # Setup
    device = torch.device(f'cuda:{config.GPU_ID}' if torch.cuda.is_available() else 'cpu')
    
    # Create output directories
    output_dir = config.OUTPUT_DIR
    checkpoint_dir = os.path.join(output_dir, 'checkpoints')
    tensorboard_dir = os.path.join(output_dir, 'tensorboard')
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(tensorboard_dir, exist_ok=True)
    
    # Setup logging
    logger = setup_logging(output_dir)
    logger.info(f"Training {config.MODEL_NAME} on device: {device}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Batch size: {config.BATCH_SIZE}")
    logger.info(f"Crop size: {config.CROP_SIZE}")
    
    # TensorBoard writer
    writer = SummaryWriter(tensorboard_dir, flush_secs=30)
    
    # Create dataloaders
    logger.info("Creating dataloaders...")
    train_loader, val_loader = create_dataloaders(config)
    logger.info(f"Train samples: {len(train_loader.dataset)}")
    logger.info(f"Val samples: {len(val_loader.dataset)}")
    
    # Create model
    logger.info("Creating model...")
    logger.info("RGB encoder: ENABLED")
    logger.info(f"Backbone: {config.BACKBONE_NAME}")
    logger.info(f"ResNet depth: {config.RESNET_DEPTH}")
    logger.info(f"Pretrained weights: {config.WEIGHTS_PATH}")
    model = build_model(config)
    model = model.to(device)
    
    # Log model parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    flops = _compute_flops(model, config.CROP_SIZE)
    if flops > 0:
        logger.info(f"FLOPs @ {config.CROP_SIZE}x{config.CROP_SIZE}: {_format_flops(flops)}")
    else:
        logger.info("FLOPs: thop not available (pip install thop)")
    
    # Create optimizer with differential LR
    optimizer = create_optimizer_with_differential_lr(
        model,
        lr_backbone=config.LR_BACKBONE,
        lr_head=config.LR_HEAD,
        weight_decay=config.WEIGHT_DECAY
    )
    
    # Create scheduler
    scheduler = PolynomialDecay(
        optimizer,
        max_iters=config.MAX_ITERS,
        power=config.POLY_POWER,
        min_lr=1e-6
    )
    
    # Create loss function
    criterion = TriBraidLoss(
        ignore_index=config.IGNORE_INDEX,
        focal_gamma=config.FOCAL_GAMMA,
        boundary_weight=config.BOUNDARY_WEIGHT
    )
    
    # Create evaluator
    evaluator = SegmentationEvaluator(
        num_classes=config.NUM_CLASSES,
        device=device
    )
    
    # AMP scaler
    scaler = GradScaler(enabled=config.USE_AMP)
    
    # Resume from checkpoint if specified
    start_iter = 0
    best_miou = 0.0
    
    if resume_path:
        logger.info(f"Resuming from checkpoint: {resume_path}")
        start_iter, metrics = load_checkpoint(
            resume_path,
            model,
            optimizer,
            device
        )
        best_miou = metrics.get('mIoU', 0.0)
        logger.info(f"Resumed from iteration {start_iter}, best mIoU: {best_miou:.4f}")
    
    # Create infinite dataloader iterator
    train_iter = infinite_dataloader(train_loader)
    
    # Training meters
    loss_meter = AverageMeter()
    
    # Training loop
    logger.info(f"Starting training from iteration {start_iter}")
    logger.info(f"Max iterations: {config.MAX_ITERS}")
    logger.info(f"Validation interval: {config.VAL_INTERVAL}")
    
    model.train()
    
    train_pbar = tqdm(
        range(start_iter, config.MAX_ITERS),
        initial=start_iter,
        total=config.MAX_ITERS,
        desc="Train",
        ncols=100
    )
    for iteration in train_pbar:
        # Get next batch
        batch = next(train_iter)
        
        rgb = batch['rgb'].to(device, non_blocking=True)
        mask = batch['mask'].to(device, non_blocking=True)
        
        # Update learning rate
        scheduler.step(iteration)
        
        # Forward pass with AMP
        optimizer.zero_grad()
        
        boundary_weight = config.BOUNDARY_WEIGHT if iteration >= config.BOUNDARY_WARMUP_ITERS else 0.0
        criterion.boundary_weight = boundary_weight
        with autocast(enabled=config.USE_AMP):
            outputs = model(rgb)
            loss, loss_dict = criterion(outputs, mask)

        if not torch.isfinite(loss):
            logger.warning(
                f"[!] Non-finite loss at iter {iteration + 1} "
                f"(loss={loss.item():.4f}), skipping update"
            )
            optimizer.zero_grad(set_to_none=True)
            continue

        # Backward pass with gradient scaling
        scaler.scale(loss).backward()
        
        # Gradient clipping
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.GRAD_CLIP)
        
        # Optimizer step
        scaler.step(optimizer)
        scaler.update()
        
        # Update meters
        loss_meter.update(loss.item())
        
        # Logging
        if (iteration + 1) % config.LOG_INTERVAL == 0:
            current_lr = scheduler.get_lr()
            logger.info(
                f"Iter [{iteration + 1}/{config.MAX_ITERS}] "
                f"Loss: {loss_meter.avg:.4f} "
                f"LR_backbone: {current_lr[0]:.2e} "
                f"LR_head: {current_lr[1]:.2e}"
            )
            train_pbar.set_postfix(
                loss=f"{loss_meter.avg:.4f}",
                lr=f"{current_lr[0]:.2e}"
            )
            
            # TensorBoard logging
            writer.add_scalar('train/loss', loss_meter.avg, iteration + 1)
            writer.add_scalar('train/loss_lovasz', loss_dict.get('lovasz', 0.0), iteration + 1)
            writer.add_scalar('train/loss_focal', loss_dict.get('focal', 0.0), iteration + 1)
            writer.add_scalar('train/loss_boundary', loss_dict.get('boundary', 0.0), iteration + 1)
            writer.add_scalar('train/lr_backbone', current_lr[0], iteration + 1)
            writer.add_scalar('train/lr_head', current_lr[1], iteration + 1)
            
            loss_meter.reset()
        
        # Validation
        if (iteration + 1) % config.VAL_INTERVAL == 0:
            logger.info(f"\n{'='*60}")
            logger.info(f"Validation at iteration {iteration + 1}")
            logger.info(f"{'='*60}")
            
            metrics = validate(
                model, val_loader, criterion,
                evaluator, device, logger,
                use_amp=config.USE_AMP,
                class_names=list(config.CLASS_NAMES)
            )
            
            # TensorBoard logging
            writer.add_scalar('val/mIoU', metrics['mIoU'], iteration + 1)
            writer.add_scalar('val/mF1', metrics['mF1'], iteration + 1)
            writer.add_scalar('val/OA', metrics['OA'], iteration + 1)
            writer.add_scalar('val/loss', metrics['val_loss'], iteration + 1)
            
            for i, class_name in enumerate(config.CLASS_NAMES):
                writer.add_scalar(
                    f'val_iou/{class_name}',
                    metrics['per_class_iou'][i],
                    iteration + 1
                )
            writer.flush()
            
            # Save checkpoint
            checkpoint_path = os.path.join(
                checkpoint_dir,
                f'checkpoint_iter_{iteration + 1}.pth'
            )
            save_checkpoint(
                model, optimizer, iteration + 1, metrics, checkpoint_path
            )
            logger.info(f"Saved checkpoint: {checkpoint_path}")
            
            # Save best model
            if metrics['mIoU'] > best_miou:
                best_miou = metrics['mIoU']
                best_path = os.path.join(checkpoint_dir, 'best.pth')
                save_checkpoint(
                    model, optimizer, iteration + 1, metrics, best_path
                )
                logger.info(f"New best mIoU: {best_miou:.4f}, saved to {best_path}")
            
            model.train()
    
    # Final save
    final_path = os.path.join(checkpoint_dir, 'final.pth')
    save_checkpoint(model, optimizer, config.MAX_ITERS, metrics, final_path)
    logger.info(f"Training complete. Final model saved to {final_path}")
    logger.info(f"Best mIoU achieved: {best_miou:.4f}")
    
    writer.close()


def load_config(name: str):
    if name == "icprs":
        from config_icprs import Config
    else:
        from config import Config
    return Config()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train TransformerUNetFormer segmentation.")
    parser.add_argument(
        "--config",
        default="loveda",
        choices=["loveda", "icprs"],
        help="Dataset config to use.",
    )
    parser.add_argument("--resume", default=None, help="Override resume checkpoint path.")
    return parser.parse_args()


def main() -> None:
    _configure_runtime()
    args = parse_args()
    config = load_config(args.config)
    if args.resume:
        config.RESUME_PATH = args.resume

    if not os.path.exists(config.DATA_ROOT):
        raise ValueError(f"Data root not found: {config.DATA_ROOT}")

    if config.WEIGHTS_PATH and not os.path.exists(config.WEIGHTS_PATH):
        print(f"Warning: Pretrained weights not found: {config.WEIGHTS_PATH}")
        print("Training will start from scratch.")

    resume_path = config.RESUME_PATH if config.RESUME_PATH else None
    train(config, resume_path=resume_path)


if __name__ == "__main__":
    main()
