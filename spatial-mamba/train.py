"""
train.py — Iteration-based training loop for UrbanMamba.

Features:
- AdamW with differential learning rates (backbone vs head)
- Polynomial decay scheduler
- Mixed precision training (AMP)
- Validation every VAL_INTERVAL iterations
- Checkpoint saving with best model tracking
- TensorBoard logging
"""

import os
import sys
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
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
if "config" in sys.modules:
    del sys.modules["config"]

for _mod in ("config", "model", "dataset", "losses", "utils"):
    if _mod in sys.modules:
        del sys.modules[_mod]

import importlib.util

_local_config_path = os.path.join(ROOT_DIR, "config.py")
_spec_config = importlib.util.spec_from_file_location("local_config", _local_config_path)
_local_config = importlib.util.module_from_spec(_spec_config)
assert _spec_config and _spec_config.loader, "Failed to load local config.py"
_spec_config.loader.exec_module(_local_config)
Config = _local_config.Config
from model import NSSTMamba
from dataset import build_dataloaders
from losses import TriBraidLoss
import importlib.util

_local_utils_path = os.path.join(ROOT_DIR, "utils.py")
_spec_utils = importlib.util.spec_from_file_location("local_utils", _local_utils_path)
_local_utils = importlib.util.module_from_spec(_spec_utils)
assert _spec_utils and _spec_utils.loader, "Failed to load local utils.py"
_spec_utils.loader.exec_module(_local_utils)

SegmentationEvaluator = _local_utils.SegmentationEvaluator
AverageMeter = _local_utils.AverageMeter
PolynomialDecay = _local_utils.PolynomialDecay
create_optimizer_with_differential_lr = _local_utils.create_optimizer_with_differential_lr
format_metrics_table = _local_utils.format_metrics_table
save_checkpoint = _local_utils.save_checkpoint
load_checkpoint = _local_utils.load_checkpoint
import importlib.util

_utils_path = os.path.join(ROOT_DIR, "utils.py")
_spec = importlib.util.spec_from_file_location("local_utils", _utils_path)
_local_utils = importlib.util.module_from_spec(_spec)
assert _spec and _spec.loader, "Failed to load local utils.py"
_spec.loader.exec_module(_local_utils)

SegmentationEvaluator = _local_utils.SegmentationEvaluator
AverageMeter = _local_utils.AverageMeter
PolynomialDecay = _local_utils.PolynomialDecay
create_optimizer_with_differential_lr = _local_utils.create_optimizer_with_differential_lr
format_metrics_table = _local_utils.format_metrics_table
save_checkpoint = _local_utils.save_checkpoint
load_checkpoint = _local_utils.load_checkpoint


def _format_flops(flops: float) -> str:
    """Format FLOPs with readable units."""
    if flops >= 1e12:
        return f"{flops / 1e12:.2f} TFLOPs"
    if flops >= 1e9:
        return f"{flops / 1e9:.2f} GFLOPs"
    if flops >= 1e6:
        return f"{flops / 1e6:.2f} MFLOPs"
    return f"{flops:.0f} FLOPs"


def _compute_flops(model: nn.Module, input_size: int) -> float:
    """Compute FLOPs using thop if available."""
    try:
        from thop import profile  # type: ignore
    except Exception:
        return -1.0

    model.eval()
    dummy = torch.randn(1, 3, input_size, input_size, device=next(model.parameters()).device)
    flops, _ = profile(model, inputs=(dummy,), verbose=False)
    return float(flops)


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
    logger = logging.getLogger("UrbanMamba")
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
    use_amp: bool
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
    logger.info("\n" + format_metrics_table(metrics))
    
    model.train()
    return metrics


def train(config: Config, resume_path: str = None):
    """Main training function."""
    # Setup
    device = torch.device(f'cuda:{config.GPU_ID}' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.benchmark = config.CUDNN_BENCHMARK
    torch.backends.cuda.matmul.allow_tf32 = config.ALLOW_TF32
    torch.backends.cudnn.allow_tf32 = config.ALLOW_TF32
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision(config.MATMUL_PRECISION)
    
    # Create output directories
    output_dir = config.OUTPUT_DIR
    checkpoint_dir = os.path.join(output_dir, 'checkpoints')
    tensorboard_dir = os.path.join(output_dir, 'tensorboard')
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(tensorboard_dir, exist_ok=True)
    
    # Setup logging
    logger = setup_logging(output_dir)
    logger.info(f"Training UrbanMamba on device: {device}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Batch size: {config.BATCH_SIZE}")
    logger.info(f"Crop size: {config.CROP_SIZE}")
    logger.info(f"CUDNN benchmark: {config.CUDNN_BENCHMARK}")
    logger.info(f"TF32 enabled: {config.ALLOW_TF32}")
    logger.info(f"Matmul precision: {config.MATMUL_PRECISION}")
    
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
    logger.info(f"Spatial-Mamba variant: {config.SPATIALMAMBA_VARIANT}")
    logger.info(f"Spatial-Mamba weights: {config.WEIGHTS_PATH}")
    logger.info(f"Spatial-Mamba depths: {config.SPATIALMAMBA_DEPTHS}")
    logger.info(f"Spatial-Mamba dims: {config.SPATIALMAMBA_DIMS}")
    logger.info(f"Spatial-Mamba drop path: {config.SPATIALMAMBA_DROP_PATH}")
    model = NSSTMamba(
        num_classes=config.NUM_CLASSES,
        encoder_dims=config.SPATIALMAMBA_DIMS,
        encoder_depths=config.SPATIALMAMBA_DEPTHS,
        drop_path_rate=config.SPATIALMAMBA_DROP_PATH,
        weights_path=config.WEIGHTS_PATH,
        encoder_variant=config.SPATIALMAMBA_VARIANT,
        d_state=config.SPATIALMAMBA_D_STATE,
        dt_init=config.SPATIALMAMBA_DT_INIT,
        mlp_ratio=config.SPATIALMAMBA_MLP_RATIO,
    )
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
        power=0.9,
        min_lr=1e-6
    )
    
    # Create loss function
    criterion = TriBraidLoss(
        ignore_index=255,
        focal_gamma=2.0,
        boundary_weight=0.5
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
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
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
                use_amp=config.USE_AMP
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


def main():
    _configure_runtime()
    # Load config
    config = Config()
    
    # Validate paths
    if not os.path.exists(config.DATA_ROOT):
        raise ValueError(f"Data root not found: {config.DATA_ROOT}")
    
    if config.WEIGHTS_PATH and not os.path.exists(config.WEIGHTS_PATH):
        print(f"Warning: Pretrained weights not found: {config.WEIGHTS_PATH}")
        print("Training will start from scratch.")
    
    # Start training
    resume_path = config.RESUME_PATH if config.RESUME_PATH else None
    train(config, resume_path=resume_path)


if __name__ == '__main__':
    main()
