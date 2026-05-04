#!/usr/bin/env python3
"""
train.py — Unified iteration-based training for all Mamba-Segmentation backbones.

Usage:
    python train.py --config configs/vmamba.yaml
    python train.py --config configs/mambavision.yaml variant=small
    python train.py --config configs/cnn_deeplabv3p.yaml output_dir=my_run

Features:
    - Single entry point for all backbone families
    - YAML config with optional CLI key=value overrides
    - AdamW with differential backbone/head learning rates
    - Polynomial decay scheduler
    - Mixed precision (AMP)
    - TensorBoard logging
    - Best checkpoint tracking
"""

import argparse
import logging
import os
import sys
from datetime import datetime
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# ── Root of this file is Mamba-Segmentation/ ──────────────────────────────
_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from core.config_loader import load_config
from core.model import SegmentationModel
from core.dataset import build_dataloaders
from core.dataset_isprs import build_potsdam_loaders
from core.losses import TriBraidLoss
from core.utils import (
    SegmentationEvaluator,
    AverageMeter,
    PolynomialDecay,
    create_optimizer_with_differential_lr,
    format_metrics_table,
    save_checkpoint,
    load_checkpoint,
)


# ── Weights auto-resolution ────────────────────────────────────────────────

_WEIGHTS_MAP = {
    # backbone -> {variant -> relative_path_from_ROOT}
    "mambavision": {
        "tiny":   "MambaVision/weights/1k/mambavision_tiny_1k.pth.tar",
        "tiny2":  "MambaVision/weights/1k/mambavision_tiny2_1k.pth.tar",
        "small":  "MambaVision/weights/1k/mambavision_small_1k.pth.tar",
        "base":   "MambaVision/weights/1k/mambavision_base_1k.pth.tar",
        "large":  "MambaVision/weights/1k/mambavision_large_1k.pth.tar",
        "large2": "MambaVision/weights/1k/mambavision_large2_1k.pth.tar",
    },
    "vmamba": {
        "tiny":  "VMamba/weights/ImageNet-1K/vssmtiny_dp01_ckpt_epoch_292.pth",
        "small": "VMamba/weights/ImageNet-1K/vssmsmall_dp03_ckpt_epoch_238.pth",
        "base":  "VMamba/weights/ImageNet-1K/vssmbase_dp06_ckpt_epoch_241.pth",
    },
    "visionmamba": {
        "tiny":  "VisionMamba/weights/vim_t_midclstok_ft_78p3acc.pth",
        "small": "VisionMamba/weights/vim_s_midclstok_ft_81p6acc.pth",
        "base":  "VisionMamba/weights/vim_b_midclstok_81p9acc.pth",
    },
    "spatialmamba": {
        "tiny":  "spatial-mamba/weights/imageNet1K/spatialmamba_tiny_224_1k.pth",
        "small": "spatial-mamba/weights/imageNet1K/spatialmamba_small_224_1k.pth",
        "base":  "spatial-mamba/weights/imageNet1K/spatialmamba_base_224_1k.pth",
    },
}


def _resolve_weights(cfg: dict) -> dict:
    """Replace 'auto' weights_path with the concrete path, if found."""
    enc_kw = cfg.get("encoder_kwargs", {})
    if enc_kw.get("weights_path", "") != "auto":
        return cfg

    backbone = cfg.get("backbone", "").lower()
    variant = cfg.get("variant", cfg.get("encoder_kwargs", {}).get("model_variant", "base"))
    if isinstance(variant, str):
        variant = variant.lower()

    backbone_map = _WEIGHTS_MAP.get(backbone, {})
    rel = backbone_map.get(variant)
    if rel:
        full = os.path.join(_ROOT, rel)
        if os.path.isfile(full):
            enc_kw["weights_path"] = full
        else:
            print(f"[weights] Expected {full} — file not found. Training without pretrained weights.")
            enc_kw["weights_path"] = None
    else:
        enc_kw["weights_path"] = None
    return cfg


# ── Helpers ────────────────────────────────────────────────────────────────

def _configure_runtime() -> None:
    try:
        torch.multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    try:
        torch.multiprocessing.set_sharing_strategy("file_system")
    except RuntimeError:
        pass


def setup_logging(output_dir: str) -> logging.Logger:
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(output_dir, f"train_{timestamp}.log")

    logger = logging.getLogger("MambaSegmentation")
    logger.setLevel(logging.INFO)

    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


def _format_flops(flops: float) -> str:
    if flops >= 1e12:
        return f"{flops / 1e12:.2f} TFLOPs"
    if flops >= 1e9:
        return f"{flops / 1e9:.2f} GFLOPs"
    return f"{flops / 1e6:.2f} MFLOPs"


def _compute_flops(model: nn.Module, input_size: int) -> float:
    try:
        from thop import profile  # type: ignore
    except Exception:
        return -1.0
    model.eval()
    dummy = torch.randn(1, 3, input_size, input_size, device=next(model.parameters()).device)
    flops, _ = profile(model, inputs=(dummy,), verbose=False)
    return float(flops)


def create_dataloaders(cfg: dict):
    """Build train and val DataLoaders from config dict."""
    dataset = cfg.get("dataset", "loveda").lower()

    if dataset == "icprs":
        train_loader, val_loader, _ = build_potsdam_loaders(
            root=cfg["data_root"],
            patch_size=cfg["crop_size"],
            train_stride=cfg.get("icprs_train_stride", cfg["crop_size"]),
            val_stride=cfg.get("icprs_val_stride", cfg["crop_size"]),
            val_split=cfg.get("icprs_val_split", 0.2),
            test_split=cfg.get("icprs_test_split", 0.0),
            batch_size=cfg["batch_size"],
            num_workers=cfg["num_workers"],
            pin_memory=cfg.get("pin_memory", True),
            persistent_workers=cfg.get("persistent_workers", True),
            normalize_mean=tuple(cfg["rgb_mean"]),
            normalize_std=tuple(cfg["rgb_std"]),
            ignore_index=cfg["ignore_index"],
            train_mode=cfg.get("icprs_train_mode", "random_crop"),
            augment=True,
            cache_tiles=cfg.get("icprs_cache_tiles", False),
            seed=cfg.get("icprs_seed", 0),
        )
        return train_loader, val_loader

    # LoveDA
    class _FakeConfig:
        """Thin adapter so build_dataloaders (which expects Config object) works with a dict."""
        pass

    fake = _FakeConfig()
    fake.DATA_ROOT = cfg["data_root"]
    fake.CROP_SIZE = cfg["crop_size"]
    fake.BATCH_SIZE = cfg["batch_size"]
    fake.NUM_WORKERS = cfg["num_workers"]
    fake.PIN_MEMORY = cfg.get("pin_memory", True)
    fake.PERSISTENT_WORKERS = cfg.get("persistent_workers", True)
    fake.PREFETCH_FACTOR = cfg.get("prefetch_factor", 4)
    fake.TRAIN_IMG_DIR = cfg["train_img_dirs"]
    fake.TRAIN_MASK_DIR = cfg["train_mask_dirs"]
    fake.VAL_IMG_DIR = cfg["val_img_dirs"]
    fake.VAL_MASK_DIR = cfg["val_mask_dirs"]
    fake.RGB_MEAN = cfg["rgb_mean"]
    fake.RGB_STD = cfg["rgb_std"]
    fake.IGNORE_INDEX = cfg["ignore_index"]
    return build_dataloaders(fake)


def infinite_dataloader(dataloader):
    while True:
        for batch in dataloader:
            yield batch


@torch.no_grad()
def validate(model, val_loader, criterion, evaluator, device, logger, use_amp, class_names=None):
    model.eval()
    evaluator.reset()
    loss_meter = AverageMeter()

    val_pbar = tqdm(val_loader, desc="Val", leave=False, ncols=100)
    for batch in val_pbar:
        rgb = batch["rgb"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)

        with autocast(enabled=use_amp):
            outputs = model(rgb)
            if isinstance(outputs, (list, tuple)):
                main_out = outputs[0]
            else:
                main_out = outputs

            if main_out.shape[-2:] != mask.shape[-2:]:
                main_out = F.interpolate(main_out, size=mask.shape[-2:], mode="bilinear", align_corners=False)

        pred = main_out.argmax(dim=1)
        evaluator.update(pred, mask)

        loss, _ = criterion(outputs, mask)
        loss_meter.update(loss.item())
        val_pbar.set_postfix(loss=f"{loss_meter.avg:.4f}")

    metrics = evaluator.get_metrics()
    metrics["val_loss"] = loss_meter.avg
    logger.info("\n" + format_metrics_table(metrics, class_names=class_names))
    model.train()
    return metrics


# ── Main training function ─────────────────────────────────────────────────

def train(cfg: dict, resume_path: Optional[str] = None) -> None:
    device = torch.device(f"cuda:{cfg['gpu_id']}" if torch.cuda.is_available() else "cpu")

    output_dir = os.path.join(_ROOT, cfg["output_dir"]) if not os.path.isabs(cfg["output_dir"]) else cfg["output_dir"]
    checkpoint_dir = os.path.join(output_dir, "checkpoints")
    tensorboard_dir = os.path.join(output_dir, "tensorboard")
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(tensorboard_dir, exist_ok=True)

    logger = setup_logging(output_dir)
    logger.info(f"Backbone: {cfg['backbone']}  variant: {cfg.get('variant', '-')}")
    logger.info(f"Device: {device}  Output: {output_dir}")

    writer = SummaryWriter(tensorboard_dir, flush_secs=30)

    logger.info("Building dataloaders…")
    train_loader, val_loader = create_dataloaders(cfg)
    logger.info(f"Train: {len(train_loader.dataset)}  Val: {len(val_loader.dataset)}")

    logger.info("Building model…")
    model = SegmentationModel(cfg).to(device)
    total_p = sum(p.numel() for p in model.parameters())
    train_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Parameters — total: {total_p:,}  trainable: {train_p:,}")
    flops = _compute_flops(model, cfg["crop_size"])
    if flops > 0:
        logger.info(f"FLOPs @ {cfg['crop_size']}px: {_format_flops(flops)}")

    optimizer = create_optimizer_with_differential_lr(
        model,
        lr_backbone=cfg["lr_backbone"],
        lr_head=cfg["lr_head"],
        weight_decay=cfg["weight_decay"],
    )
    scheduler = PolynomialDecay(optimizer, max_iters=cfg["max_iters"], power=cfg["poly_power"], min_lr=1e-6)

    criterion = TriBraidLoss(
        ignore_index=cfg["ignore_index"],
        focal_gamma=cfg["focal_gamma"],
        boundary_weight=cfg["boundary_weight"],
    )
    evaluator = SegmentationEvaluator(num_classes=cfg["num_classes"], device=device)
    scaler = GradScaler(enabled=cfg["use_amp"])

    start_iter = 0
    best_miou = 0.0
    if resume_path:
        logger.info(f"Resuming from: {resume_path}")
        start_iter, m = load_checkpoint(resume_path, model, optimizer, device)
        best_miou = m.get("mIoU", 0.0)
        logger.info(f"Resumed iter {start_iter}, best mIoU {best_miou:.4f}")

    train_iter = infinite_dataloader(train_loader)
    loss_meter = AverageMeter()
    class_names = cfg.get("class_names", None)

    logger.info(f"Training {start_iter} → {cfg['max_iters']} iters, val every {cfg['val_interval']}")
    model.train()

    pbar = tqdm(range(start_iter, cfg["max_iters"]), initial=start_iter, total=cfg["max_iters"], desc="Train", ncols=100)
    for iteration in pbar:
        batch = next(train_iter)
        rgb = batch["rgb"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)

        scheduler.step(iteration)
        optimizer.zero_grad()

        bw = cfg["boundary_weight"] if iteration >= cfg["boundary_warmup_iters"] else 0.0
        criterion.boundary_weight = bw

        with autocast(enabled=cfg["use_amp"]):
            outputs = model(rgb)
            loss, loss_dict = criterion(outputs, mask)

        if not torch.isfinite(loss):
            logger.warning(f"Non-finite loss at iter {iteration + 1}, skipping")
            optimizer.zero_grad(set_to_none=True)
            continue

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg["grad_clip"])
        scaler.step(optimizer)
        scaler.update()
        loss_meter.update(loss.item())

        if (iteration + 1) % cfg["log_interval"] == 0:
            lr = scheduler.get_lr()
            logger.info(
                f"Iter [{iteration + 1}/{cfg['max_iters']}] "
                f"Loss: {loss_meter.avg:.4f}  LR_bb: {lr[0]:.2e}  LR_hd: {lr[1]:.2e}"
            )
            pbar.set_postfix(loss=f"{loss_meter.avg:.4f}", lr=f"{lr[0]:.2e}")
            writer.add_scalar("train/loss", loss_meter.avg, iteration + 1)
            for k, v in loss_dict.items():
                writer.add_scalar(f"train/loss_{k}", v, iteration + 1)
            writer.add_scalar("train/lr_backbone", lr[0], iteration + 1)
            writer.add_scalar("train/lr_head", lr[1], iteration + 1)
            loss_meter.reset()

        if (iteration + 1) % cfg["val_interval"] == 0:
            logger.info(f"\n{'='*60}\nValidation at iter {iteration + 1}\n{'='*60}")
            metrics = validate(model, val_loader, criterion, evaluator, device, logger,
                               use_amp=cfg["use_amp"], class_names=class_names)

            writer.add_scalar("val/mIoU", metrics["mIoU"], iteration + 1)
            writer.add_scalar("val/mF1", metrics["mF1"], iteration + 1)
            writer.add_scalar("val/OA", metrics["OA"], iteration + 1)
            writer.add_scalar("val/loss", metrics["val_loss"], iteration + 1)
            writer.flush()

            ckpt_path = os.path.join(checkpoint_dir, f"checkpoint_iter_{iteration + 1}.pth")
            save_checkpoint(model, optimizer, iteration + 1, metrics, ckpt_path)
            logger.info(f"Saved: {ckpt_path}")

            if metrics["mIoU"] > best_miou:
                best_miou = metrics["mIoU"]
                best_path = os.path.join(checkpoint_dir, "best.pth")
                save_checkpoint(model, optimizer, iteration + 1, metrics, best_path)
                logger.info(f"New best mIoU: {best_miou:.4f}  → {best_path}")

            model.train()

    final_path = os.path.join(checkpoint_dir, "final.pth")
    save_checkpoint(model, optimizer, cfg["max_iters"], metrics, final_path)
    logger.info(f"Done. Best mIoU: {best_miou:.4f}  Final saved: {final_path}")
    writer.close()


# ── Entry point ────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(description="Train a segmentation model")
    p.add_argument("--config", required=True, help="Path to YAML config (e.g. configs/vmamba.yaml)")
    p.add_argument("overrides", nargs="*", metavar="key=value",
                   help="Optional config overrides, e.g. variant=small batch_size=4")
    return p.parse_args()


def main():
    _configure_runtime()
    args = _parse_args()

    cfg = load_config(args.config, overrides=args.overrides)
    cfg = _resolve_weights(cfg)

    if not os.path.exists(cfg["data_root"]):
        raise ValueError(
            f"Data root not found: {cfg['data_root']}\n"
            f"Set the LOVEDA_ROOT (or POTSDAM_ROOT) environment variable, "
            f"or pass data_root=/path/to/dataset as a CLI override."
        )

    resume = cfg.get("resume_path") or None
    train(cfg, resume_path=resume)


if __name__ == "__main__":
    main()
