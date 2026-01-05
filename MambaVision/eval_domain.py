#!/usr/bin/env python3
"""
Domain evaluation for LoveDA.

# U->R example (train Urban separately, then evaluate on Rural val):
# cd MambaVision
# python eval_domain.py --variant tiny --ckpt /storage2/ChangeDetection/NSST-mamba/Mamba-Segmentation/Comparison_Experiments/mambavision_tiny_512/checkpoints/best.pth --domain rural --split val --gpu 0 --amp 1 --append_csv ../Comparison_Experiments/mambavision_tiny_512/domain_results.csv
"""

import argparse
import csv
import json
import os
import time
from typing import List, Tuple

import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader

from config import Config
from dataset import LovedaDataset
from model import build_model
from utils import SegmentationEvaluator, format_metrics_table, load_checkpoint


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cross-domain evaluation for LoveDA")
    parser.add_argument("--ckpt", required=True, help="Path to checkpoint (.pth)")
    parser.add_argument("--domain", choices=["urban", "rural", "all"], default="urban")
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--variant", required=True, help="MambaVision variant (tiny/tiny2/small/base/large/large2)")
    parser.add_argument("--data_root", default="/storage2/ChangeDetection/Datasets/Loveda")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--amp", type=int, choices=[0, 1], default=None)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--out_json", default="")
    parser.add_argument("--append_csv", default="")
    parser.add_argument("--profile", type=int, choices=[0, 1], default=1)
    return parser.parse_args()


def _resolve_val_dirs(domain: str, split: str) -> Tuple[List[str], List[str]]:
    if split == "val":
        base = "Val/Val"
    else:
        base = "Test/Test"

    mapping = {
        "urban": (
            [f"{base}/Urban/images_png"],
            [f"{base}/Urban/masks_png"],
        ),
        "rural": (
            [f"{base}/Rural/images_png"],
            [f"{base}/Rural/masks_png"],
        ),
        "all": (
            [f"{base}/Urban/images_png", f"{base}/Rural/images_png"],
            [f"{base}/Urban/masks_png", f"{base}/Rural/masks_png"],
        ),
    }
    return mapping[domain]


def _compute_flops(model: torch.nn.Module, input_size: int) -> float:
    try:
        from thop import profile  # type: ignore
    except Exception:
        return -1.0

    model.eval()
    dummy = torch.randn(1, 3, input_size, input_size, device=next(model.parameters()).device)
    flops, _ = profile(model, inputs=(dummy,), verbose=False)
    return float(flops)


def _profile_model(model: torch.nn.Module, device: torch.device, use_amp: bool) -> Tuple[float, float]:
    if device.type != "cuda":
        return -1.0, -1.0

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    dummy = torch.randn(1, 3, 512, 512, device=device)
    model.eval()

    with torch.no_grad():
        for _ in range(20):
            with autocast(enabled=use_amp):
                _ = model(dummy)
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(100):
            with autocast(enabled=use_amp):
                _ = model(dummy)
        torch.cuda.synchronize()
        elapsed = time.time() - start

    fps = 100.0 / elapsed if elapsed > 0 else 0.0
    mem_gb = torch.cuda.max_memory_reserved() / (1024 ** 3)
    return fps, mem_gb


def _load_weights(ckpt_path: str, model: torch.nn.Module, device: torch.device) -> None:
    try:
        load_checkpoint(ckpt_path, model, device=device, strict=True)
    except RuntimeError as exc:
        print(f"Warning: strict checkpoint load failed: {exc}")
        load_checkpoint(ckpt_path, model, device=device, strict=False)


def _append_csv(path: str, row: dict) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    file_exists = os.path.exists(path)
    fieldnames = [
        "family",
        "variant",
        "split",
        "domain",
        "mIoU",
        "OA",
        "ParamsM",
        "FLOPsG",
        "FPS",
        "MemGB",
        "ckpt_path",
    ]
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists or os.path.getsize(path) == 0:
            writer.writeheader()
        writer.writerow(row)


def main() -> None:
    args = _parse_args()

    cfg = Config(MAMBAVISION_VARIANT=args.variant, DATA_ROOT=args.data_root)
    cfg.GPU_ID = args.gpu
    cfg.VAL_IMG_DIR, cfg.VAL_MASK_DIR = _resolve_val_dirs(args.domain, args.split)

    use_amp = cfg.USE_AMP if args.amp is None else bool(args.amp)

    device = torch.device(f"cuda:{cfg.GPU_ID}" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg).to(device)
    _load_weights(args.ckpt, model, device)

    img_dirs = [cfg.get_full_path(p) for p in cfg.VAL_IMG_DIR]
    mask_dirs = [cfg.get_full_path(p) for p in cfg.VAL_MASK_DIR]

    dataset = LovedaDataset(
        img_dirs=img_dirs,
        mask_dirs=mask_dirs,
        crop_size=cfg.CROP_SIZE,
        is_train=False,
        rgb_mean=cfg.RGB_MEAN,
        rgb_std=cfg.RGB_STD,
        ignore_index=cfg.IGNORE_INDEX,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )

    evaluator = SegmentationEvaluator(cfg.NUM_CLASSES, device)
    evaluator.reset()
    model.eval()

    with torch.no_grad():
        for batch in loader:
            rgb = batch["rgb"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)
            with autocast(enabled=use_amp):
                outputs = model(rgb)
                if isinstance(outputs, dict):
                    main_out = outputs.get("main") or outputs.get("out") or outputs.get("logits")
                elif isinstance(outputs, (list, tuple)):
                    main_out = outputs[0]
                else:
                    main_out = outputs

                if main_out.shape[-2:] != mask.shape[-2:]:
                    main_out = F.interpolate(
                        main_out,
                        size=mask.shape[-2:],
                        mode="bilinear",
                        align_corners=False,
                    )

            pred = main_out.argmax(dim=1)
            evaluator.update(pred, mask)

    metrics = evaluator.get_metrics()
    print(format_metrics_table(metrics))

    params_m = sum(p.numel() for p in model.parameters()) / 1e6
    flops = _compute_flops(model, 512)
    flops_g = flops / 1e9 if flops >= 0 else -1.0

    fps = -1.0
    mem_gb = -1.0
    if args.profile:
        fps, mem_gb = _profile_model(model, device, use_amp)
    print(f"FPS={fps:.2f}, MemGB={mem_gb:.2f}")

    if args.out_json:
        payload = {
            "model_family": "MambaVision",
            "variant": args.variant,
            "split": args.split,
            "domain": args.domain,
            "ckpt_path": args.ckpt,
            "mIoU": metrics.get("mIoU", 0.0),
            "OA": metrics.get("OA", 0.0),
            "per_class_iou": metrics.get("per_class_iou", []),
            "per_class_f1": metrics.get("per_class_f1", []),
            "FPS": fps,
            "MemGB": mem_gb,
        }
        with open(args.out_json, "w") as f:
            json.dump(payload, f, indent=2)

    if args.append_csv:
        _append_csv(
            args.append_csv,
            {
                "family": "MambaVision",
                "variant": args.variant,
                "split": args.split,
                "domain": args.domain,
                "mIoU": metrics.get("mIoU", 0.0),
                "OA": metrics.get("OA", 0.0),
                "ParamsM": params_m,
                "FLOPsG": flops_g,
                "FPS": fps,
                "MemGB": mem_gb,
                "ckpt_path": args.ckpt,
            },
        )


if __name__ == "__main__":
    main()
