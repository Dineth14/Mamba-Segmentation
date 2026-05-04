#!/usr/bin/env python3
"""
eval_domain.py — Cross-domain evaluation for LoveDA (Urban/Rural splits).

Usage:
    python eval_domain.py --config configs/vmamba.yaml \\
        --ckpt Comparison_Experiments/Vmamb_small_512/checkpoints/best.pth \\
        --domain rural --split val

    # Append results to a CSV:
    python eval_domain.py --config configs/mambavision.yaml \\
        --ckpt path/to/best.pth --domain urban \\
        --append_csv results/domain_eval.csv
"""

import argparse
import csv
import json
import os
import sys
import time
from typing import List, Tuple

import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from core.config_loader import load_config
from core.model import SegmentationModel
from core.dataset import LovedaDataset
from core.utils import SegmentationEvaluator, format_metrics_table, load_checkpoint
from train import _resolve_weights


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Cross-domain evaluation for LoveDA")
    p.add_argument("--config", required=True, help="Path to YAML config")
    p.add_argument("--ckpt", required=True, help="Path to checkpoint (.pth)")
    p.add_argument("--domain", choices=["urban", "rural", "all"], default="urban")
    p.add_argument("--split", choices=["val", "test"], default="val")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--amp", type=int, choices=[0, 1], default=None)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--out_json", default="")
    p.add_argument("--append_csv", default="")
    p.add_argument("--profile", type=int, choices=[0, 1], default=1)
    p.add_argument("overrides", nargs="*", metavar="key=value",
                   help="Optional config overrides")
    return p.parse_args()


def _resolve_val_dirs(domain: str, split: str) -> Tuple[List[str], List[str]]:
    base = "Val/Val" if split == "val" else "Test/Test"
    mapping = {
        "urban": ([f"{base}/Urban/images_png"], [f"{base}/Urban/masks_png"]),
        "rural": ([f"{base}/Rural/images_png"], [f"{base}/Rural/masks_png"]),
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


def _append_csv(path: str, row: dict) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    file_exists = os.path.exists(path)
    fieldnames = ["backbone", "variant", "split", "domain", "mIoU", "OA",
                  "ParamsM", "FLOPsG", "FPS", "MemGB", "ckpt_path"]
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists or os.path.getsize(path) == 0:
            writer.writeheader()
        writer.writerow(row)


def main() -> None:
    args = _parse_args()

    cfg = load_config(args.config, overrides=args.overrides)
    cfg = _resolve_weights(cfg)

    cfg["gpu_id"] = args.gpu
    img_dirs, mask_dirs = _resolve_val_dirs(args.domain, args.split)
    cfg["val_img_dirs"] = img_dirs
    cfg["val_mask_dirs"] = mask_dirs

    use_amp = cfg.get("use_amp", True) if args.amp is None else bool(args.amp)
    device = torch.device(f"cuda:{cfg['gpu_id']}" if torch.cuda.is_available() else "cpu")

    model = SegmentationModel(cfg).to(device)
    try:
        load_checkpoint(args.ckpt, model, device=device, strict=True)
    except RuntimeError as exc:
        print(f"Strict load failed ({exc}), retrying with strict=False")
        load_checkpoint(args.ckpt, model, device=device, strict=False)

    data_root = cfg["data_root"]
    full_img_dirs = [os.path.join(data_root, p) for p in img_dirs]
    full_mask_dirs = [os.path.join(data_root, p) for p in mask_dirs]

    dataset = LovedaDataset(
        img_dirs=full_img_dirs,
        mask_dirs=full_mask_dirs,
        crop_size=cfg["crop_size"],
        is_train=False,
        rgb_mean=cfg["rgb_mean"],
        rgb_std=cfg["rgb_std"],
        ignore_index=cfg["ignore_index"],
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )

    evaluator = SegmentationEvaluator(cfg["num_classes"], device)
    evaluator.reset()
    model.eval()

    with torch.no_grad():
        for batch in loader:
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

    metrics = evaluator.get_metrics()
    print(format_metrics_table(metrics))

    params_m = sum(p.numel() for p in model.parameters()) / 1e6
    flops = _compute_flops(model, 512)
    flops_g = flops / 1e9 if flops >= 0 else -1.0

    fps, mem_gb = -1.0, -1.0
    if args.profile:
        fps, mem_gb = _profile_model(model, device, use_amp)
        print(f"FPS={fps:.2f}  MemGB={mem_gb:.3f}")

    if args.out_json:
        payload = {
            "backbone": cfg["backbone"],
            "variant": cfg.get("variant", "-"),
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
        _append_csv(args.append_csv, {
            "backbone": cfg["backbone"],
            "variant": cfg.get("variant", "-"),
            "split": args.split,
            "domain": args.domain,
            "mIoU": metrics.get("mIoU", 0.0),
            "OA": metrics.get("OA", 0.0),
            "ParamsM": f"{params_m:.2f}",
            "FLOPsG": f"{flops_g:.2f}",
            "FPS": f"{fps:.2f}",
            "MemGB": f"{mem_gb:.3f}",
            "ckpt_path": args.ckpt,
        })


if __name__ == "__main__":
    main()
