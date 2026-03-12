#!/usr/bin/env python3
"""
Rotation robustness analysis across backbones.
"""

from __future__ import annotations

import argparse
import csv
import gc
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast

try:
    from analysis.analysis_utils import (
        FAMILY_CONFIG,
        activate_family_dir,
        build_config,
        build_val_loader,
        cleanup_family_dir,
        default_experiments_root,
        default_output_dir,
        detect_family_variant,
        disable_pretrained,
        extract_variant_from_hints,
        extract_vmamba_weight_set,
        import_family_modules,
        infer_vmamba_weight_set_from_logs,
        load_model_with_checkpoint,
        create_analysis_logger,
        ProgressBar,
        parse_family_list,
        filter_experiments_by_family,
        read_config_hints,
        resolve_checkpoint,
        scan_experiments,
        select_main_output,
        set_deterministic,
    )
except ImportError:  # pragma: no cover - fallback for direct script execution
    from analysis_utils import (
        FAMILY_CONFIG,
        activate_family_dir,
        build_config,
        build_val_loader,
        cleanup_family_dir,
        default_experiments_root,
        default_output_dir,
        detect_family_variant,
        disable_pretrained,
        extract_variant_from_hints,
        extract_vmamba_weight_set,
        import_family_modules,
        infer_vmamba_weight_set_from_logs,
        load_model_with_checkpoint,
        create_analysis_logger,
        ProgressBar,
        parse_family_list,
        filter_experiments_by_family,
        read_config_hints,
        resolve_checkpoint,
        scan_experiments,
        select_main_output,
        set_deterministic,
    )


def _setup_logger(output_dir: Path) -> logging.Logger:
    return create_analysis_logger("RotationAnalysis", output_dir)


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["backbone_name", "miou_0", "miou_90", "miou_180", "miou_270", "rotation_std"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _plot(path: Path, rows: List[Dict[str, Any]]) -> None:
    angles = [0, 90, 180, 270]
    fig_w = max(6.0, 0.9 * len(rows))
    fig, ax = plt.subplots(figsize=(fig_w, 4.5))
    for row in rows:
        y = [row["miou_0"], row["miou_90"], row["miou_180"], row["miou_270"]]
        ax.plot(angles, y, marker="o", linewidth=2, label=row["backbone_name"])
    ax.set_xlabel("Rotation (degrees)")
    ax.set_ylabel("mIoU")
    ax.set_title("Rotation Robustness")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(ncol=2, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def _evaluate_experiment(
    exp_dir: Path,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    data_root: Optional[str],
    use_amp_override: Optional[int],
    use_pretrained: bool,
    pack_rotations: bool,
    logger: logging.Logger,
) -> Optional[Dict[str, Any]]:
    family, variant = detect_family_variant(exp_dir.name)
    if not family:
        logger.warning("Skipping unrecognized experiment: %s", exp_dir.name)
        return None

    meta = FAMILY_CONFIG[family]
    hints = read_config_hints(exp_dir)
    hinted_variant = extract_variant_from_hints(hints, family)
    if hinted_variant:
        variant = hinted_variant
    if not variant or variant not in meta["variants"]:
        logger.warning("Unsupported variant for %s: %s", exp_dir.name, variant)
        return None

    ckpt_path = resolve_checkpoint(exp_dir)
    if not ckpt_path:
        logger.warning("Missing checkpoint for %s", exp_dir.name)
        return None

    weight_set = extract_vmamba_weight_set(hints) if family == "vmamba" else None
    if family == "vmamba" and not weight_set:
        weight_set = infer_vmamba_weight_set_from_logs(exp_dir)
        if weight_set:
            logger.info("Inferred VMAMBA_WEIGHT_SET=%s from logs for %s", weight_set, exp_dir.name)
    Config, build_model, dataset_mod, utils_mod = import_family_modules(
        Path(__file__).resolve().parents[1] / FAMILY_CONFIG[family]["dir_name"]
    )
    cfg = build_config(Config, meta, exp_dir, variant, weight_set=weight_set, data_root=data_root)
    if not use_pretrained:
        disable_pretrained(cfg)

    use_amp = cfg.USE_AMP if hasattr(cfg, "USE_AMP") else False
    if use_amp_override is not None:
        use_amp = bool(use_amp_override)

    cfg, model = load_model_with_checkpoint(
        family=family,
        cfg=cfg,
        build_model=build_model,
        utils_mod=utils_mod,
        ckpt_path=ckpt_path,
        device=device,
        meta=meta,
        variant=variant,
        data_root=data_root,
        weight_set=weight_set,
        use_pretrained=bool(use_pretrained),
        logger=logger,
    )
    if model is None or cfg is None:
        logger.warning("Skipping %s due to checkpoint load failure.", exp_dir.name)
        return None

    model.eval()
    evaluators = {
        0: utils_mod.SegmentationEvaluator(cfg.NUM_CLASSES, device),
        1: utils_mod.SegmentationEvaluator(cfg.NUM_CLASSES, device),
        2: utils_mod.SegmentationEvaluator(cfg.NUM_CLASSES, device),
        3: utils_mod.SegmentationEvaluator(cfg.NUM_CLASSES, device),
    }
    for evaluator in evaluators.values():
        evaluator.reset()

    loader = build_val_loader(
        cfg,
        dataset_mod,
        batch_size=batch_size,
        num_workers=num_workers,
        device=device,
        domain=None,
        split="val",
    )

    with torch.inference_mode():
        for batch in loader:
            rgb = batch["rgb"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)
            batch_size = rgb.shape[0]
            if pack_rotations:
                rgb_rots = [rgb] + [torch.rot90(rgb, k, dims=(2, 3)).contiguous() for k in (1, 2, 3)]
                mask_rots = [mask] + [torch.rot90(mask, k, dims=(1, 2)).contiguous() for k in (1, 2, 3)]
                rgb_pack = torch.cat(rgb_rots, dim=0).contiguous()
                mask_pack = torch.cat(mask_rots, dim=0).contiguous()
                with autocast(enabled=use_amp):
                    outputs = model(rgb_pack)
                    main_out = select_main_output(outputs)
                    if main_out.shape[-2:] != mask_pack.shape[-2:]:
                        main_out = F.interpolate(
                            main_out,
                            size=mask_pack.shape[-2:],
                            mode="bilinear",
                            align_corners=False,
                        )
                    pred_pack = main_out.argmax(dim=1)
                for k in (0, 1, 2, 3):
                    start = k * batch_size
                    end = start + batch_size
                    evaluators[k].update(
                        pred_pack[start:end].contiguous(),
                        mask_pack[start:end].contiguous(),
                    )
            else:
                for k in (0, 1, 2, 3):
                    rgb_rot = torch.rot90(rgb, k, dims=(2, 3)).contiguous() if k else rgb
                    mask_rot = torch.rot90(mask, k, dims=(1, 2)).contiguous() if k else mask
                    with autocast(enabled=use_amp):
                        outputs = model(rgb_rot)
                        main_out = select_main_output(outputs)
                        if main_out.shape[-2:] != mask_rot.shape[-2:]:
                            main_out = F.interpolate(
                                main_out,
                                size=mask_rot.shape[-2:],
                                mode="bilinear",
                                align_corners=False,
                            )
                    pred = main_out.argmax(dim=1)
                    evaluators[k].update(pred.contiguous(), mask_rot.contiguous())

    metrics = {k: evaluators[k].get_metrics() for k in evaluators}
    del model
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()
    gc.collect()

    miou_values = [metrics[k].get("mIoU", 0.0) for k in (0, 1, 2, 3)]
    rotation_std = float(np.std(miou_values))

    return {
        "backbone_name": f"{meta['backbone']}-{variant}",
        "miou_0": miou_values[0],
        "miou_90": miou_values[1],
        "miou_180": miou_values[2],
        "miou_270": miou_values[3],
        "rotation_std": rotation_std,
    }


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rotation robustness analysis.")
    parser.add_argument("--root", default=str(default_experiments_root()), help="Experiments root.")
    parser.add_argument("--output_dir", default=str(default_output_dir()), help="Output directory.")
    parser.add_argument("--device", default="cuda:0", help="Torch device (cuda:0 or cpu).")
    parser.add_argument("--batch", type=int, default=1, help="Validation batch size.")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers.")
    parser.add_argument("--data_root", default=None, help="Override dataset root.")
    parser.add_argument("--amp", type=int, choices=[0, 1], default=None, help="Override AMP usage.")
    parser.add_argument("--use_pretrained", type=int, choices=[0, 1], default=1, help="Use pretrained weights.")
    parser.add_argument(
        "--families",
        default="mambavision,vmamba,spatialmamba",
        help="Comma-separated model families to include.",
    )
    parser.add_argument(
        "--pack_rotations",
        type=int,
        choices=[0, 1],
        default=1,
        help="Process 0/90/180/270 in one forward pass (faster, more memory).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir)
    logger = _setup_logger(output_dir)
    set_deterministic(args.seed)

    root = Path(args.root)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA requested but not available; using CPU.")
        device = torch.device("cpu")

    exp_dirs = scan_experiments(root, include_domain=False)
    families = parse_family_list(args.families)
    exp_dirs = filter_experiments_by_family(exp_dirs, families)
    if not exp_dirs:
        logger.warning("No experiments found under %s", root)
        return 1

    family_dirs = {key: Path(__file__).resolve().parents[1] / meta["dir_name"] for key, meta in FAMILY_CONFIG.items()}

    rows: List[Dict[str, Any]] = []
    previous_family_dir: Optional[Path] = None
    grouped: Dict[str, List[Path]] = {k: [] for k in FAMILY_CONFIG}
    for exp_dir in exp_dirs:
        family, _ = detect_family_variant(exp_dir.name)
        if family:
            grouped[family].append(exp_dir)

    total = sum(len(v) for v in grouped.values())
    processed = 0
    logger.info("Starting rotation analysis with %d experiments", total)
    start_time = time.perf_counter()
    progress = ProgressBar(total, prefix="Rotation ")
    for family in sorted(grouped.keys()):
        if not grouped[family]:
            continue
        family_dir = family_dirs[family]
        if not family_dir.exists():
            logger.warning("Family directory missing: %s", family_dir)
            continue
        previous_family_dir = activate_family_dir(family_dir, previous_family_dir)

        for exp_dir in grouped[family]:
            processed += 1
            logger.info("Processing %d/%d: %s", processed, total, exp_dir.name)
            exp_start = time.perf_counter()
            try:
                result = _evaluate_experiment(
                    exp_dir=exp_dir,
                    device=device,
                    batch_size=args.batch,
                    num_workers=args.num_workers,
                    data_root=args.data_root,
                    use_amp_override=args.amp,
                use_pretrained=bool(args.use_pretrained),
                pack_rotations=bool(args.pack_rotations),
                logger=logger,
            )
            except Exception as exc:
                logger.warning("Failed %s: %s", exp_dir.name, exc)
                result = None
            if result:
                rows.append(result)
                elapsed = time.perf_counter() - exp_start
                logger.info("Finished %s in %.1fs", exp_dir.name, elapsed)
                pct = 100.0 * processed / max(1, total)
                total_elapsed = time.perf_counter() - start_time
                eta = (total_elapsed / max(1, processed)) * (total - processed) if processed else 0.0
                logger.info(
                    "Progress %d/%d (%.1f%%) elapsed %.1fs ETA %.1fs",
                    processed,
                    total,
                    pct,
                    total_elapsed,
                    eta,
                )
            progress.update(processed, exp_dir.name)

    cleanup_family_dir(previous_family_dir)
    progress.finish()
    logger.info("Rotation analysis completed in %.1fs", time.perf_counter() - start_time)

    rows = sorted(rows, key=lambda r: r["backbone_name"])
    csv_path = output_dir / "rotation_robustness.csv"
    plot_path = output_dir / "rotation_robustness.png"
    _write_csv(csv_path, rows)
    _plot(plot_path, rows)
    logger.info("Wrote %s and %s", csv_path, plot_path)
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
