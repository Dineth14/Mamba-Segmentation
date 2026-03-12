#!/usr/bin/env python3
"""
Boundary vs interior IoU analysis across backbones.
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
    return create_analysis_logger("BoundaryAnalysis", output_dir)


def _compute_boundary_mask(mask: torch.Tensor, ignore_index: int, radius: int = 2) -> torch.Tensor:
    """Compute dilated boundary mask using a morphological gradient."""
    valid = mask != ignore_index
    mask_f = mask.float().unsqueeze(1)
    max_pool = F.max_pool2d(mask_f, kernel_size=3, stride=1, padding=1)
    min_pool = -F.max_pool2d(-mask_f, kernel_size=3, stride=1, padding=1)
    edge = (max_pool != min_pool).squeeze(1)
    edge = edge & valid
    if radius > 0:
        kernel = radius * 2 + 1
        edge = F.max_pool2d(edge.float().unsqueeze(1), kernel_size=kernel, stride=1, padding=radius) > 0
        edge = edge.squeeze(1)
    return edge & valid


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["backbone_name", "mIoU_boundary", "mIoU_interior", "delta"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _plot(path: Path, rows: List[Dict[str, Any]]) -> None:
    labels = [row["backbone_name"] for row in rows]
    boundary = [row["mIoU_boundary"] for row in rows]
    interior = [row["mIoU_interior"] for row in rows]

    x = np.arange(len(labels))
    width = 0.36
    fig_w = max(6.0, 0.8 * len(labels))
    fig, ax = plt.subplots(figsize=(fig_w, 4.5))
    ax.bar(x - width / 2, boundary, width, label="Boundary", color="#d95f02")
    ax.bar(x + width / 2, interior, width, label="Interior", color="#1b9e77")
    ax.set_ylabel("mIoU")
    ax.set_title("Boundary vs Interior IoU")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylim(0.0, 1.0)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend()
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
    evaluator_boundary = utils_mod.SegmentationEvaluator(cfg.NUM_CLASSES, device)
    evaluator_interior = utils_mod.SegmentationEvaluator(cfg.NUM_CLASSES, device)
    evaluator_boundary.reset()
    evaluator_interior.reset()

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
            with autocast(enabled=use_amp):
                outputs = model(rgb)
                main_out = select_main_output(outputs)
                if main_out.shape[-2:] != mask.shape[-2:]:
                    main_out = F.interpolate(
                        main_out,
                        size=mask.shape[-2:],
                        mode="bilinear",
                        align_corners=False,
                    )
            pred = main_out.argmax(dim=1)
            boundary = _compute_boundary_mask(mask, cfg.IGNORE_INDEX, radius=2)
            interior = (mask != cfg.IGNORE_INDEX) & (~boundary)

            boundary_target = mask.clone()
            boundary_target[~boundary] = cfg.IGNORE_INDEX
            evaluator_boundary.update(pred, boundary_target)

            interior_target = mask.clone()
            interior_target[~interior] = cfg.IGNORE_INDEX
            evaluator_interior.update(pred, interior_target)

    metrics_boundary = evaluator_boundary.get_metrics()
    metrics_interior = evaluator_interior.get_metrics()
    del model
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()
    gc.collect()

    return {
        "backbone_name": f"{meta['backbone']}-{variant}",
        "mIoU_boundary": metrics_boundary.get("mIoU", 0.0),
        "mIoU_interior": metrics_interior.get("mIoU", 0.0),
    }


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Boundary vs interior IoU analysis.")
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
    logger.info("Starting boundary analysis with %d experiments", total)
    start_time = time.perf_counter()
    progress = ProgressBar(total, prefix="Boundary ")
    processed = 0
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
            idx = processed
            logger.info("Processing %d/%d: %s", idx, total, exp_dir.name)
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
                    logger=logger,
                )
            except Exception as exc:
                logger.warning("Failed %s: %s", exp_dir.name, exc)
                result = None
            if not result:
                progress.update(processed, exp_dir.name)
                continue
            result["delta"] = result["mIoU_interior"] - result["mIoU_boundary"]
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
    logger.info("Boundary analysis completed in %.1fs", time.perf_counter() - start_time)

    rows = sorted(rows, key=lambda r: r["backbone_name"])
    csv_path = output_dir / "boundary_vs_interior.csv"
    plot_path = output_dir / "boundary_interior_iou.png"
    _write_csv(csv_path, rows)
    _plot(plot_path, rows)
    logger.info("Wrote %s and %s", csv_path, plot_path)
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
