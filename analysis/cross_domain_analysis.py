#!/usr/bin/env python3
"""
Cross-domain error decomposition analysis for LoveDA.
"""

from __future__ import annotations

import argparse
import csv
import gc
import logging
import time
from math import ceil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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


STRUCTURAL_CLASSES = [1, 2]  # Building, Road
NATURAL_CLASSES = [3, 5, 6]  # Water, Forest, Agricultural


def _setup_logger(output_dir: Path) -> logging.Logger:
    return create_analysis_logger("CrossDomainAnalysis", output_dir)


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["backbone_name", "direction", "structural_mIoU", "natural_mIoU"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _plot(path: Path, rows: List[Dict[str, Any]]) -> None:
    data: Dict[str, Dict[str, Dict[str, float]]] = {}
    for row in rows:
        backbone = row["backbone_name"]
        direction = row["direction"]
        data.setdefault(backbone, {})[direction] = {
            "structural": row["structural_mIoU"],
            "natural": row["natural_mIoU"],
        }

    backbones = sorted(data.keys())
    if not backbones:
        return

    ncols = min(3, len(backbones))
    nrows = int(ceil(len(backbones) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.0 * ncols, 3.6 * nrows), sharey=True)
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    axes = axes.reshape(nrows, ncols)

    directions = ["U2R", "R2U"]
    x = np.arange(len(directions))
    width = 0.35

    for idx, backbone in enumerate(backbones):
        ax = axes[idx // ncols, idx % ncols]
        values = data[backbone]
        structural = [values.get(d, {}).get("structural", 0.0) for d in directions]
        natural = [values.get(d, {}).get("natural", 0.0) for d in directions]
        ax.bar(x - width / 2, structural, width, label="Structural", color="#7570b3")
        ax.bar(x + width / 2, natural, width, label="Natural", color="#66a61e")
        ax.set_title(backbone)
        ax.set_xticks(x)
        ax.set_xticklabels(directions)
        ax.set_ylim(0.0, 1.0)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        if idx == 0:
            ax.legend(fontsize=8)

    for idx in range(len(backbones), nrows * ncols):
        axes[idx // ncols, idx % ncols].axis("off")

    fig.suptitle("Cross-Domain Group IoU", y=1.02)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _group_mean_iou(confusion: torch.Tensor, indices: List[int]) -> float:
    cm = confusion.float()
    tp = cm.diag()
    fp = cm.sum(dim=0) - tp
    fn = cm.sum(dim=1) - tp
    denom = tp + fp + fn
    iou = torch.where(denom > 0, tp / denom, torch.zeros_like(tp))
    idx = torch.tensor(indices, dtype=torch.long)
    denom_sel = denom[idx]
    iou_sel = iou[idx]
    valid = denom_sel > 0
    if valid.any():
        return iou_sel[valid].mean().item()
    return 0.0


def _evaluate_direction(
    exp_dir: Path,
    domain: str,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    data_root: Optional[str],
    use_amp_override: Optional[int],
    use_pretrained: bool,
    logger: logging.Logger,
) -> Optional[Tuple[float, float]]:
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

    dataset_name = getattr(cfg, "DATASET", "loveda").lower()
    if dataset_name != "loveda":
        logger.warning("Cross-domain analysis supports LoveDA only (%s)", exp_dir.name)
        return None

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
    evaluator = utils_mod.SegmentationEvaluator(cfg.NUM_CLASSES, device)
    evaluator.reset()

    loader = build_val_loader(
        cfg,
        dataset_mod,
        batch_size=batch_size,
        num_workers=num_workers,
        device=device,
        domain=domain,
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
            evaluator.update(pred, mask)

    confusion = evaluator.confusion_matrix.detach().cpu()
    structural = _group_mean_iou(confusion, STRUCTURAL_CLASSES)
    natural = _group_mean_iou(confusion, NATURAL_CLASSES)

    del model
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()
    gc.collect()
    return structural, natural


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cross-domain group IoU analysis.")
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

    exp_dirs = scan_experiments(root, include_domain=True)
    families = parse_family_list(args.families)
    exp_dirs = filter_experiments_by_family(exp_dirs, families)
    if not exp_dirs:
        logger.warning("No experiments found under %s", root)
        return 1

    family_dirs = {key: Path(__file__).resolve().parents[1] / meta["dir_name"] for key, meta in FAMILY_CONFIG.items()}

    grouped: Dict[str, Dict[str, Path]] = {}
    for exp_dir in exp_dirs:
        name_lower = exp_dir.name.lower()
        if "urbantrain" in name_lower:
            domain = "urban"
        elif "ruraltrain" in name_lower:
            domain = "rural"
        else:
            continue
        family, variant = detect_family_variant(exp_dir.name)
        if not family or not variant:
            continue
        hints = read_config_hints(exp_dir)
        hinted_variant = extract_variant_from_hints(hints, family)
        if hinted_variant:
            variant = hinted_variant
        base_name = f"{FAMILY_CONFIG[family]['backbone']}-{variant}"
        grouped.setdefault(base_name, {})[domain] = exp_dir

    rows: List[Dict[str, Any]] = []
    previous_family_dir: Optional[Path] = None
    total = sum(2 for pair in grouped.values() if "urban" in pair and "rural" in pair)
    processed = 0
    logger.info("Starting cross-domain analysis with %d evaluations", total)
    start_time = time.perf_counter()
    progress = ProgressBar(total, prefix="CrossDomain ")

    for base_name in sorted(grouped.keys()):
        pair = grouped[base_name]
        if "urban" not in pair or "rural" not in pair:
            logger.warning("Missing urban/rural pair for %s", base_name)
            continue
        for domain, direction in (("rural", "U2R"), ("urban", "R2U")):
            exp_dir = pair["urban"] if direction == "U2R" else pair["rural"]
            family, _ = detect_family_variant(exp_dir.name)
            if not family:
                continue
            family_dir = family_dirs[family]
            if not family_dir.exists():
                logger.warning("Family directory missing: %s", family_dir)
                continue
            previous_family_dir = activate_family_dir(family_dir, previous_family_dir)

            processed += 1
            logger.info("Processing %d/%d: %s (%s)", processed, total, exp_dir.name, direction)
            exp_start = time.perf_counter()
            try:
                result = _evaluate_direction(
                    exp_dir=exp_dir,
                    domain=domain,
                    device=device,
                    batch_size=args.batch,
                    num_workers=args.num_workers,
                    data_root=args.data_root,
                    use_amp_override=args.amp,
                    use_pretrained=bool(args.use_pretrained),
                    logger=logger,
                )
            except Exception as exc:
                logger.warning("Failed %s (%s): %s", exp_dir.name, direction, exc)
                result = None
            if result is None:
                progress.update(processed, f"{exp_dir.name}:{direction}")
                continue
            structural, natural = result
            rows.append(
                {
                    "backbone_name": base_name,
                    "direction": direction,
                    "structural_mIoU": structural,
                    "natural_mIoU": natural,
                }
            )
            elapsed = time.perf_counter() - exp_start
            logger.info("Finished %s (%s) in %.1fs", exp_dir.name, direction, elapsed)
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
            progress.update(processed, f"{exp_dir.name}:{direction}")

    cleanup_family_dir(previous_family_dir)
    progress.finish()
    logger.info("Cross-domain analysis completed in %.1fs", time.perf_counter() - start_time)

    rows = sorted(rows, key=lambda r: (r["backbone_name"], r["direction"]))
    csv_path = output_dir / "cross_domain_groups.csv"
    plot_path = output_dir / "cross_domain_groups.png"
    _write_csv(csv_path, rows)
    _plot(plot_path, rows)
    logger.info("Wrote %s and %s", csv_path, plot_path)
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
