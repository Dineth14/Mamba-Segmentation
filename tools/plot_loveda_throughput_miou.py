#!/usr/bin/env python3
"""Plot Loveda 512 throughput vs mIoU for all model families.

Reads FPS from fps_mem_allall.csv and best mIoU from training logs under
Comparison_Experiments. Produces a single scatter/line plot similar to
paper-style throughput vs accuracy figures.
"""
from __future__ import annotations

import argparse
import csv
import os
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt


@dataclass(frozen=True)
class Point:
    family: str
    variant: str
    exp_name: str
    fps: float
    miou: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot Loveda 512 throughput vs mIoU for all models."
    )
    parser.add_argument(
        "--exp-root",
        default="/storage2/ChangeDetection/NSST-mamba/Mamba-Segmentation/Comparison_Experiments",
        help="Path to Comparison_Experiments directory.",
    )
    parser.add_argument(
        "--fps-csv",
        default="/storage2/ChangeDetection/NSST-mamba/Mamba-Segmentation/Comparison_Experiments/fps_mem_allall.csv",
        help="CSV with FPS measurements.",
    )
    parser.add_argument(
        "--out",
        default="/storage2/ChangeDetection/NSST-mamba/Mamba-Segmentation/Comparison_Experiments/loveda_512_throughput_miou.png",
        help="Output plot path.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show the plot interactively.",
    )
    return parser.parse_args()


def parse_family_variant(exp_name: str) -> Tuple[Optional[str], Optional[str]]:
    name = exp_name.lower()
    if name.startswith("mambavision_"):
        parts = name.split("_")
        return "MambaVision", parts[1] if len(parts) > 1 else None
    if name.startswith("spatialmamba_"):
        parts = name.split("_")
        return "SpatialMamba", parts[1] if len(parts) > 1 else None
    if name.startswith("vmamba_") or name.startswith("vmamb_"):
        parts = name.split("_")
        return "VMamba", parts[1] if len(parts) > 1 else None
    return None, None


def normalize_exp_name(exp_name: str) -> str:
    return re.sub(r"_\d+$", "", exp_name)


def to_percent(value: float) -> float:
    return value * 100.0 if value <= 1.5 else value


def read_fps_csv(path: str) -> Tuple[Dict[str, float], Dict[Tuple[str, str], float]]:
    fps_by_exp: Dict[str, float] = {}
    fps_by_family: Dict[Tuple[str, str], float] = {}
    if not os.path.exists(path):
        raise FileNotFoundError(f"FPS CSV not found: {path}")
    with open(path, "r", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            exp_name = row.get("exp_name", "")
            fps_str = row.get("fps", "")
            if not exp_name or not fps_str:
                continue
            try:
                fps = float(fps_str)
            except ValueError:
                continue
            fps_by_exp[exp_name] = fps
            family, variant = parse_family_variant(exp_name)
            if family and variant:
                fps_by_family[(family, variant)] = fps
    return fps_by_exp, fps_by_family


def extract_best_miou(log_path: str) -> Optional[float]:
    best: Optional[float] = None
    mean_candidates: List[float] = []
    best_re = re.compile(r"New best mIoU: ([0-9.]+)")
    achieved_re = re.compile(r"Best mIoU achieved: ([0-9.]+)")
    mean_re = re.compile(r"^Mean\s+([0-9.]+)")

    with open(log_path, "r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            match = best_re.search(line)
            if match:
                value = to_percent(float(match.group(1)))
                best = value if best is None else max(best, value)
                continue
            match = achieved_re.search(line)
            if match:
                value = to_percent(float(match.group(1)))
                best = value if best is None else max(best, value)
                continue
            match = mean_re.search(line.strip())
            if match:
                mean_candidates.append(to_percent(float(match.group(1))))

    if best is not None:
        return best
    return max(mean_candidates) if mean_candidates else None


def iter_experiments(exp_root: str) -> Iterable[str]:
    for entry in sorted(os.listdir(exp_root)):
        path = os.path.join(exp_root, entry)
        if not os.path.isdir(path):
            continue
        name = entry.lower()
        if "_512" not in name:
            continue
        if any(token in name for token in ("urbantrain", "ruraltrain", "icprs", "isprs")):
            continue
        yield entry


def collect_points(exp_root: str, fps_csv: str) -> List[Point]:
    fps_by_exp, fps_by_family = read_fps_csv(fps_csv)
    points: Dict[Tuple[str, str], Point] = {}

    for exp_name in iter_experiments(exp_root):
        family, variant = parse_family_variant(exp_name)
        if not family or not variant:
            continue

        exp_dir = os.path.join(exp_root, exp_name)
        log_files = [
            os.path.join(exp_dir, f)
            for f in os.listdir(exp_dir)
            if f.startswith("train_") and f.endswith(".log")
        ]
        if not log_files:
            continue

        miou_values = [v for v in (extract_best_miou(p) for p in log_files) if v is not None]
        if not miou_values:
            continue
        miou = max(miou_values)

        fps = fps_by_exp.get(exp_name)
        if fps is None:
            fps = fps_by_exp.get(normalize_exp_name(exp_name))
        if fps is None:
            fps = fps_by_family.get((family, variant))
        if fps is None:
            print(f"[warn] Missing FPS for {exp_name} ({family}/{variant})")
            continue

        key = (family, variant)
        point = Point(family=family, variant=variant, exp_name=exp_name, fps=fps, miou=miou)
        if key not in points or miou > points[key].miou:
            points[key] = point

    return list(points.values())


def format_variant_label(variant: str) -> str:
    if variant.endswith("2"):
        return variant[:-1].capitalize() + "2"
    return variant.capitalize()


def plot(points: List[Point], out_path: str, show: bool) -> None:
    style = {
        "MambaVision": dict(color="#1f77b4", marker="o", label="MambaVision"),
        "SpatialMamba": dict(color="#2ca02c", marker="^", label="SpatialMamba"),
        "VMamba": dict(color="#ff7f0e", marker="s", label="VMamba"),
    }

    plt.figure(figsize=(9, 6), dpi=150)

    for family, cfg in style.items():
        family_points = sorted(
            (p for p in points if p.family == family),
            key=lambda p: p.fps,
        )
        if not family_points:
            continue
        xs = [p.fps for p in family_points]
        ys = [p.miou for p in family_points]
        plt.plot(xs, ys, color=cfg["color"], linewidth=2)
        plt.scatter(xs, ys, color=cfg["color"], marker=cfg["marker"], s=90, label=cfg["label"])

        for p in family_points:
            label = format_variant_label(p.variant)
            plt.text(p.fps + 0.3, p.miou + 0.1, label, color=cfg["color"], fontsize=9)

    plt.xlabel("Throughput (FPS)", fontsize=12)
    plt.ylabel("mIoU (%)", fontsize=12)
    plt.grid(False)
    plt.legend(loc="lower left", frameon=False)
    plt.tight_layout()

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=200)
    if show:
        plt.show()
    plt.close()


def main() -> None:
    args = parse_args()
    points = collect_points(args.exp_root, args.fps_csv)
    if not points:
        raise SystemExit("No points found. Check log paths and FPS CSV.")

    print("Collected points:")
    for p in sorted(points, key=lambda x: (x.family, x.variant)):
        print(f"- {p.family} {p.variant}: FPS={p.fps:.2f}, mIoU={p.miou:.2f} ({p.exp_name})")

    plot(points, args.out, args.show)
    print(f"Saved plot -> {args.out}")


if __name__ == "__main__":
    main()
