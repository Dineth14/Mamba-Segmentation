#!/usr/bin/env python3
"""Parent runner to benchmark All->All experiments in isolated processes."""

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark All->All experiments in isolated processes."
    )
    parser.add_argument(
        "--root",
        default="/storage2/ChangeDetection/NSST-mamba/Mamba-Segmentation/Comparison_Experiments",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--img-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--iters", type=int, default=300)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--out", default=None)
    return parser.parse_args()


def warn(msg: str) -> None:
    print(f"[warn] {msg}", file=sys.stderr)


def scan_experiments(root: str) -> List[Path]:
    root_path = Path(root)
    if not root_path.exists():
        raise FileNotFoundError(f"Root directory not found: {root}")
    exp_dirs: List[Path] = []
    for entry in root_path.iterdir():
        if not entry.is_dir():
            continue
        name_lower = entry.name.lower()
        if "urbantrain" in name_lower or "ruraltrain" in name_lower:
            continue
        exp_dirs.append(entry)
    return sorted(exp_dirs)


def resolve_checkpoint(exp_dir: Path) -> Optional[Path]:
    ckpt = exp_dir / "checkpoints" / "best.pth"
    return ckpt if ckpt.exists() else None


def print_env_info(device: str) -> None:
    try:
        import torch

        print(f"[info] torch={torch.__version__} cuda={torch.version.cuda}")
        if torch.cuda.is_available():
            idx = torch.device(device).index or 0
            print(f"[info] gpu={torch.cuda.get_device_name(idx)}")
        else:
            print("[info] cuda not available")
    except Exception as exc:
        warn(f"Failed to read torch/cuda info: {exc}")


def parse_worker_json(stdout: str) -> Optional[Dict[str, Any]]:
    for line in stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict) and payload.get("exp_name"):
            return payload
    return None


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "exp_name",
        "backbone",
        "variant",
        "ckpt_path",
        "img_size",
        "batch_size",
        "amp",
        "iters",
        "warmup",
        "fps",
        "ms_per_img",
        "baseline_alloc_gb",
        "baseline_reserved_gb",
        "peak_alloc_gb",
        "peak_reserved_gb",
        "peak_activation_alloc_gb",
        "peak_activation_reserved_gb",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def print_summary(rows: List[Dict[str, Any]]) -> None:
    if not rows:
        print("No results to summarize.")
        return

    def _fmt(row: Dict[str, Any]) -> str:
        return (
            f"{row['exp_name']:<28} "
            f"{row['backbone']:<12} "
            f"{row['variant']:<8} "
            f"fps={row['fps']:.2f} "
            f"mem={row['peak_alloc_gb']:.2f}GB"
        )

    by_fps = sorted(rows, key=lambda r: r["fps"], reverse=True)
    by_mem = sorted(rows, key=lambda r: r["peak_alloc_gb"])

    print("\n=== Sorted by FPS (desc) ===")
    for row in by_fps:
        print(_fmt(row))

    print("\n=== Sorted by Peak Alloc (asc) ===")
    for row in by_mem:
        print(_fmt(row))


def main() -> int:
    args = parse_args()
    out_path = Path(args.out) if args.out else Path(args.root) / "fps_mem_allall_isolated.csv"
    worker_path = Path(__file__).resolve().parent / "bench_worker.py"
    if not worker_path.exists():
        raise FileNotFoundError(f"Worker not found: {worker_path}")

    print_env_info(args.device)

    exp_dirs = scan_experiments(args.root)
    if not exp_dirs:
        warn("No experiment folders found after filtering")
        return 1

    rows: List[Dict[str, Any]] = []
    for exp_dir in exp_dirs:
        ckpt_path = resolve_checkpoint(exp_dir)
        if not ckpt_path:
            warn(f"Missing checkpoint for {exp_dir.name}; skipping")
            continue

        cmd = [
            sys.executable,
            str(worker_path),
            "--exp_dir",
            str(exp_dir),
            "--device",
            args.device,
            "--img-size",
            str(args.img_size),
            "--batch-size",
            str(args.batch_size),
            "--iters",
            str(args.iters),
            "--warmup",
            str(args.warmup),
        ]
        if args.amp:
            cmd.append("--amp")

        result = subprocess.run(cmd, capture_output=True, text=True)
        payload = parse_worker_json(result.stdout)
        if result.stderr:
            warn(result.stderr.strip())
        if not payload:
            warn(
                f"Worker failed for {exp_dir.name} (code {result.returncode})."
            )
            continue

        rows.append(payload)

    write_csv(out_path, rows)
    print_summary(rows)
    print(f"\nWrote CSV to: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
