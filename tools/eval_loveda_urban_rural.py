#!/usr/bin/env python3
"""Evaluate LoveDA Urban/Rural validation splits for trained checkpoints.

Usage examples:
  python tools/eval_loveda_urban_rural.py \
    --loveda_root /data/LoveDA \
    --runs cnn_deeplabv3p_resnet50_urbantrain_512,cnn_deeplabv3p_resnet50_ruraltrain_512

  python tools/eval_loveda_urban_rural.py \
    --loveda_root /data/LoveDA \
    --runs cnn_deeplabv3p_resnet50_urbantrain_512,cnn_deeplabv3p_resnet50_ruraltrain_512,transformerunetformer_resnet18_urbantrain_512,transformerunetformer_resnet18_ruraltrain_512 \
    --config_map cnn_deeplabv3p_resnet50_urbantrain_512=/abs/configs/cnn_urban.py,cnn_deeplabv3p_resnet50_ruraltrain_512=/abs/configs/cnn_rural.py,transformerunetformer_resnet18_urbantrain_512=/abs/configs/unetformer_urban.py,transformerunetformer_resnet18_ruraltrain_512=/abs/configs/unetformer_rural.py

  python tools/eval_loveda_urban_rural.py \
    --root_exp Comparison_Experiments \
    --loveda_root /data/LoveDA \
    --runs transformerunetformer_resnet18_urbantrain_512 \
    --gpu 1 \
    --eval mIoU

  python tools/eval_loveda_urban_rural.py \
    --loveda_root /data/LoveDA \
    --runs vmamba_base_urbantrain_512 \
    --split_mode val \
    --dry_run
"""

import argparse
import csv
import datetime
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate LoveDA Urban/Rural validation splits for checkpoints."
    )
    parser.add_argument(
        "--root_exp",
        default="Comparison_Experiments",
        help="Path to experiment folder (default: Comparison_Experiments).",
    )
    parser.add_argument(
        "--loveda_root",
        required=True,
        help="LoveDA dataset root path.",
    )
    parser.add_argument(
        "--runs",
        required=True,
        help="Comma-separated list of run folder names.",
    )
    parser.add_argument(
        "--config_map",
        default=None,
        help=(
            "Optional map of run_name=/abs/path/to/config.py entries, "
            "comma-separated."
        ),
    )
    parser.add_argument(
        "--gpu",
        default="0",
        help="CUDA device id (default: 0).",
    )
    parser.add_argument(
        "--eval",
        default="mIoU",
        help='Metrics string for mmseg test (default: "mIoU").',
    )
    parser.add_argument(
        "--split_mode",
        default="test",
        choices=("test", "val"),
        help='Override data.<split_mode> in cfg-options (default: "test").',
    )
    parser.add_argument(
        "--allow_config_from_ckpt",
        dest="allow_config_from_ckpt",
        action="store_true",
        default=True,
        help="Allow recovering config from checkpoint meta (default: True).",
    )
    parser.add_argument(
        "--no_allow_config_from_ckpt",
        dest="allow_config_from_ckpt",
        action="store_false",
        help="Disable recovering config from checkpoint meta.",
    )
    parser.add_argument(
        "--allow_config_from_log",
        dest="allow_config_from_log",
        action="store_true",
        default=True,
        help="Allow recovering config from train logs (default: True).",
    )
    parser.add_argument(
        "--no_allow_config_from_log",
        dest="allow_config_from_log",
        action="store_false",
        help="Disable recovering config from train logs.",
    )
    parser.add_argument(
        "--dump_recovered_config_name",
        default="config_recovered.py",
        help="Filename to save recovered config text into run folder.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Only print commands without running evaluation.",
    )
    return parser.parse_args()


def parse_runs(runs_csv: str) -> List[str]:
    runs = [run.strip() for run in runs_csv.split(",") if run.strip()]
    if not runs:
        raise ValueError("No runs provided after parsing --runs.")
    return runs


def parse_config_map(config_map: Optional[str]) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    if not config_map:
        return mapping
    for entry in config_map.split(","):
        entry = entry.strip()
        if not entry:
            continue
        key, sep, value = entry.partition("=")
        if not sep or not key.strip() or not value.strip():
            raise ValueError(
                f"Invalid --config_map entry: '{entry}'. "
                "Expected run_name=/abs/path/to/config.py"
            )
        mapping[key.strip()] = value.strip()
    return mapping


def resolve_root_exp(root_exp: str, repo_root: Path) -> Path:
    root_path = Path(root_exp).expanduser()
    if not root_path.is_absolute():
        root_path = repo_root / root_path
    return root_path


def warn(message: str) -> None:
    print(f"[warn] {message}", file=sys.stderr)


def _newest_file(paths: Iterable[Path]) -> Optional[Path]:
    candidates = [p for p in paths if p.is_file()]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def find_checkpoint(run_dir: Path) -> Path:
    ckpt_dir = run_dir / "checkpoints"
    best_path = ckpt_dir / "best.pth"
    if best_path.exists():
        return best_path

    best_candidates = _newest_file(ckpt_dir.glob("best_*.pth"))
    if best_candidates:
        return best_candidates

    any_candidates = _newest_file(ckpt_dir.glob("*.pth"))
    if any_candidates:
        return any_candidates

    raise FileNotFoundError(
        f"No checkpoint found in {ckpt_dir}. "
        "Searched best.pth, best_*.pth, and *.pth."
    )


def _clean_path_value(value: str) -> str:
    return value.strip().strip("'\"").rstrip(",")


def _resolve_config_path(value: str, run_dir: Path, repo_root: Path) -> Optional[Path]:
    cleaned = _clean_path_value(value)
    if not cleaned:
        return None
    expanded = os.path.expandvars(cleaned)
    path = Path(expanded).expanduser()
    if path.is_absolute():
        return path if path.exists() else None
    for base in (run_dir, repo_root):
        candidate = base / path
        if candidate.exists():
            return candidate
    return None


def _looks_like_config_text(text: str) -> bool:
    lowered = text.lower()
    has_model = "model" in lowered
    has_data = "data" in lowered or "dataset_type" in lowered
    return has_model and has_data


def _recover_config_from_checkpoint(
    ckpt_path: Path,
    run_dir: Path,
    repo_root: Path,
    dump_recovered_config_name: str,
) -> Optional[Path]:
    try:
        import torch
    except Exception as exc:
        warn(f"torch not available to load checkpoint meta: {exc}")
        return None
    try:
        checkpoint = torch.load(str(ckpt_path), map_location="cpu")
    except Exception as exc:
        warn(f"Failed to load checkpoint meta from {ckpt_path}: {exc}")
        return None
    if not isinstance(checkpoint, dict):
        return None
    meta = checkpoint.get("meta", {})
    if not isinstance(meta, dict):
        return None
    for key in ("config", "cfg", "mmengine_config", "mmseg_config"):
        value = meta.get(key)
        if not isinstance(value, str) or not value.strip():
            continue
        if _looks_like_config_text(value):
            recovered_path = run_dir / dump_recovered_config_name
            recovered_path.write_text(value, encoding="utf-8")
            return recovered_path
        resolved = _resolve_config_path(value, run_dir, repo_root)
        if resolved:
            return resolved
    return None


def _recover_config_from_log(run_dir: Path, repo_root: Path) -> Optional[Path]:
    log_files = sorted(
        run_dir.glob("train_*.log"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not log_files:
        return None
    log_path = log_files[0]
    head_lines: List[str] = []
    try:
        with log_path.open("r", encoding="utf-8", errors="ignore") as f:
            for _ in range(200):
                line = f.readline()
                if not line:
                    break
                head_lines.append(line)
    except OSError as exc:
        warn(f"Failed to read log {log_path}: {exc}")
        return None

    patterns = [
        r"Config\s*[:=]\s*['\"]?([^'\"]+\.py)",
        r"cfg\s*[:=]\s*['\"]?([^'\"]+\.py)",
        r"load from\s*['\"]?([^'\"]+\.py)",
    ]

    for line in head_lines:
        for pattern in patterns:
            matches = re.findall(pattern, line, flags=re.IGNORECASE)
            for match in matches:
                resolved = _resolve_config_path(match, run_dir, repo_root)
                if resolved:
                    return resolved
        generic = re.findall(r"[A-Za-z0-9_./-]+\.py", line)
        for match in generic:
            lowered = match.lower()
            if "config" not in lowered and "cfg" not in lowered:
                continue
            resolved = _resolve_config_path(match, run_dir, repo_root)
            if resolved:
                return resolved
    return None


def _tokenize_run_name(run_name: str) -> List[str]:
    stop_tokens = {
        "urbantrain",
        "ruraltrain",
        "train",
        "val",
        "test",
        "batch",
        "bs",
    }
    tokens: List[str] = []
    for raw in re.split(r"[_\-]", run_name.lower()):
        if not raw or raw in stop_tokens:
            continue
        if not any(ch.isalpha() for ch in raw):
            continue
        tokens.append(raw)
    return sorted(set(tokens))


def _fallback_search_config(repo_root: Path, run_name: str) -> Optional[Path]:
    keywords = ("deeplab", "unetformer", "loveda")
    tokens = _tokenize_run_name(run_name)
    if not tokens:
        return None
    best_path: Optional[Path] = None
    best_score = -1
    best_mtime = -1.0
    for path in repo_root.rglob("*.py"):
        name = path.name.lower()
        if not any(keyword in name for keyword in keywords):
            continue
        token_hits = [token for token in tokens if token in name]
        if not token_hits:
            continue
        score = len(token_hits) + sum(1 for keyword in keywords if keyword in name)
        mtime = path.stat().st_mtime
        if score > best_score or (score == best_score and mtime > best_mtime):
            best_score = score
            best_mtime = mtime
            best_path = path
    return best_path


def _find_local_config(run_dir: Path) -> Optional[Path]:
    explicit = [run_dir / "config.py", run_dir / "cfg.py"]
    for candidate in explicit:
        if candidate.is_file():
            return candidate

    run_py = _newest_file(run_dir.glob("*.py"))
    if run_py:
        return run_py

    logs_dir = run_dir / "logs"
    logs_py = _newest_file(logs_dir.glob("*.py")) if logs_dir.exists() else None
    if logs_py:
        return logs_py

    extra_dirs = [run_dir / "work_dirs", run_dir / "configs"]
    extra_candidates: List[Path] = []
    for extra_dir in extra_dirs:
        if extra_dir.exists():
            extra_candidates.extend(extra_dir.glob("*.py"))
    extra_py = _newest_file(extra_candidates)
    if extra_py:
        return extra_py

    return None


def find_config(
    run_dir: Path,
    run_name: str,
    repo_root: Path,
    ckpt_path: Path,
    config_map: Dict[str, str],
    allow_config_from_ckpt: bool,
    allow_config_from_log: bool,
    dump_recovered_config_name: str,
) -> Path:
    if run_name in config_map:
        mapped = _resolve_config_path(config_map[run_name], run_dir, repo_root)
        if mapped:
            return mapped
        raise FileNotFoundError(
            f"Config map path does not exist for run '{run_name}': "
            f"{config_map[run_name]}"
        )

    local_config = _find_local_config(run_dir)
    if local_config:
        return local_config

    if allow_config_from_ckpt:
        recovered = _recover_config_from_checkpoint(
            ckpt_path, run_dir, repo_root, dump_recovered_config_name
        )
        if recovered:
            return recovered

    if allow_config_from_log:
        recovered = _recover_config_from_log(run_dir, repo_root)
        if recovered:
            return recovered

    fallback = _fallback_search_config(repo_root, run_name)
    if fallback:
        return fallback

    raise FileNotFoundError(
        f"Config not found for run '{run_name}'.\n"
        f"  run_dir: {run_dir}\n"
        f"  ckpt: {ckpt_path}\n"
        "Suggestions:\n"
        "  - add config.py to the run folder\n"
        "  - use --config_map to specify the config\n"
        "  - ensure checkpoints include meta config (if using mmseg)"
    )


def detect_train_domain(run_name: str) -> Tuple[str, List[str]]:
    name_lower = run_name.lower()
    is_urban = "urbantrain" in name_lower
    is_rural = "ruraltrain" in name_lower
    if is_urban and is_rural:
        raise ValueError(
            f"Run name '{run_name}' contains both urbantrain and ruraltrain."
        )
    if not is_urban and not is_rural:
        raise ValueError(
            f"Run name '{run_name}' does not contain urbantrain or ruraltrain."
        )
    if is_urban:
        return "Urban", ["Urban", "Rural"]
    return "Rural", ["Rural", "Urban"]


def resolve_domain_dirs(
    loveda_root: Path, domain: str, split: str
) -> Tuple[str, str]:
    rel_base = Path(domain) / split
    img_candidates = ["images", "images_png"]
    ann_candidates = ["masks", "masks_png", "masks_png_fixed"]

    def _pick(candidates: Sequence[str], kind: str) -> str:
        tried = []
        for name in candidates:
            rel_path = rel_base / name
            abs_path = loveda_root / rel_path
            tried.append(str(abs_path))
            if abs_path.exists():
                return str(rel_path.as_posix())
        tried_msg = "\n  - " + "\n  - ".join(tried)
        raise FileNotFoundError(
            f"Could not find {kind} directory for {domain}/{split}. Tried:{tried_msg}"
        )

    img_dir = _pick(img_candidates, "image")
    ann_dir = _pick(ann_candidates, "mask")
    return img_dir, ann_dir


def build_cmd(
    cfg: Path,
    ckpt: Path,
    eval_str: str,
    split_mode: str,
    loveda_root: Path,
    img_dir: str,
    ann_dir: str,
) -> List[str]:
    return [
        sys.executable,
        "tools/test.py",
        str(cfg),
        str(ckpt),
        "--eval",
        eval_str,
        "--cfg-options",
        f"data.{split_mode}.data_root={loveda_root}",
        f"data.{split_mode}.img_dir={img_dir}",
        f"data.{split_mode}.ann_dir={ann_dir}",
    ]


def run_command(
    cmd: Sequence[str],
    cwd: Path,
    gpu: str,
    dry_run: bool,
) -> Tuple[str, int]:
    cmd_str = " ".join(shlex.quote(str(part)) for part in cmd)
    print(cmd_str)
    if dry_run:
        return "", 0
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    result = subprocess.run(
        cmd,
        cwd=str(cwd),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return result.stdout, result.returncode


def _last_match(patterns: Sequence[str], text: str) -> Optional[str]:
    for pattern in patterns:
        matches = re.findall(pattern, text, flags=re.IGNORECASE)
        if not matches:
            continue
        last = matches[-1]
        if isinstance(last, tuple):
            last = last[0]
        return last
    return None


def parse_metrics(stdout: str) -> Dict[str, str]:
    miou = _last_match(
        [r"\bmIoU\b\s*[:=]\s*([0-9]*\.?[0-9]+)"],
        stdout,
    )
    if miou is None:
        miou = _last_match(
            [r"\bmIoU\b[^0-9]*([0-9]*\.?[0-9]+)"],
            stdout,
        )

    oa = _last_match(
        [r"\bOA\b\s*[:=]\s*([0-9]*\.?[0-9]+)", r"\baAcc\b\s*[:=]\s*([0-9]*\.?[0-9]+)"],
        stdout,
    )
    if oa is None:
        oa = _last_match(
            [r"\bOA\b[^0-9]*([0-9]*\.?[0-9]+)", r"\baAcc\b[^0-9]*([0-9]*\.?[0-9]+)"],
            stdout,
        )

    mf1 = _last_match(
        [r"\bmF1\b\s*[:=]\s*([0-9]*\.?[0-9]+)", r"\bmFscore\b\s*[:=]\s*([0-9]*\.?[0-9]+)"],
        stdout,
    )
    if mf1 is None:
        mf1 = _last_match(
            [r"\bmF1\b[^0-9]*([0-9]*\.?[0-9]+)", r"\bmFscore\b[^0-9]*([0-9]*\.?[0-9]+)"],
            stdout,
        )

    return {
        "miou": miou or "NA",
        "oa": oa or "NA",
        "mf1": mf1 or "NA",
    }


def safe_float(value: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return -1.0


def write_csv(path: Path, rows: List[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "run",
        "train_domain",
        "eval_domain",
        "miou",
        "oa",
        "mf1",
        "ckpt",
        "cfg",
        "timestamp",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def format_table(rows: List[Dict[str, str]]) -> str:
    headers = ["run", "train_domain", "eval_domain", "miou", "oa", "mf1"]
    col_widths = {h: len(h) for h in headers}
    for row in rows:
        for h in headers:
            col_widths[h] = max(col_widths[h], len(str(row.get(h, ""))))

    def _fmt(row: Dict[str, str]) -> str:
        return "  ".join(
            str(row.get(h, "")).ljust(col_widths[h]) for h in headers
        )

    lines = [_fmt({h: h for h in headers})]
    lines.append(_fmt({h: "-" * col_widths[h] for h in headers}))
    lines.extend(_fmt(row) for row in rows)
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    root_exp = resolve_root_exp(args.root_exp, repo_root)
    loveda_root = Path(args.loveda_root).expanduser()
    config_map = parse_config_map(args.config_map)

    if not root_exp.exists():
        raise FileNotFoundError(f"Experiment root not found: {root_exp}")
    if not loveda_root.exists():
        raise FileNotFoundError(f"LoveDA root not found: {loveda_root}")

    runs = parse_runs(args.runs)
    rows: List[Dict[str, str]] = []

    for run in runs:
        run_dir = root_exp / run
        if not run_dir.exists():
            raise FileNotFoundError(f"Run directory not found: {run_dir}")

        ckpt_path = find_checkpoint(run_dir)
        cfg_path = find_config(
            run_dir,
            run,
            repo_root,
            ckpt_path,
            config_map,
            args.allow_config_from_ckpt,
            args.allow_config_from_log,
            args.dump_recovered_config_name,
        )
        train_domain, eval_domains = detect_train_domain(run)

        for eval_domain in eval_domains:
            img_dir, ann_dir = resolve_domain_dirs(
                loveda_root, eval_domain, "val"
            )
            cmd = build_cmd(
                cfg_path,
                ckpt_path,
                args.eval,
                args.split_mode,
                loveda_root,
                img_dir,
                ann_dir,
            )
            stdout, returncode = run_command(
                cmd, repo_root, args.gpu, args.dry_run
            )
            metrics = parse_metrics(stdout) if stdout else {"miou": "NA", "oa": "NA", "mf1": "NA"}
            if returncode != 0 and not args.dry_run:
                print(
                    f"[warn] Command failed for {run} ({eval_domain}) with code {returncode}.",
                    file=sys.stderr,
                )

            timestamp = datetime.datetime.now().isoformat(timespec="seconds")
            rows.append(
                {
                    "run": run,
                    "train_domain": train_domain,
                    "eval_domain": eval_domain,
                    "miou": metrics["miou"],
                    "oa": metrics["oa"],
                    "mf1": metrics["mf1"],
                    "ckpt": str(ckpt_path),
                    "cfg": str(cfg_path),
                    "timestamp": timestamp,
                }
            )

    out_csv = root_exp / "eval_loveda_urban_rural_summary.csv"
    if not args.dry_run:
        write_csv(out_csv, rows)

    sorted_rows = sorted(rows, key=lambda r: safe_float(r["miou"]), reverse=True)
    if sorted_rows:
        print("\nResults (sorted by mIoU):")
        print(format_table(sorted_rows))
    else:
        print("No results to summarize.")

    if not args.dry_run:
        print(f"\nWrote CSV to: {out_csv}")
    else:
        print("\nDry run: skipped evaluation execution and CSV write.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
