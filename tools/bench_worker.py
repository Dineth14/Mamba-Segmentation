#!/usr/bin/env python3
"""Worker to benchmark one experiment in an isolated process."""

import argparse
import gc
import json
import os
import re
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch

FAMILY_CONFIG = {
    "mambavision": {
        "dir_name": "MambaVision",
        "variant_field": "MAMBAVISION_VARIANT",
        "backbone": "MambaVision",
        "variants": {"tiny", "tiny2", "small", "base", "large", "large2"},
    },
    "vmamba": {
        "dir_name": "VMamba",
        "variant_field": "VMAMBA_VARIANT",
        "backbone": "VMamba",
        "variants": {"tiny", "small", "base"},
    },
    "spatialmamba": {
        "dir_name": "spatial-mamba",
        "variant_field": "SPATIALMAMBA_VARIANT",
        "backbone": "SpatialMamba",
        "variants": {"tiny", "small", "base"},
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark a single experiment.")
    parser.add_argument("--exp_dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--img-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--iters", type=int, default=300)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--amp", action="store_true")
    return parser.parse_args()


def warn(msg: str) -> None:
    print(f"[warn] {msg}", file=sys.stderr)


def detect_family_variant(exp_name: str) -> Tuple[Optional[str], Optional[str]]:
    name = exp_name.lower()
    if name.startswith("mambavision"):
        family = "mambavision"
    elif name.startswith("spatialmamba"):
        family = "spatialmamba"
    elif name.startswith("vmamba") or name.startswith("vmamb"):
        family = "vmamba"
    else:
        return None, None

    tokens = [t for t in name.split("_") if t not in ("urbantrain", "ruraltrain")]
    while tokens and tokens[-1].isdigit():
        tokens.pop()
    variant = tokens[1] if len(tokens) >= 2 else None
    return family, variant


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        warn(f"Failed to read JSON config {path}: {exc}")
        return None


def _read_yaml(path: Path) -> Optional[Dict[str, Any]]:
    try:
        import yaml  # type: ignore
    except Exception:
        warn(f"PyYAML not available; skipping YAML config: {path}")
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except Exception as exc:
        warn(f"Failed to read YAML config {path}: {exc}")
        return None


def _read_config_py(path: Path) -> Dict[str, str]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    keys = [
        "MAMBAVISION_VARIANT",
        "VMAMBA_VARIANT",
        "SPATIALMAMBA_VARIANT",
        "VMAMBA_WEIGHT_SET",
    ]
    values: Dict[str, str] = {}
    for key in keys:
        match = re.search(rf"{key}\s*=\s*['\"]([^'\"]+)['\"]", text)
        if match:
            values[key] = match.group(1)
    return values


def read_config_hints(exp_dir: Path) -> Dict[str, Any]:
    hints: Dict[str, Any] = {}
    candidates = [
        exp_dir / "config.json",
        exp_dir / "args.json",
        exp_dir / "config.yaml",
        exp_dir / "config.yml",
        exp_dir / "config.py",
    ]
    for path in candidates:
        if not path.exists():
            continue
        if path.suffix == ".json":
            data = _read_json(path)
            if isinstance(data, dict):
                hints.update(data)
        elif path.suffix in {".yaml", ".yml"}:
            data = _read_yaml(path)
            if isinstance(data, dict):
                hints.update(data)
        elif path.suffix == ".py":
            hints.update(_read_config_py(path))
    return hints


def extract_variant_from_hints(hints: Dict[str, Any], family: str) -> Optional[str]:
    key_map = {
        "mambavision": "MAMBAVISION_VARIANT",
        "vmamba": "VMAMBA_VARIANT",
        "spatialmamba": "SPATIALMAMBA_VARIANT",
    }
    target = key_map.get(family)
    if not target:
        return None
    for key in (target, target.lower(), "variant", "backbone"):
        if key in hints and isinstance(hints[key], str):
            return hints[key].lower()
    return None


def extract_vmamba_weight_set(hints: Dict[str, Any]) -> Optional[str]:
    for key in ("VMAMBA_WEIGHT_SET", "vmamba_weight_set"):
        if key in hints and isinstance(hints[key], str):
            return hints[key]
    return None


def import_family_modules() -> Tuple[Any, Any]:
    import importlib

    importlib.invalidate_caches()
    config_mod = importlib.import_module("config")
    model_mod = importlib.import_module("model")
    if not hasattr(config_mod, "Config") or not hasattr(model_mod, "build_model"):
        raise RuntimeError("Missing Config/build_model in active family directory")
    return config_mod.Config, model_mod.build_model


def resolve_checkpoint(exp_dir: Path) -> Optional[Path]:
    ckpt = exp_dir / "checkpoints" / "best.pth"
    return ckpt if ckpt.exists() else None


def _extract_state_dict(ckpt: Any) -> Dict[str, Any]:
    if isinstance(ckpt, dict):
        for key in (
            "state_dict",
            "model_state_dict",
            "model_state",
            "model",
            "net",
            "network",
        ):
            if key in ckpt and isinstance(ckpt[key], dict):
                return ckpt[key]
        if all(isinstance(v, torch.Tensor) for v in ckpt.values()):
            return ckpt
    if isinstance(ckpt, dict):
        return ckpt
    raise ValueError("Unsupported checkpoint format")


def _normalize_state_dict(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    prefixes = ["module.", "model.", "net.", "backbone."]
    for prefix in prefixes:
        if state_dict and all(k.startswith(prefix) for k in state_dict.keys()):
            return {k[len(prefix):]: v for k, v in state_dict.items()}
    if any(k.startswith("module.") for k in state_dict.keys()):
        return {
            k[len("module."):] if k.startswith("module.") else k: v
            for k, v in state_dict.items()
        }
    return state_dict


def load_checkpoint(path: Path, model: torch.nn.Module) -> Tuple[int, int]:
    try:
        import argparse as _argparse

        torch.serialization.add_safe_globals([_argparse.Namespace])
    except Exception:
        pass
    ckpt = torch.load(str(path), map_location="cpu")
    state_dict = _extract_state_dict(ckpt)
    state_dict = _normalize_state_dict(state_dict)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    return len(missing), len(unexpected)


def _autocast_ctx(use_cuda: bool, use_amp: bool):
    return torch.cuda.amp.autocast(enabled=use_amp) if use_cuda else nullcontext()


def _run_iters(
    model: torch.nn.Module,
    dummy: torch.Tensor,
    iters: int,
    use_cuda: bool,
    use_amp: bool,
) -> None:
    with torch.no_grad():
        for _ in range(iters):
            with _autocast_ctx(use_cuda, use_amp):
                out = model(dummy)
            if isinstance(out, (list, tuple)):
                _ = out[0]
            elif isinstance(out, dict):
                _ = out.get("main") or out.get("out") or out.get("logits")
            out = None


def benchmark_model(
    model: torch.nn.Module,
    device: torch.device,
    img_size: int,
    batch_size: int,
    iters: int,
    warmup: int,
    use_amp: bool,
) -> Dict[str, float]:
    use_cuda = device.type == "cuda"

    if use_cuda:
        torch.cuda.empty_cache()
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)

    model.to(device)
    model.eval()

    if use_cuda:
        torch.cuda.synchronize(device)
        baseline_alloc = torch.cuda.memory_allocated(device)
        baseline_reserved = torch.cuda.memory_reserved(device)
    else:
        baseline_alloc = 0.0
        baseline_reserved = 0.0

    dtype = torch.float16 if (use_cuda and use_amp) else torch.float32
    dummy = torch.randn(batch_size, 3, img_size, img_size, device=device, dtype=dtype)

    if warmup > 0:
        _run_iters(model, dummy, warmup, use_cuda, use_amp)
        if use_cuda:
            torch.cuda.synchronize(device)

    if use_cuda:
        torch.cuda.reset_peak_memory_stats(device)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize(device)
        start.record()
    else:
        start = end = None

    t0 = time.perf_counter()
    _run_iters(model, dummy, iters, use_cuda, use_amp)
    if use_cuda:
        end.record()
        torch.cuda.synchronize(device)
    t1 = time.perf_counter()

    if use_cuda:
        elapsed_ms = start.elapsed_time(end) if start and end else 0.0
        total_seconds = elapsed_ms / 1000.0
    else:
        total_seconds = t1 - t0

    total_images = iters * batch_size
    fps = total_images / total_seconds if total_seconds > 0 else 0.0
    ms_per_img = (total_seconds * 1000.0) / total_images if total_images > 0 else 0.0

    if use_cuda:
        peak_alloc = torch.cuda.max_memory_allocated(device)
        peak_reserved = torch.cuda.max_memory_reserved(device)
    else:
        peak_alloc = 0.0
        peak_reserved = 0.0

    baseline_alloc_gb = baseline_alloc / (1024 ** 3)
    baseline_reserved_gb = baseline_reserved / (1024 ** 3)
    peak_alloc_gb = peak_alloc / (1024 ** 3)
    peak_reserved_gb = peak_reserved / (1024 ** 3)
    peak_activation_alloc_gb = max(0.0, peak_alloc_gb - baseline_alloc_gb)
    peak_activation_reserved_gb = max(0.0, peak_reserved_gb - baseline_reserved_gb)

    return {
        "fps": fps,
        "ms_per_img": ms_per_img,
        "baseline_alloc_gb": baseline_alloc_gb,
        "baseline_reserved_gb": baseline_reserved_gb,
        "peak_alloc_gb": peak_alloc_gb,
        "peak_reserved_gb": peak_reserved_gb,
        "peak_activation_alloc_gb": peak_activation_alloc_gb,
        "peak_activation_reserved_gb": peak_activation_reserved_gb,
    }


def main() -> int:
    args = parse_args()
    exp_dir = Path(args.exp_dir)
    if not exp_dir.exists():
        warn(f"Experiment directory not found: {exp_dir}")
        return 2

    ckpt_path = resolve_checkpoint(exp_dir)
    if not ckpt_path:
        warn(f"Missing checkpoint in {exp_dir}")
        return 2

    family, variant = detect_family_variant(exp_dir.name)
    if not family:
        warn(f"Unrecognized experiment name: {exp_dir.name}")
        return 2

    hints = read_config_hints(exp_dir)
    hinted_variant = extract_variant_from_hints(hints, family)
    if hinted_variant:
        variant = hinted_variant

    meta = FAMILY_CONFIG[family]
    if not variant or variant not in meta["variants"]:
        warn(f"Unsupported variant '{variant}' for {exp_dir.name}")
        return 2

    repo_root = Path(__file__).resolve().parents[1]
    family_dir = repo_root / meta["dir_name"]
    if not family_dir.exists():
        warn(f"Family directory not found: {family_dir}")
        return 2

    family_path = str(family_dir.resolve())
    if family_path in sys.path:
        sys.path.remove(family_path)
    sys.path.insert(0, family_path)

    try:
        Config, build_model = import_family_modules()
    except Exception as exc:
        warn(f"Failed to import {family} modules: {exc}")
        return 2

    weight_set = extract_vmamba_weight_set(hints) if family == "vmamba" else None
    cfg_kwargs = {meta["variant_field"]: variant, "OUTPUT_DIR": str(exp_dir)}
    if family == "vmamba" and weight_set:
        cfg_kwargs["VMAMBA_WEIGHT_SET"] = weight_set

    try:
        cfg = Config(**cfg_kwargs)
    except TypeError:
        cfg = Config()
        setattr(cfg, meta["variant_field"], variant)
        if hasattr(cfg, "OUTPUT_DIR"):
            setattr(cfg, "OUTPUT_DIR", str(exp_dir))
        if family == "vmamba" and weight_set and hasattr(cfg, "VMAMBA_WEIGHT_SET"):
            setattr(cfg, "VMAMBA_WEIGHT_SET", weight_set)
        if hasattr(cfg, "__post_init__"):
            cfg.__post_init__()

    if hasattr(cfg, "WEIGHTS_PATH"):
        setattr(cfg, "WEIGHTS_PATH", "")

    model = build_model(cfg)
    missing, unexpected = load_checkpoint(ckpt_path, model)
    if missing or unexpected:
        warn(f"{exp_dir.name}: load_state_dict missing={missing} unexpected={unexpected}")

    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            warn("CUDA requested but not available")
            return 2
        torch.cuda.set_device(device)
        torch.backends.cudnn.benchmark = True

    metrics = benchmark_model(
        model=model,
        device=device,
        img_size=args.img_size,
        batch_size=args.batch_size,
        iters=args.iters,
        warmup=args.warmup,
        use_amp=args.amp,
    )

    result = {
        "exp_name": exp_dir.name,
        "backbone": meta["backbone"],
        "variant": variant,
        "ckpt_path": str(ckpt_path),
        "img_size": args.img_size,
        "batch_size": args.batch_size,
        "amp": bool(args.amp),
        "iters": args.iters,
        "warmup": args.warmup,
        **metrics,
    }

    print(json.dumps(result))

    del model
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()
    gc.collect()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
