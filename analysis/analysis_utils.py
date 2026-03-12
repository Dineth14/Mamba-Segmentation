"""
Shared analysis helpers for Mamba-Segmentation experiments.
"""

from __future__ import annotations

import importlib.util
import json
import logging
import os
import random
import time
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import sys

FAMILY_CONFIG: Dict[str, Dict[str, Any]] = {
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
    "visionmamba": {
        "dir_name": "VisionMamba",
        "variant_field": "VIM_VARIANT",
        "backbone": "VisionMamba",
        "variants": {"tiny", "small", "base"},
    },
    "spatialmamba": {
        "dir_name": "spatial-mamba",
        "variant_field": "SPATIALMAMBA_VARIANT",
        "backbone": "SpatialMamba",
        "variants": {"tiny", "small", "base"},
    },
    "cnn_deeplabv3p": {
        "dir_name": "CNN_DeepLabv3p",
        "variant_field": "BACKBONE_NAME",
        "backbone": "CNN_DeepLabv3p",
        "variants": {"resnet50"},
    },
    "transformer_unetformer": {
        "dir_name": "TransformerUNetFormer",
        "variant_field": "BACKBONE_NAME",
        "backbone": "TransformerUNetFormer",
        "variants": {"resnet18"},
    },
}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def default_experiments_root() -> Path:
    return repo_root() / "Comparison_Experiments"


def default_output_dir() -> Path:
    return repo_root() / "analysis_outputs"


def set_deterministic(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_analysis_logger(name: str, output_dir: Path) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_path = output_dir / f"{name.lower()}_{timestamp}.log"
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger


class ProgressBar:
    def __init__(self, total: int, prefix: str = "") -> None:
        self.total = max(1, total)
        self.prefix = prefix
        self.start = time.perf_counter()
        self.last_len = 0

    def _format(self, current: int, desc: str) -> str:
        elapsed = max(0.0, time.perf_counter() - self.start)
        rate = current / elapsed if elapsed > 0 else 0.0
        remaining = self.total - current
        eta = remaining / rate if rate > 0 else 0.0
        pct = 100.0 * current / self.total
        bar_len = 24
        filled = int(bar_len * pct / 100.0)
        bar = "#" * filled + "-" * (bar_len - filled)
        return (
            f"{self.prefix}[{bar}] {current}/{self.total} "
            f"{pct:5.1f}% | {desc} | ETA {eta:6.1f}s"
        )

    def update(self, current: int, desc: str = "") -> None:
        line = self._format(current, desc)
        pad = " " * max(0, self.last_len - len(line))
        self.last_len = len(line)
        print("\r" + line + pad, end="", file=sys.stderr, flush=True)

    def finish(self) -> None:
        self.update(self.total, "done")
        print("", file=sys.stderr)


def scan_experiments(root: Path, include_domain: bool) -> List[Path]:
    if not root.exists():
        raise FileNotFoundError(f"Experiments root not found: {root}")
    exp_dirs: List[Path] = []
    for entry in root.iterdir():
        if not entry.is_dir():
            continue
        name_lower = entry.name.lower()
        if not include_domain and ("urbantrain" in name_lower or "ruraltrain" in name_lower):
            continue
        exp_dirs.append(entry)
    return sorted(exp_dirs)


def parse_family_list(raw: str) -> List[str]:
    families = [f.strip().lower() for f in raw.split(",") if f.strip()]
    return [f for f in families if f in FAMILY_CONFIG]


def filter_experiments_by_family(exp_dirs: List[Path], families: List[str]) -> List[Path]:
    if not families:
        return exp_dirs
    keep: List[Path] = []
    for exp_dir in exp_dirs:
        family, _ = detect_family_variant(exp_dir.name)
        if family in families:
            keep.append(exp_dir)
    return keep


def _normalize_resnet_variant(token: Optional[str]) -> Optional[str]:
    if not token:
        return None
    token = token.lower()
    if token in {"r18", "resnet18", "resnet_18"}:
        return "resnet18"
    if token in {"r50", "resnet50", "resnet_50"}:
        return "resnet50"
    return token


def detect_family_variant(exp_name: str) -> Tuple[Optional[str], Optional[str]]:
    name = exp_name.lower()
    family = None
    prefix_tokens = 1
    if name.startswith("mambavision"):
        family = "mambavision"
    elif name.startswith("spatialmamba"):
        family = "spatialmamba"
    elif name.startswith("visionmamba"):
        family = "visionmamba"
    elif name.startswith("vmamba") or name.startswith("vmamb"):
        family = "vmamba"
    elif name.startswith("cnn_deeplabv3p"):
        family = "cnn_deeplabv3p"
        prefix_tokens = 2
    elif name.startswith("transformer_unetformer"):
        family = "transformer_unetformer"
        prefix_tokens = 2
    else:
        return None, None

    tokens = [t for t in name.split("_") if t not in ("urbantrain", "ruraltrain")]
    while tokens and tokens[-1].isdigit():
        tokens.pop()
    variant = tokens[prefix_tokens] if len(tokens) > prefix_tokens else None
    if family in {"cnn_deeplabv3p", "transformer_unetformer"}:
        variant = _normalize_resnet_variant(variant)
    return family, variant


def resolve_checkpoint(exp_dir: Path) -> Optional[Path]:
    ckpt = exp_dir / "checkpoints" / "best.pth"
    return ckpt if ckpt.exists() else None


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _read_yaml(path: Path) -> Optional[Dict[str, Any]]:
    try:
        import yaml  # type: ignore
    except Exception:
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except Exception:
        return None


def _read_config_py(path: Path) -> Dict[str, str]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    keys = [
        "MAMBAVISION_VARIANT",
        "VMAMBA_VARIANT",
        "SPATIALMAMBA_VARIANT",
        "VIM_VARIANT",
        "VMAMBA_WEIGHT_SET",
        "BACKBONE_NAME",
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
        "visionmamba": "VIM_VARIANT",
        "cnn_deeplabv3p": "BACKBONE_NAME",
        "transformer_unetformer": "BACKBONE_NAME",
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


def infer_vmamba_weight_set_from_logs(exp_dir: Path) -> Optional[str]:
    candidates = list(exp_dir.glob("train_*.log"))
    for log_path in candidates:
        try:
            text = log_path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        for line in text.splitlines():
            if "VMamba weights:" in line:
                lower = line.lower()
                if "imagenet-1k" in lower:
                    return "imagenet1k"
                if "vanilla-vmamba" in lower:
                    return "vanilla_ade20k"
                if "ade20k" in lower:
                    return "ade20k"
    return None


def clear_modules_from_dir(dir_path: Path) -> None:
    dir_path = dir_path.resolve()
    dir_prefix = f"{dir_path}{os.sep}"
    for name, mod in list(sys.modules.items()):
        mod_path = getattr(mod, "__file__", None)
        if not mod_path:
            continue
        try:
            if str(Path(mod_path).resolve()).startswith(dir_prefix):
                del sys.modules[name]
        except Exception:
            continue


def activate_family_dir(
    family_dir: Path, previous_family_dir: Optional[Path]
) -> Optional[Path]:
    if previous_family_dir is not None:
        clear_modules_from_dir(previous_family_dir)
        prev_path = str(previous_family_dir.resolve())
        if prev_path in sys.path:
            sys.path.remove(prev_path)

    family_path = str(family_dir.resolve())
    if family_path in sys.path:
        sys.path.remove(family_path)
    sys.path.insert(0, family_path)
    return family_dir


def cleanup_family_dir(previous_family_dir: Optional[Path]) -> None:
    if previous_family_dir is None:
        return
    clear_modules_from_dir(previous_family_dir)
    prev_path = str(previous_family_dir.resolve())
    if prev_path in sys.path:
        sys.path.remove(prev_path)


def _load_module_from_path(name: str, path: Path, register: bool = True) -> Any:
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    if register:
        sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def import_family_modules(family_dir: Path) -> Tuple[Any, Any, Any, Any]:
    family_dir = family_dir.resolve()
    family_path = str(family_dir)
    if family_path not in sys.path:
        sys.path.insert(0, family_path)

    for name in ("config", "dataset", "model", "utils"):
        sys.modules.pop(name, None)

    config_mod = _load_module_from_path("config", family_dir / "config.py")
    dataset_mod = _load_module_from_path("dataset", family_dir / "dataset.py")
    model_mod = _load_module_from_path("model", family_dir / "model.py")
    utils_name = f"analysis_utils_{family_dir.name.replace('-', '_')}"
    utils_mod = _load_module_from_path(utils_name, family_dir / "utils.py", register=False)
    if not hasattr(config_mod, "Config") or not hasattr(model_mod, "build_model"):
        raise RuntimeError("Missing Config/build_model in active family directory")
    return config_mod.Config, model_mod.build_model, dataset_mod, utils_mod


def load_model_with_checkpoint(
    family: str,
    cfg: Any,
    build_model: Any,
    utils_mod: Any,
    ckpt_path: Path,
    device: torch.device,
    meta: Dict[str, Any],
    variant: str,
    data_root: Optional[str],
    weight_set: Optional[str],
    use_pretrained: bool,
    logger: Any = None,
) -> Tuple[Optional[Any], Optional[torch.nn.Module]]:
    def _log(level: str, msg: str) -> None:
        if logger is None:
            return
        getattr(logger, level, logger.info)(msg)

    def _try_load(model: torch.nn.Module, strict: bool) -> bool:
        try:
            utils_mod.load_checkpoint(str(ckpt_path), model, device=device, strict=strict)
            return True
        except Exception as exc:
            _log("warning", f"Checkpoint load failed ({'strict' if strict else 'non-strict'}): {exc}")
            return False

    def _extract_state_dict(ckpt_obj: Any) -> Optional[Dict[str, Any]]:
        if isinstance(ckpt_obj, dict):
            for key in (
                "model_state_dict",
                "state_dict",
                "model",
                "net",
                "network",
            ):
                if key in ckpt_obj and isinstance(ckpt_obj[key], dict):
                    return ckpt_obj[key]
            if all(isinstance(v, torch.Tensor) for v in ckpt_obj.values()):
                return ckpt_obj
        return None

    def _sanitize_state_dict(state_dict: Dict[str, Any]) -> Tuple[Dict[str, Any], int]:
        filtered = {
            k: v
            for k, v in state_dict.items()
            if "total_ops" not in k and "total_params" not in k
        }
        return filtered, len(state_dict) - len(filtered)

    def _try_load_direct(model: torch.nn.Module) -> bool:
        try:
            ckpt_obj = torch.load(str(ckpt_path), map_location="cpu")
        except Exception as exc:
            _log("warning", f"Direct checkpoint read failed: {exc}")
            return False
        state_dict = _extract_state_dict(ckpt_obj)
        if state_dict is None:
            _log("warning", "Unsupported checkpoint format for direct load.")
            return False
        state_dict, dropped = _sanitize_state_dict(state_dict)
        if dropped and logger is not None:
            logger.info("Filtered %d profiling keys from checkpoint.", dropped)
        try:
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            if logger is not None:
                logger.info(
                    "Loaded checkpoint with strict=False (missing=%d, unexpected=%d).",
                    len(missing),
                    len(unexpected),
                )
            return True
        except Exception as exc:
            _log("warning", f"Direct checkpoint load failed: {exc}")
            return False

    if logger is not None:
        logger.info(
            "Loading checkpoint %s (family=%s, variant=%s, weight_set=%s, pretrained=%s)",
            ckpt_path.name,
            family,
            variant,
            weight_set or "n/a",
            "on" if use_pretrained else "off",
        )
        if hasattr(cfg, "WEIGHTS_PATH") and getattr(cfg, "WEIGHTS_PATH"):
            logger.info("Pretrained weights path: %s", getattr(cfg, "WEIGHTS_PATH"))

    model = build_model(cfg).to(device)
    if _try_load_direct(model) or _try_load(model, strict=True) or _try_load(model, strict=False):
        return cfg, model

    if family != "vmamba":
        return None, None

    alt_sets = ["imagenet1k", "ade20k", "vanilla_ade20k"]
    for alt in alt_sets:
        if alt == weight_set:
            continue
        cfg_alt = build_config(
            Config=type(cfg),
            meta=meta,
            exp_dir=Path(cfg.OUTPUT_DIR),
            variant=variant,
            weight_set=alt,
            data_root=data_root,
        )
        if not use_pretrained:
            disable_pretrained(cfg_alt)
        disable_pretrained(cfg_alt)
        model_alt = build_model(cfg_alt).to(device)
        if _try_load(model_alt, strict=True) or _try_load(model_alt, strict=False):
            _log("warning", f"Using VMAMBA_WEIGHT_SET='{alt}' for checkpoint {ckpt_path.name}")
            return cfg_alt, model_alt
        del model_alt
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return None, None


def build_config(
    Config: Any,
    meta: Dict[str, Any],
    exp_dir: Path,
    variant: str,
    weight_set: Optional[str] = None,
    data_root: Optional[str] = None,
) -> Any:
    cfg_kwargs: Dict[str, Any] = {meta["variant_field"]: variant, "OUTPUT_DIR": str(exp_dir)}
    if data_root:
        cfg_kwargs["DATA_ROOT"] = data_root
    if weight_set:
        cfg_kwargs["VMAMBA_WEIGHT_SET"] = weight_set
    try:
        cfg = Config(**cfg_kwargs)
    except TypeError:
        cfg = Config()
        setattr(cfg, meta["variant_field"], variant)
        if hasattr(cfg, "OUTPUT_DIR"):
            setattr(cfg, "OUTPUT_DIR", str(exp_dir))
        if data_root and hasattr(cfg, "DATA_ROOT"):
            setattr(cfg, "DATA_ROOT", data_root)
        if weight_set and hasattr(cfg, "VMAMBA_WEIGHT_SET"):
            setattr(cfg, "VMAMBA_WEIGHT_SET", weight_set)
        if hasattr(cfg, "__post_init__"):
            cfg.__post_init__()
    return cfg


def disable_pretrained(cfg: Any) -> None:
    if hasattr(cfg, "WEIGHTS_PATH"):
        setattr(cfg, "WEIGHTS_PATH", "")


def resolve_val_dirs(domain: str, split: str) -> Tuple[List[str], List[str]]:
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


def build_val_loader(
    cfg: Any,
    dataset_mod: Any,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    domain: Optional[str] = None,
    split: str = "val",
):
    dataset_name = getattr(cfg, "DATASET", "loveda").lower()
    if dataset_name == "loveda":
        if domain:
            rel_img, rel_mask = resolve_val_dirs(domain, split)
            img_dirs = [cfg.get_full_path(p) for p in rel_img]
            mask_dirs = [cfg.get_full_path(p) for p in rel_mask]
        else:
            img_dirs, mask_dirs = cfg.get_val_paths()
        dataset = dataset_mod.LovedaDataset(
            img_dirs=img_dirs,
            mask_dirs=mask_dirs,
            crop_size=cfg.CROP_SIZE,
            is_train=False,
            rgb_mean=cfg.RGB_MEAN,
            rgb_std=cfg.RGB_STD,
            ignore_index=cfg.IGNORE_INDEX,
        )
    elif dataset_name == "icprs":
        img_dirs, mask_dirs = cfg.get_val_paths()
        dataset = dataset_mod.ICPRSDataset(
            img_dirs=img_dirs,
            mask_dirs=mask_dirs,
            crop_size=cfg.CROP_SIZE,
            is_train=False,
            rgb_mean=cfg.RGB_MEAN,
            rgb_std=cfg.RGB_STD,
            ignore_index=cfg.IGNORE_INDEX,
        )
    else:
        raise ValueError(f"Unsupported dataset for analysis: {dataset_name}")

    cpu_affinity = _get_cpu_affinity_for_device(device)
    _set_process_affinity(cpu_affinity)

    worker_init_fn = None
    if cpu_affinity:
        def _init_worker(_: int) -> None:
            _set_process_affinity(cpu_affinity)

        worker_init_fn = _init_worker

    prefetch_factor = getattr(cfg, "PREFETCH_FACTOR", 2)
    persistent_workers = getattr(cfg, "PERSISTENT_WORKERS", num_workers > 0)
    pin_memory = getattr(cfg, "PIN_MEMORY", device.type == "cuda")
    loader_kwargs: Dict[str, Any] = {}
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = prefetch_factor
        loader_kwargs["persistent_workers"] = persistent_workers

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=worker_init_fn,
        **loader_kwargs,
    )


def select_main_output(outputs: Any) -> torch.Tensor:
    if isinstance(outputs, dict):
        return outputs.get("main") or outputs.get("out") or outputs.get("logits")
    if isinstance(outputs, (list, tuple)):
        return outputs[0]
    return outputs


def _parse_cpu_list(text: str) -> List[int]:
    cpus: List[int] = []
    for part in text.strip().split(","):
        if not part:
            continue
        if "-" in part:
            start_s, end_s = part.split("-", 1)
            try:
                start = int(start_s)
                end = int(end_s)
            except ValueError:
                continue
            cpus.extend(range(start, end + 1))
        else:
            try:
                cpus.append(int(part))
            except ValueError:
                continue
    return sorted(set(cpus))


def _read_sysfs_text(path: Path) -> Optional[str]:
    try:
        return path.read_text(encoding="utf-8").strip()
    except Exception:
        return None


def _get_cpu_affinity_for_device(device: torch.device) -> Optional[List[int]]:
    if device.type != "cuda" or not torch.cuda.is_available():
        return None
    idx = device.index if device.index is not None else torch.cuda.current_device()
    try:
        props = torch.cuda.get_device_properties(idx)
    except Exception:
        return None
    pci_bus_id = getattr(props, "pci_bus_id", None)
    if not pci_bus_id:
        return None
    sysfs_dir = Path("/sys/bus/pci/devices") / pci_bus_id
    cpulist = _read_sysfs_text(sysfs_dir / "local_cpulist")
    if cpulist:
        cpus = _parse_cpu_list(cpulist)
        return cpus or None
    cpus_hex = _read_sysfs_text(sysfs_dir / "local_cpus")
    if cpus_hex:
        try:
            mask = int(cpus_hex.replace(",", ""), 16)
            cpus = [i for i in range(mask.bit_length()) if mask & (1 << i)]
            return cpus or None
        except ValueError:
            pass
    numa_node = _read_sysfs_text(sysfs_dir / "numa_node")
    if numa_node and numa_node.isdigit():
        node = int(numa_node)
        if node >= 0:
            node_path = Path("/sys/devices/system/node") / f"node{node}" / "cpulist"
            cpulist = _read_sysfs_text(node_path)
            if cpulist:
                cpus = _parse_cpu_list(cpulist)
                return cpus or None
    return None


def _set_process_affinity(cpus: Optional[List[int]]) -> None:
    if not cpus:
        return
    try:
        os.sched_setaffinity(0, cpus)
    except Exception:
        return
