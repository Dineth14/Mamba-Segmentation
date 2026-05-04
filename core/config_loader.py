"""
core/config_loader.py
=====================
Loads a YAML config file and performs environment-variable substitution.

Syntax inside YAML values: ${ENV_VAR:default_value}
  - If ENV_VAR is set, its value is used.
  - If not set, default_value is used as a plain string.

Inheritance:
  A config file may begin with ``extends: relative/path/to/base.yaml``.
  The base is loaded first; the child config's values are deep-merged on top.

Usage:
    from core.config_loader import load_config
    cfg = load_config("configs/vmamba.yaml")
    # Optional CLI overrides (key=value pairs):
    cfg = load_config("configs/vmamba.yaml", overrides=["variant=base", "batch_size=4"])
"""

from __future__ import annotations

import os
import re
from copy import deepcopy
from typing import Any, Dict, List, Optional

import yaml

_ENV_PATTERN = re.compile(r"\$\{([^}:]+)(?::([^}]*))?\}")


def _substitute(value: Any) -> Any:
    """Recursively apply env-var substitution to strings in a config dict."""
    if isinstance(value, str):
        def _replace(m: re.Match) -> str:
            var_name, default = m.group(1), m.group(2) or ""
            return os.environ.get(var_name, default)
        return _ENV_PATTERN.sub(_replace, value)
    if isinstance(value, dict):
        return {k: _substitute(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_substitute(v) for v in value]
    return value


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Deep-merge *override* into a copy of *base*. Override wins on conflicts."""
    result = deepcopy(base)
    for key, val in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(val, dict):
            result[key] = _deep_merge(result[key], val)
        else:
            result[key] = deepcopy(val)
    return result


def _apply_override(cfg: Dict[str, Any], key: str, raw_value: str) -> None:
    """
    Set cfg[key] = raw_value, attempting to coerce the type to match the
    existing value where possible. Supports dot-notation for nested keys.
    """
    keys = key.split(".")
    target = cfg
    for k in keys[:-1]:
        if k not in target or not isinstance(target[k], dict):
            target[k] = {}
        target = target[k]

    leaf = keys[-1]
    existing = target.get(leaf)

    # Type coercion
    try:
        if isinstance(existing, bool):
            coerced: Any = raw_value.lower() in ("1", "true", "yes")
        elif isinstance(existing, int):
            coerced = int(raw_value)
        elif isinstance(existing, float):
            coerced = float(raw_value)
        elif isinstance(existing, list):
            coerced = yaml.safe_load(raw_value)
        else:
            coerced = raw_value
    except (ValueError, yaml.YAMLError):
        coerced = raw_value

    target[leaf] = coerced


def load_config(
    config_path: str,
    overrides: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Load a YAML config and return a plain dict.

    If the YAML contains ``extends: <path>``, the referenced file is loaded
    first and the current file is deep-merged on top.

    Args:
        config_path: Path to a YAML file.
        overrides:   Optional list of "key=value" strings to override after loading.

    Returns:
        Flat/nested dict with env-vars resolved and overrides applied.
    """
    config_path = os.path.abspath(config_path)
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Config not found: {config_path}")

    with open(config_path, "r") as fh:
        raw = yaml.safe_load(fh)

    if raw is None:
        raw = {}

    # Handle inheritance
    extends = raw.pop("extends", None)
    if extends:
        base_path = os.path.join(os.path.dirname(config_path), extends)
        base_cfg = load_config(base_path)  # recursive — supports chaining
        raw = _deep_merge(base_cfg, raw)

    cfg: Dict[str, Any] = _substitute(raw)

    # Apply CLI overrides
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"Override must be key=value, got: {item!r}")
        k, _, v = item.partition("=")
        _apply_override(cfg, k.strip(), v.strip())

    return cfg
