"""
True NSST (Non-Subsampled Shearlet Transform) preprocessing utilities.
Uses PyShearLab for mathematically faithful shearlet coefficients.
"""

from typing import Iterable, Tuple

import numpy as np
import cv2
import pyshearlab as psl


def compute_true_nsst(
    rgb: np.ndarray,
    shear_levels: Iterable[int] = (2, 2, 2),
    normalize: bool = True,
) -> np.ndarray:
    """
    Compute true NSST coefficients using ShearLab (PyShearLab).

    Args:
        rgb: Input RGB image, shape (H, W, 3), dtype uint8 or float32.
        shear_levels: Tuple/list of shear levels per scale.
        normalize: If True, normalize input to [0, 1].

    Returns:
        coeffs: NSST coefficients, shape (C, H, W), dtype float32.
    """
    if rgb.dtype != np.float32:
        rgb = rgb.astype(np.float32)
    if normalize and rgb.max() > 1.5:
        rgb = rgb / 255.0

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape

    shearlet_system = psl.SLgetShearletSystem2D(
        rows=h,
        cols=w,
        scales=len(tuple(shear_levels)),
        shearLevels=list(shear_levels),
        full=True,
    )

    coeffs = psl.SLsheardec2D(gray, shearlet_system)

    # PyShearLab returns a list/array of coefficient maps
    coeffs = np.stack(coeffs, axis=0).astype(np.float32)

    # Safety checks
    if coeffs.shape[1:] != (h, w):
        raise ValueError(f"NSST coeffs shape mismatch: {coeffs.shape} vs {(h, w)}")
    if not np.isfinite(coeffs).all():
        raise ValueError("NSST coeffs contain NaN/Inf values.")

    return coeffs


def select_nsst_channels(coeffs: np.ndarray, num_channels: int = 27) -> np.ndarray:
    """
    Select a fixed number of NSST channels.

    Strategy:
      - Always keep low-pass (assumed index 0).
      - Select remaining channels by highest mean energy.
    """
    if coeffs.ndim != 3:
        raise ValueError(f"Expected coeffs shape (C, H, W), got {coeffs.shape}")
    c, _, _ = coeffs.shape
    if num_channels >= c:
        return coeffs

    energies = np.mean(np.abs(coeffs), axis=(1, 2))
    # Always keep channel 0 (low-pass)
    keep = [0]
    remaining = np.argsort(-energies[1:]) + 1
    keep.extend(remaining[: max(0, num_channels - 1)].tolist())
    keep = sorted(keep)
    return coeffs[keep].astype(np.float32)


__all__ = ["compute_true_nsst", "select_nsst_channels"]
