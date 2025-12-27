"""
Offline NSST preprocessing script using PyShearLab.
"""

import argparse
import os
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
from tqdm import tqdm

from nsst_true import compute_true_nsst, select_nsst_channels


def _collect_images(image_dir: str) -> List[Path]:
    exts = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")
    paths = [p for p in Path(image_dir).rglob("*") if p.suffix.lower() in exts]
    return sorted(paths)


def _resolve_loveda_paths(
    data_root: str,
    split: str,
    area: str,
) -> Tuple[str, str]:
    split = split.capitalize()
    area = area.capitalize()
    image_dir = os.path.join(data_root, split, split, area, "images_png")
    output_dir = os.path.join(data_root, "NSST_27ch", split, area, "images_png")
    return image_dir, output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute true NSST features offline.")
    parser.add_argument("--image_dir", default=None, help="Input RGB image directory")
    parser.add_argument("--output_dir", default=None, help="Output directory for .npy files")
    parser.add_argument("--data_root", default="/storage2/ChangeDetection/Datasets/Loveda")
    parser.add_argument("--split", default="train", choices=["train", "val"])
    parser.add_argument("--area", default="urban", choices=["urban", "rural"])
    parser.add_argument("--shear_levels", nargs="+", type=int, default=[2, 2, 2])
    parser.add_argument("--num_channels", type=int, default=27)
    args = parser.parse_args()

    if args.image_dir is None or args.output_dir is None:
        image_dir, output_dir = _resolve_loveda_paths(args.data_root, args.split, args.area)
    else:
        image_dir, output_dir = args.image_dir, args.output_dir

    image_paths = _collect_images(image_dir)
    if not image_paths:
        raise ValueError(f"No images found in {image_dir}")

    os.makedirs(output_dir, exist_ok=True)

    sample_channels = None
    for img_path in tqdm(image_paths, desc="NSST"):
        bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if bgr is None:
            print(f"Warning: failed to read {img_path}, skipping")
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        coeffs = compute_true_nsst(rgb, shear_levels=tuple(args.shear_levels), normalize=True)
        coeffs = select_nsst_channels(coeffs, num_channels=args.num_channels)

        if sample_channels is None:
            sample_channels = coeffs.shape[0]
            print(f"NSST channels: {sample_channels}")

        if coeffs.shape[1:] != rgb.shape[:2]:
            raise ValueError(f"Shift invariance check failed for {img_path}")
        if not np.isfinite(coeffs).all():
            raise ValueError(f"NaN/Inf in coeffs for {img_path}")

        out_path = Path(output_dir) / (img_path.stem + ".npy")
        np.save(out_path, coeffs.astype(np.float32))

    print("NSST preprocessing complete.")


if __name__ == "__main__":
    main()
