"""
Convert MATLAB NSST .mat files to NumPy .npy for PyTorch datasets.
"""

import argparse
from pathlib import Path

import numpy as np
import scipy.io as sio
from tqdm import tqdm


def convert_mat_to_npy(mat_path: Path, out_path: Path) -> None:
    data = sio.loadmat(str(mat_path))
    if "coeffs" not in data:
        raise ValueError(f"Missing 'coeffs' in {mat_path}")
    coeffs = data["coeffs"].astype(np.float32)
    if coeffs.ndim != 3:
        raise ValueError(f"Expected coeffs shape (C,H,W) in {mat_path}, got {coeffs.shape}")
    if not np.isfinite(coeffs).all():
        raise ValueError(f"NaN/Inf in coeffs for {mat_path}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, coeffs)


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert NSST .mat to .npy")
    parser.add_argument("--mat_dir", required=True, help="Input directory with .mat files")
    parser.add_argument("--npy_dir", required=True, help="Output directory for .npy files")
    args = parser.parse_args()

    mat_dir = Path(args.mat_dir)
    npy_dir = Path(args.npy_dir)
    mats = sorted(mat_dir.rglob("*.mat"))
    if not mats:
        raise ValueError(f"No .mat files found under {mat_dir}")

    for m in tqdm(mats, desc="Converting"):
        rel = m.relative_to(mat_dir)
        out = npy_dir / rel.with_suffix(".npy")
        convert_mat_to_npy(m, out)

    print(f"Converted {len(mats)} files to {npy_dir}")


if __name__ == "__main__":
    main()
