# True NSST Preprocessing (UrbanMamba)

This folder provides **offline** Non-Subsampled Shearlet Transform (NSST)
feature extraction using **PyShearLab** (ShearLab for Python).

Key points:
- **True NSST**, not an approximation (uses shearlet filter banks).
- **Non-subsampled** (shift-invariant) decomposition.
- **Offline preprocessing only** (not differentiable, no backprop).
- Outputs `.npy` files suitable for PyTorch datasets.

## Install

```
pip install pyshearlab opencv-python numpy tqdm
```

## Usage

```
python preprocess_nsst.py \
  --image_dir /path/to/images \
  --output_dir /path/to/nsst \
  --shear_levels 2 2 2 \
  --num_channels 27
```

## Output

- Per-image `.npy` files
- Shape: `(C, H, W)` with `C` defaulting to 27
- `float32` coefficients

## Notes

Channel selection keeps the low-pass band and the highest-energy directional
subbands to reach a fixed channel count.
