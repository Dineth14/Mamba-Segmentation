# MATLAB NSST Preprocessing (UrbanMamba)

This pipeline uses the **official MATLAB NSST toolbox** to compute **true**
Non-Subsampled Shearlet Transform coefficients. Python-only NSST libraries
are not available on PyPI, so MATLAB is required for mathematically exact NSST.

Key points:
- **True NSST** via `nsst_dec2`
- **Non-subsampled** (shift-invariant)
- **Offline preprocessing only** (not differentiable)
- Outputs `.mat` then converts to `.npy` for PyTorch

## Requirements

- MATLAB with NSST toolbox on path
- Python with `numpy`, `scipy`, `tqdm`

## MATLAB batch processing

From `UrbanMamba/tools/matlab_nsst`:

```
./run_nsst_batch.sh /path/to/Loveda /path/to/output_mat
```

This produces:
- `<output_mat>/Train/Urban/images_png`
- `<output_mat>/Train/Rural/images_png`
- `<output_mat>/Val/Urban/images_png`
- `<output_mat>/Val/Rural/images_png`

## Convert .mat to .npy

```
python convert_nsst_mat_to_npy.py \
  --mat_dir /path/to/output_mat \
  --npy_dir /path/to/output_npy
```

Outputs `.npy` with shape `(C, H, W)` and dtype `float32`.

## Notes

This pipeline is **research-grade** and **reviewer-safe** because it uses the
official NSST implementation rather than approximations.
