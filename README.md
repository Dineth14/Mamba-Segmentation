# Mamba-Segmentation

Semantic segmentation models built on Mamba-style backbones for remote sensing datasets.

## Projects

- `MambaVision/` - MambaVision backbone + LightUNetDecoder.
- `VMamba/` - VMamba backbone + LightUNetDecoder.
- `VisionMamba/` - VisionMamba backbone + LightUNetDecoder (plus domain training helpers).
- `spatial-mamba/` - Spatial-Mamba backbone (UrbanMamba RGB-only).

## Datasets

- Loveda (RGB): handled by `dataset.py` in each model folder.
- ISPRS Potsdam (ICPRS): handled by `MambaVision/dataset_isprs.py` and reused by VMamba and spatial-mamba for ICPRS runs.

Expected Potsdam structure:

```
DATA_ROOT/
  Images/
  Labels/
  splits/
    train.txt
    val.txt
    test.txt
```

## Training

Each model folder provides a `train.py` script. The train scripts import `config_icprs.py` by default.

To run Loveda instead, switch the import in the model's `train.py` to:

```
from config import Config
```

Then update `DATA_ROOT`, `OUTPUT_DIR`, and variant settings in the selected config file.

Example:

```
cd MambaVision
python train.py
```

## Results and Outputs

Training artifacts are written to the configured `OUTPUT_DIR` (checkpoints, logs, tensorboard, val_preds). These folders are gitignored by the root `.gitignore` so the repo can be pushed without results.
