# UrbanMamba

Production-grade RGB-only semantic segmentation system for the LOVEDA dataset.

## Architecture Overview

```
RGB Image (3ch) ──► RGBEncoder (Spatial-Mamba: tiny/small/base) ──► LightUNetDecoder ──► Predictions
```

## Files

| File | Purpose |
|------|---------|
| `config.py` | Central configuration with all hyperparameters |
| `parts.py` | Building blocks: Conv/attention utilities |
| `encoders.py` | RGBEncoder (Spatial-Mamba) |
| `light_decoder.py` | Lightweight U-Net decoder with deep supervision outputs |
| `model.py` | NSSTMamba RGB-only assembly |
| `dataset.py` | LovedaDataset with albumentations for RGB + masks |
| `utils.py` | SegmentationEvaluator, PolynomialDecay, checkpoint utilities |
| `losses.py` | TriBraidLoss (Lovász + Focal + Boundary) with deep supervision |
| `train.py` | Iteration-based training loop with AMP |

## Key Features

- **RGB-only Architecture**: Spatial-Mamba for RGB features + lightweight U-Net decoder
- **Spatial-Mamba Variants**: Select between tiny, small, and base via config
- **TriBraidLoss**: Lovász-Softmax + Focal Loss + Boundary Loss
- **Loss**: Lovasz + Focal + Boundary (single main output)
- **Differential Learning Rates**: backbone=6e-5, head=3e-4
- **Mixed Precision Training**: AMP for Tensor Core utilization

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Batch Size | 8 |
| Crop Size | 640×640 |
| Max Iterations | 10,000 |
| Validation Interval | 2,000 |
| Optimizer | AdamW |
| LR (Backbone) | 6e-5 |
| LR (Head) | 3e-4 |
| Scheduler | PolynomialDecay (power=0.9) |
| Weight Decay | 0.01 |

## Usage

### Training

```bash
python train.py
```

Configuration is controlled via `UrbanMamba/config.py`:
- `GPU_ID` for device selection
- `OUTPUT_DIR` for logs/checkpoints/tensorboard
- `RESUME_PATH` to resume training

### Data Directory Structure

```
DATA_ROOT/
├── Train/Train/
│   ├── Urban/images_png
│   ├── Urban/masks_png
│   ├── Rural/images_png
│   └── Rural/masks_png
└── Val/Val/
    ├── Urban/images_png
    ├── Urban/masks_png
    ├── Rural/images_png
    └── Rural/masks_png
```

## Hardware Requirements

- **GPU**: NVIDIA Quadro GV100 (32GB HBM2) or equivalent
- **VRAM**: ~24GB with batch_size=12, crop_size=640
- **Disk**: ~50GB for LOVEDA



## Pretrained Weights

Spatial-Mamba pretrained weights should be placed under the weights directory and selected via config:
```
/storage2/ChangeDetection/NSST-mamba/spatial-mamba/Spatial-Mamba/weights/
```

## Results

Outputs are written to the configured `OUTPUT_DIR` (checkpoints, logs, tensorboard, val_preds). These folders are gitignored by the root `.gitignore`.

## License

MIT License
