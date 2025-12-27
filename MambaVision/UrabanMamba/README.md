# UrbanMamba (MambaVision)

Production-grade RGB-only semantic segmentation system for the LOVEDA dataset using a MambaVision encoder.

## Architecture Overview

```
RGB Image (3ch) ──► RGBEncoder (MambaVision: tiny/small/base) ──► LightUNetDecoder ──► Predictions
```

## Files

| File | Purpose |
|------|---------|
| `config.py` | Central configuration with all hyperparameters |
| `parts.py` | Building blocks: Conv/attention utilities |
| `encoders.py` | RGBEncoder (MambaVision) |
| `light_decoder.py` | Lightweight U-Net decoder (single main head) |
| `model.py` | NSSTMamba RGB-only assembly |
| `dataset.py` | LovedaDataset with albumentations for RGB + masks |
| `utils.py` | SegmentationEvaluator, PolynomialDecay, checkpoint utilities |
| `losses.py` | TriBraidLoss (Lovász + Focal + Boundary) with deep supervision |
| `train.py` | Iteration-based training loop with AMP |

## Key Features

- **RGB-only Architecture**: MambaVision for RGB features + lightweight U-Net decoder
- **MambaVision Variants**: tiny, tiny2, small, base, base_21k, large, large_21k, large2, large2_512_21k, large3_256_21k, large3_512_21k
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

MambaVision pretrained weights should be placed under the weights directory and selected via config:
```
/storage2/ChangeDetection/NSST-mamba/NSST_Mamba_v2/Vmamba_weights/
```

## License

MIT License
