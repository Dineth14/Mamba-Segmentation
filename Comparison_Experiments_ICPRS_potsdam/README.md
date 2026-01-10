# Comparison Experiments - ISPRS Potsdam Dataset

This directory contains experimental results for various Mamba-based segmentation models trained on the **ISPRS Potsdam** dataset.

## Dataset
- **Dataset:** ISPRS 2D Semantic Labeling Contest - Potsdam
- **Task:** Urban semantic segmentation
- **Modality:** RGB + Infrared (IRRG) remote sensing imagery
- **Resolution:** 512×512 patches (downsampled from 6000×6000 tiles)
- **Classes:** 6 classes (Impervious surfaces, Building, Low vegetation, Tree, Car, Clutter/background)

## Models

### MambaVision Models
- **mambavision_tiny_512** - Tiny MambaVision variant
- **mambavision_tiny_32** - Tiny MambaVision with 32×32 patches
- **mambavision_tiny2_512** - Alternative tiny configuration
- **mambavision_small_512** - Small MambaVision variant
- **mambavision_base_512** - Base MambaVision variant
- **mambavision_large_512** - Large MambaVision variant
- **mambavision_large2_512** - Alternative large configuration

### VMamba Models
- **vmamba_tiny_512** - VMamba tiny variant
- **vmamba_small_512** - VMamba small variant
- **vmamba_base_512** - VMamba base variant
- **vmamba_gpu** - VMamba GPU-optimized configuration

### SpatialMamba Models
- **spatialmamba_tiny_512** - SpatialMamba tiny variant
- **spatialmamba_small_512** - SpatialMamba small variant
- **spatialmamba_base_512** - SpatialMamba base variant
- **spatialmamba_gpu** - SpatialMamba GPU-optimized configuration

## Directory Structure

Each experiment directory contains:
- `checkpoints/` - Saved model weights (gitignored)
- `logs/` - Training logs (gitignored)
- `tensorboard/` - TensorBoard event files (gitignored)
- `val_preds/` - Validation predictions (gitignored)
- `*.log` - Training log files (gitignored)

## Dataset Structure

Expected Potsdam dataset structure:
```
DATA_ROOT/
  Images/           # RGB-IR tiles (top_potsdam_X_Y_RGBIR.tif)
  Labels/           # Ground truth labels (top_potsdam_X_Y_label.tif)
  splits/
    train.txt       # Training tile IDs
    val.txt         # Validation tile IDs
    test.txt        # Test tile IDs (optional)
```

## Results

Results are stored locally in each experiment directory but are excluded from git tracking. Key metrics tracked:
- **mIoU (mean Intersection over Union)** - Primary metric
- **Overall Accuracy**
- **Per-class IoU**
- **F1 Score**

## Usage

To reproduce or continue experiments:
1. Navigate to the model directory (e.g., `MambaVision/`, `VMamba/`, `spatial-mamba/`)
2. Use `config_icprs.py` configuration
3. Update `DATA_ROOT` to point to your Potsdam dataset
4. Update `OUTPUT_DIR` to the desired experiment directory
5. Run `python train.py`

Example:
```bash
cd ../MambaVision
# Edit config_icprs.py to set:
# OUTPUT_DIR = '../Comparison_Experiments_ICPRS_potsdam/mambavision_base_512'
python train.py
```

## Notes

- ISPRS Potsdam experiments use IRRG (Infrared-Red-Green) or RGB modalities
- Image resolution is standardized to 512×512 for computational efficiency
- Models can be compared across different architectural variants (tiny/small/base/large)
- GPU-optimized variants are tuned for specific hardware configurations
