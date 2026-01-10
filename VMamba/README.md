# VMamba (UrbanMamba RGB)

RGB-only semantic segmentation model using a VMamba backbone and a lightweight U-Net decoder.

## Architecture

```
RGB Image (3ch) -> VMamba Encoder -> LightUNetDecoder -> Predictions
```

## Key Files

- `config.py` - Loveda configuration (RGB dataset).
- `config_icprs.py` - ISPRS Potsdam configuration.
- `dataset.py` - Loveda dataset loader.
- `encoders.py` - VMamba RGB encoder.
- `light_decoder.py` - Light U-Net decoder.
- `model.py` - Full model assembly.
- `train.py` - Iteration-based training loop.

## Training

The training script imports `config_icprs.py` by default. To use Loveda instead, swap the import in `train.py` to:

```
from config import Config
```

Then update `DATA_ROOT`, `OUTPUT_DIR`, and `VMAMBA_VARIANT` in the selected config file.

Run training:

```
# VMamba Wrapper

This directory contains the VMamba-based semantic segmentation implementation for remote sensing datasets.

## Structure

- **VMamba/** - Submodule containing the official VMamba implementation
	- Source: https://github.com/MzeroMiko/VMamba
	- Used as the backbone encoder
- **train.py** - Main training script
- **config.py** - Configuration for LoveDA dataset
- **config_icprs.py** - Configuration for ISPRS Potsdam dataset
- **dataset.py** - LoveDA dataset loader
- **dataset_isprs.py** - ISPRS Potsdam dataset loader (reused from MambaVision)
- **model.py** - Integration of VMamba backbone with LightUNetDecoder
- **encoders.py** - Encoder wrapper for VMamba
- **light_decoder.py** - Lightweight U-Net decoder
- **losses.py** - Loss functions
- **utils.py** - Training utilities

## Architecture

```
RGB Image (3ch) -> VMamba Encoder -> LightUNetDecoder -> Predictions
```

VMamba (Visual State Space Model) features:
- **2D-Selective Scan (SS2D)** for spatial data processing
- **State Space Models (SSM)** for efficient sequential modeling
- **Cross-Scan mechanism** for bidirectional spatial information flow
- **Linear complexity** with respect to image size

## Model Variants

Available variants:
- **tiny** - Smallest model
- **small** - Small model
- **base** - Base model

## Training

### LoveDA Dataset
```bash
python train.py
```

### ISPRS Potsdam Dataset
```bash
# Edit train.py to import from config_icprs
python train.py
```

### Configuration

Key parameters:
```python
VARIANT = 'base'  # 'tiny', 'small', 'base'
DATA_ROOT = '/path/to/dataset'
OUTPUT_DIR = '../Comparison_Experiments/vmamba_base_512'
BATCH_SIZE = 8
LEARNING_RATE = 1e-4
```

## Output

Training outputs saved to `OUTPUT_DIR`:
- `checkpoints/` - Model weights (gitignored)
- `logs/` - Training logs (gitignored)
- `tensorboard/` - Training metrics (gitignored)
- `val_preds/` - Validation predictions (gitignored)

## Submodule

The **VMamba/** subdirectory is a git submodule pointing to the official implementation.

To update:
```bash
cd VMamba
git pull origin main
cd ..
git add VMamba
```

## Performance Notes

VMamba offers:
- Lower memory footprint than Transformers
- Faster inference speed
- Good performance on dense prediction tasks
- Efficient for high-resolution imagery

## Citation

```
@article{liu2024vmamba,
	title={VMamba: Visual State Space Model},
	author={Liu, Yue and Tian, Yunjie and Zhao, Yuzhong and others},
	journal={arXiv preprint arXiv:2401.10166},
	year={2024}
}
```
