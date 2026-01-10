# MambaVision (UrbanMamba RGB)

Production-grade RGB-only semantic segmentation model using a MambaVision backbone and a lightweight U-Net decoder.

## Architecture

```
RGB Image (3ch) -> MambaVision Encoder -> LightUNetDecoder -> Predictions
```

## Key Files

- `config.py` - Loveda configuration (RGB dataset).
- `config_icprs.py` - ISPRS Potsdam configuration.
- `dataset.py` - Loveda dataset loader.
- `dataset_isprs.py` - ISPRS Potsdam dataset loader.
- `encoders.py` - MambaVision RGB encoder.
- `light_decoder.py` - Light U-Net decoder.
- `model.py` - Full model assembly.
- `train.py` - Iteration-based training loop.

## Training

The training script imports `config_icprs.py` by default. To use Loveda instead, swap the import in `train.py` to:

```
from config import Config
```

Then adjust `DATA_ROOT`, `OUTPUT_DIR`, and `MAMBAVISION_VARIANT` in the chosen config file.

Run training:

```
# MambaVision Wrapper

This directory contains the MambaVision-based semantic segmentation implementation for remote sensing datasets.

## Structure

- **MambaVision/** - Submodule containing the official MambaVision implementation from NVIDIA
	- Source: https://github.com/NVlabs/MambaVision
	- Used as the backbone encoder
- **train.py** - Main training script
- **config.py** - Configuration for LoveDA dataset
- **config_icprs.py** - Configuration for ISPRS Potsdam dataset
- **dataset.py** - LoveDA dataset loader
- **dataset_isprs.py** - ISPRS Potsdam dataset loader
- **model.py** - Integration of MambaVision backbone with LightUNetDecoder
- **encoders.py** - Encoder wrapper for MambaVision
- **light_decoder.py** - Lightweight U-Net decoder
- **losses.py** - Loss functions (CrossEntropy, Dice, Focal)
- **parts.py** - Model building blocks
- **utils.py** - Training utilities

## Architecture

```
RGB Image (3ch) -> MambaVision Encoder -> LightUNetDecoder -> Predictions
```

MambaVision is a hybrid Mamba-Transformer architecture that combines:
- **Mamba layers** for efficient sequence modeling
- **Self-attention blocks** for long-range dependencies
- **Hierarchical feature extraction** at multiple scales

## Model Variants

Available variants (configured in `config.py`):
- **tiny** - Smallest model (~5M params)
- **tiny2** - Alternative tiny configuration
- **small** - Small model (~25M params)
- **base** - Base model (~50M params)
- **large** - Large model (~100M params)
- **large2** - Alternative large configuration

## Training

### LoveDA Dataset
```bash
python train.py  # Uses config.py by default
```

### ISPRS Potsdam Dataset
```bash
# Edit train.py to import from config_icprs
python train.py
```

### Configuration

Key parameters in config files:
```python
DATA_ROOT = '/path/to/dataset'
OUTPUT_DIR = '../Comparison_Experiments/mambavision_base_512'
VARIANT = 'base'  # or 'tiny', 'small', 'large'
BATCH_SIZE = 8
LEARNING_RATE = 1e-4
```

## Output

Results are saved to `OUTPUT_DIR`:
- `checkpoints/` - Model weights (gitignored)
- `logs/` - Training logs (gitignored)
- `tensorboard/` - TensorBoard events (gitignored)
- `val_preds/` - Validation predictions (gitignored)

## Submodule

The **MambaVision/** subdirectory is a git submodule pointing to the official NVIDIA implementation.

To update:
```bash
cd MambaVision
git pull origin main
cd ..
git add MambaVision
```

## Citation

```
@article{hatamizadeh2024mambavision,
	title={MambaVision: A Hybrid Mamba-Transformer Vision Backbone},
	author={Hatamizadeh, Ali and Kautz, Jan},
	journal={arXiv preprint},
	year={2024}
}
```
