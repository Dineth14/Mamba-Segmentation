# VisionMamba (UrbanMamba RGB)

RGB-only semantic segmentation model using a VisionMamba backbone and a lightweight U-Net decoder.

## Architecture

```
RGB Image (3ch) -> VisionMamba Encoder -> LightUNetDecoder -> Predictions
```

## Key Files

- `config.py` - Loveda configuration (RGB dataset).
- `config_icprs.py` - ISPRS Potsdam configuration.
- `dataset.py` - Loveda dataset loader.
- `encoders.py` - VisionMamba RGB encoder.
- `light_decoder.py` - Light U-Net decoder.
- `model.py` - Full model assembly.
- `train.py` - Iteration-based training loop.
- `train_domain.py` / `eval_domain.py` - Domain-specific training and evaluation helpers.

## Training

The training script imports `config_icprs.py` by default. To use Loveda instead, swap the import in `train.py` to:

```
from config import Config
```

Then update `DATA_ROOT`, `OUTPUT_DIR`, and `VISIONMAMBA_VARIANT` in the selected config file.

Run training:

```
# VisionMamba (Vim) Wrapper

This directory contains the VisionMamba-based semantic segmentation implementation for remote sensing datasets with domain adaptation support.

## Structure

- **Vim/** - Submodule containing the official VisionMamba implementation
	- Source: https://github.com/hustvl/Vim
	- Used as the backbone encoder
- **train.py** - Main training script with domain adaptation
- **config.py** - Configuration for LoveDA dataset
- **config_icprs.py** - Configuration for ISPRS Potsdam dataset
- **dataset.py** - LoveDA dataset loader with domain splits
- **model.py** - Integration of VisionMamba backbone with LightUNetDecoder
- **encoders.py** - Encoder wrapper for VisionMamba
- **light_decoder.py** - Lightweight U-Net decoder
- **train_domain.py** - Domain-specific training utilities
- **eval_domain.py** - Domain-specific evaluation
- **losses.py** - Loss functions
- **utils.py** - Training utilities

## Architecture

```
RGB Image (3ch) -> VisionMamba Encoder -> LightUNetDecoder -> Predictions
```

VisionMamba (Vim) features:
- **Bidirectional State Space Models** for vision tasks
- **Mamba blocks** adapted for 2D image data
- **Position encoding** for spatial awareness
- **Efficient long-range dependency modeling**
- **Linear complexity** with respect to sequence length

## Model Variants

Available variants:
- **tiny** - Smallest model
- **small** - Small model
- **base** - Base model

## Training

### Standard Training (LoveDA)
```bash
python train.py
```

### Domain-Specific Training
```bash
# Train on Urban domain only
# Edit config.py: TRAIN_ON_URBAN=True, TRAIN_ON_RURAL=False
python train.py

# Train on Rural domain only
# Edit config.py: TRAIN_ON_URBAN=False, TRAIN_ON_RURAL=True
python train.py
```

### ISPRS Potsdam
```bash
# Edit train.py to import from config_icprs
python train.py
```

## Domain Adaptation

VisionMamba wrapper includes domain adaptation features for LoveDA:
- **Urban domain**: High-density building areas
- **Rural domain**: Agricultural and natural landscapes
- **Mixed training**: Full dataset (default)
- **Domain-specific evaluation**: Separate metrics for each domain

Configuration example:
```python
# config.py
TRAIN_ON_URBAN = True  # Include urban samples
TRAIN_ON_RURAL = True  # Include rural samples
EVAL_BY_DOMAIN = True  # Report separate metrics
```

## Configuration

Key parameters:
```python
VARIANT = 'base'  # Model size
DATA_ROOT = '/path/to/loveda'
OUTPUT_DIR = '../Comparison_Experiments/visionmamba_base_512'
BATCH_SIZE = 8
LEARNING_RATE = 1e-4

# Domain settings
TRAIN_ON_URBAN = True
TRAIN_ON_RURAL = True
EVAL_BY_DOMAIN = False
```

## Output

```
OUTPUT_DIR/
├── checkpoints/      # Model weights (gitignored)
├── logs/            # Training logs (gitignored)
├── tensorboard/     # Training metrics (gitignored)
└── val_preds/       # Validation predictions (gitignored)
		├── urban/       # (if EVAL_BY_DOMAIN=True)
		└── rural/       # (if EVAL_BY_DOMAIN=True)
```

## Evaluation

Domain-specific evaluation:
```bash
python eval_domain.py --checkpoint path/to/checkpoint.pth
```

## Submodule

The **Vim/** subdirectory is a git submodule pointing to the official implementation.

To update:
```bash
cd Vim
git pull origin main
cd ..
git add Vim
```

## Citation

```
@article{zhu2024vision,
	title={Vision Mamba: Efficient Visual Representation Learning with Bidirectional State Space Model},
	author={Zhu, Lianghui and Liao, Bencheng and Zhang, Qian and Wang, Xinlong and Liu, Wenyu and Wang, Xinggang},
	journal={arXiv preprint arXiv:2401.09417},
	year={2024}
}
```
