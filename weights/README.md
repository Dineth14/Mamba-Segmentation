# Pre-trained Weights

This directory contains pre-trained model weights used for initialization and fine-tuning.

## Directory Structure

```
weights/
├── imagenet/
│   ├── resnet18-f37072fd.pth    # ResNet-18 pre-trained on ImageNet
│   └── resnet50-11ad3fa6.pth    # ResNet-50 pre-trained on ImageNet
```

## Pre-trained Models

### ImageNet Weights

**`resnet18-f37072fd.pth`**
- ResNet-18 backbone pre-trained on ImageNet-1K
- Used for comparison experiments (CNN_UNet and CNN_DeepLabv3p)
- Provides baseline comparison for Mamba-based models

**`resnet50-11ad3fa6.pth`**
- ResNet-50 backbone pre-trained on ImageNet-1K
- Used for deeper comparison baselines
- Standard initialization for CNN-based segmentation models

## Model-Specific Weights

Pre-trained encoder weights for Mamba architectures are typically:
1. Automatically downloaded from official sources during first use
2. Cached locally after initial download
3. Referenced by model variant (tiny, small, base, large)

Available Mamba model variants with pre-trained encoders:
- MambaVision (tiny, small, base, large, tiny2, large2)
- VMamba (tiny, small, base)
- VisionMamba (tiny, small, base)
- SpatialMamba / UrbanMamba variants

## Usage

### Loading Pre-trained Weights in Training

Pre-trained weights are automatically loaded based on configuration:

```python
# In config.py
ENCODER_PRETRAINED = True  # Load pre-trained encoder if available
MODEL_NAME = 'mambavision'
VARIANT = 'base'
```

### Manual Weight Loading

```python
import torch
from model import MambaVisionSegmentation

model = MambaVisionSegmentation(variant='base')

# Load encoder pre-training
checkpoint = torch.load('weights/imagenet/resnet50-11ad3fa6.pth')
model.encoder.load_state_dict(checkpoint, strict=False)
```

## Fine-tuned Segmentation Weights

Trained model checkpoints are stored in individual experiment directories:
- `Comparison_Experiments/[model_name]/checkpoints/` - LoveDA training checkpoints
- `Comparison_Experiments_ICPRS_potsdam/[model_name]/checkpoints/` - ISPRS Potsdam checkpoints

Key checkpoints:
- `best.pth` - Best validation performance model
- `latest.pth` - Most recent checkpoint (for resuming training)
- `checkpoint_[epoch].pth` - Per-epoch checkpoints

## Attribution

ImageNet pre-trained weights are sourced from official model repositories:
- ResNet models: [PyTorch Official Model Zoo](https://pytorch.org/vision/stable/models.html)
- Mamba models: Original architecture repositories
  - MambaVision: https://github.com/NVlabs/MambaVision
  - VMamba: https://github.com/MzeroMiko/VMamba
  - VisionMamba: https://github.com/hustvl/Vim
  - UrbanMamba/SpatialMamba: Local implementations

## License

- ImageNet pre-trained weights: Follow PyTorch license terms
- Fine-tuned weights: Same license as their respective model implementations
- See individual model directories for specific licensing information

