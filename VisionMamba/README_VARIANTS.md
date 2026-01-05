# Vision Mamba Segmentation - Multi-Variant Support

## Overview

This updated Vision Mamba configuration supports **all three variants** (tiny, small, base) with proper weight loading and training capabilities.

## Available Variants

| Variant | Embed Dim | Depth | Weights | Accuracy | Size |
|---------|-----------|-------|---------|----------|------|
| **tiny** | 192 | 24 | `vim_t_midclstok_76p1acc.pth` | 76.1% | 109.7 MB |
| **tiny** (FT) | 192 | 24 | `vim_t_midclstok_ft_78p3acc.pth` | 78.3% | 111.2 MB |
| **small** | 384 | 24 | `vim_s_midclstok_80p5acc.pth` | 80.5% | 394.2 MB |
| **small** (FT) | 384 | 24 | `vim_s_midclstok_ft_81p6acc.pth` | 81.6% | 397.4 MB |
| **base** | 768 | 24 | `vim_b_midclstok_81p9acc.pth` | 81.9% | 1489.8 MB |

**Note:** (FT) = Fine-tuned weights. Fine-tuned versions are available for tiny and small variants.

## Configuration

### config.py

The updated `config.py` provides a dataclass-based configuration system with:

```python
from config import cfg

# Predefined configurations
cfg.VIM_VARIANT = "tiny"        # "tiny", "small", or "base"
cfg.VIM_DEPTHS = (1,)           # (1,) for single block, (24,) for full
cfg.USE_FINETUNED_WEIGHTS = True # Use fine-tuned weights if available
cfg.WEIGHTS_PATH                 # Auto-resolved from variant and weights dir
cfg.VIM_WEIGHTS_DIR              # Points to ./weights/ folder
```

### Quick Start

```python
from config import Config

# Create config with specific variant
cfg = Config()
cfg.VIM_VARIANT = "small"
cfg.USE_FINETUNED_WEIGHTS = True

# Weights are automatically resolved
print(f"Variant: {cfg.VIM_VARIANT}")
print(f"Weights: {cfg.WEIGHTS_PATH}")
print(f"Dims: {cfg.VIM_DIMS}")
print(f"Depths: {cfg.VIM_DEPTHS}")
```

## Using Different Variants

### Via Configuration

```python
from config import Config
from model import build_model

# Create config for small variant
cfg = Config()
cfg.VIM_VARIANT = "small"
cfg.VIM_DEPTHS = (1,)  # Single block encoder
cfg.USE_FINETUNED_WEIGHTS = True

# Build model
model = build_model(cfg)
```

### Direct Model Creation

```python
from model import VisionMambaSegmentation

# Tiny variant (fast)
model_tiny = VisionMambaSegmentation(
    num_classes=7,
    variant="tiny",
    encoder_depths=(1,),
    pretrained_rgb=True,
    weights_path="/path/to/vim_t_midclstok_ft_78p3acc.pth"
)

# Small variant (balanced)
model_small = VisionMambaSegmentation(
    num_classes=7,
    variant="small",
    encoder_depths=(1,),
    pretrained_rgb=True,
    weights_path="/path/to/vim_s_midclstok_ft_81p6acc.pth"
)

# Base variant (powerful)
model_base = VisionMambaSegmentation(
    num_classes=7,
    variant="base",
    encoder_depths=(24,),  # Full depth
    pretrained_rgb=True,
    weights_path="/path/to/vim_b_midclstok_81p9acc.pth"
)
```

## Weights Folder Structure

All weights should be placed in:
```
/storage2/ChangeDetection/NSST-mamba/Mamba-Segmentation/VisionMamba/weights/
├── vim_t_midclstok_76p1acc.pth
├── vim_t_midclstok_ft_78p3acc.pth
├── vim_s_midclstok_80p5acc.pth
├── vim_s_midclstok_ft_81p6acc.pth
└── vim_b_midclstok_81p9acc.pth
```

The configuration automatically loads the correct weights based on:
1. **Variant** (`tiny`, `small`, `base`)
2. **Weight type** (fine-tuned or base)
3. **File existence** (fallback to base if fine-tuned unavailable)

## Model Architecture Components

### RGBEncoder (encoders.py)

Multi-variant Vision Mamba encoder supporting:
- Automatic dimension resolution per variant
- Configurable depth (1-24 blocks)
- Proper spatial feature extraction for segmentation
- Weight loading with fallback logic

```python
from encoders import RGBEncoder

encoder = RGBEncoder(
    variant="small",
    depths=(1,),
    dims=(384,),
    pretrained=True,
    weights_path=cfg.WEIGHTS_PATH
)
features = encoder(rgb_input)  # Returns [(B, C, H', W')]
```

### VisionMambaSegmentation (model.py)

Complete segmentation model supporting all variants:
- Vision Mamba encoder (all variants)
- Lightweight UNet decoder
- Differential learning rates
- Test-time augmentation support

```python
from model import VisionMambaSegmentation, build_model

model = build_model(cfg)  # From config
output, _ = model(rgb_input)  # (B, 7, H, W)
```

## Training Configuration

### Variant-Specific Settings

```python
# Tiny variant (smallest/fastest)
cfg.VIM_VARIANT = "tiny"
cfg.BATCH_SIZE = 8  # Can use larger batch
cfg.LR_BACKBONE = 6e-5
cfg.LR_HEAD = 3e-4

# Small variant (balanced)
cfg.VIM_VARIANT = "small"
cfg.BATCH_SIZE = 4  # Moderate batch size
cfg.LR_BACKBONE = 6e-5
cfg.LR_HEAD = 3e-4

# Base variant (most powerful)
cfg.VIM_VARIANT = "base"
cfg.BATCH_SIZE = 2  # Smaller batch for GPU memory
cfg.LR_BACKBONE = 3e-5  # Lower LR for larger model
cfg.LR_HEAD = 1e-4
```

### Automatic Weight Selection

The configuration intelligently selects weights:

```python
# If using fine-tuned weights and they exist
cfg.USE_FINETUNED_WEIGHTS = True
# → Selects: vim_t_midclstok_ft_78p3acc.pth (for tiny)

# If fine-tuned weights don't exist, falls back to base
cfg.USE_FINETUNED_WEIGHTS = True
cfg.VIM_VARIANT = "base"
# → Selects: vim_b_midclstok_81p9acc.pth (base only)

# To use base weights explicitly
cfg.USE_FINETUNED_WEIGHTS = False
# → Selects: vim_t_midclstok_76p1acc.pth (for tiny)
```

## Parameter Information

### Configuration Class Attributes

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `VIM_VARIANT` | str | "tiny" | Model variant (tiny/small/base) |
| `VIM_DEPTHS` | Tuple | (1,) | Mamba blocks in encoder |
| `VIM_DIMS` | Tuple | (192,) | Auto-resolved embedding dimension |
| `VIM_DROP_PATH` | float | 0.0 | Drop path rate |
| `USE_FINETUNED_WEIGHTS` | bool | True | Prefer fine-tuned weights |
| `DECODER_CHANNELS` | int | 256 | Decoder hidden channels |
| `USE_RMS_NORM` | bool | True | Use RMSNorm (default for Vim) |
| `VIM_FUSED_ADD_NORM` | bool | True | Fused add+norm operation |
| `VIM_RESIDUAL_IN_FP32` | bool | True | Keep residuals in FP32 |
| `VIM_IF_ROPE` | bool | False | Use RoPE (Vim uses absolute pos embed) |
| `VIM_BIMAMBA_TYPE` | str | "v2" | BiMamba version |

## Example: Complete Training Setup

```python
from config import Config
from model import build_model
from train import Trainer

# Step 1: Create and configure
cfg = Config()
cfg.VIM_VARIANT = "small"           # Use small variant
cfg.VIM_DEPTHS = (1,)               # Single block encoder
cfg.USE_FINETUNED_WEIGHTS = True    # Use fine-tuned weights
cfg.BATCH_SIZE = 4
cfg.MAX_ITERS = 50000

# Step 2: Build model
model = build_model(cfg)

# Step 3: Check weights loaded
print(f"Model weights from: {cfg.WEIGHTS_PATH}")

# Step 4: Train
trainer = Trainer(model, cfg)
trainer.train()
```

## Testing Variants

### Config Test
```bash
cd /storage2/ChangeDetection/NSST-mamba/Mamba-Segmentation/VisionMamba
python config.py
```

Output shows:
- Selected variant
- Resolved dimensions
- Weight file path
- Weight existence verification
- Available weights in directory

### Model Components Test
```bash
python encoders.py    # Tests all encoder variants
python model.py       # Tests all model variants
```

## Notes

1. **Automatic Weight Resolution**: The config automatically selects the correct weight file based on variant and fine-tune preference.

2. **Fallback Logic**: If fine-tuned weights are requested but unavailable, the system automatically falls back to base weights.

3. **Spatial Feature Extraction**: The encoder is configured to return spatial feature maps suitable for segmentation (not pooled class tokens).

4. **Multi-variant Training**: You can train and compare all variants systematically by iterating:
   ```python
   for variant in ["tiny", "small", "base"]:
       cfg.VIM_VARIANT = variant
       model = build_model(cfg)
       # Train...
   ```

## Architecture Details

### Encoder (RGBEncoder)
- Patch embedding (16x16 patches by default)
- Vision Mamba blocks (configurable count)
- Absolute positional embeddings
- RMSNorm + residual connections
- Returns spatial feature maps (B, C, H/16, W/16)

### Decoder (LightUNetDecoder)
- Lightweight upsampling path
- Skip connections from encoder
- Progressive resolution restoration
- Final segmentation head

### Model Assembly (VisionMambaSegmentation)
- Combines encoder + decoder
- Differential learning rates
- Parameter grouping
- Test-time augmentation support

## Performance Expectations

| Variant | Params | VRAM (GB) | Speed | Quality |
|---------|--------|-----------|-------|---------|
| Tiny | ~85M | 2-3 | Fast | Good |
| Small | ~110M | 4-6 | Medium | Better |
| Base | ~307M | 8-12 | Slow | Best |

*Approximate values for 512x512 inputs, batch size 4*

## Troubleshooting

### Weights not found
- Check weights exist in `/storage2/ChangeDetection/NSST-mamba/Mamba-Segmentation/VisionMamba/weights/`
- Set `cfg.WEIGHTS_PATH` manually if auto-resolution fails

### Dimension mismatch
- Verify variant matches encoder dims: tiny→192, small→384, base→768
- Check `cfg.VIM_DIMS` after `__post_init__`

### Memory issues
- Reduce batch size
- Use smaller variant (tiny → small → base)
- Reduce encoder depth (24 → 12 → 1)

## References

- Vision Mamba: https://github.com/Hustvl/Vim
- Paper: Vision Mamba: Efficient Visual Representation Learning with Bidirectional State Space Models
