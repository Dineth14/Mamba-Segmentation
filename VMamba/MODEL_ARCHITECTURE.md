# UrbanMamba Model Architecture

> **RGB-only semantic segmentation model for remote sensing**
>
> VMamba backbone (tiny/small/base) with a lightweight U-Net decoder and
> iteration-based training for LOVEDA.

---

## Architecture Overview

```
┌────────────────────────────────────────────────────────────────────────────┐
│                                UrbanMamba                                  │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│   ┌──────────────┐                                                        │
│   │  RGB Image   │                                                        │
│   │  (3, H, W)   │                                                        │
│   └──────┬───────┘                                                        │
│          ▼                                                                │
│   ┌──────────────┐                                                        │
│   │  RGBEncoder  │  VMamba-Tiny/Small/Base                                │
│   │  (VMamba)    │  Multi-scale feature pyramid                           │
│   │ Stage 0: C0  │  1/4 scale                                              │
│   │ Stage 1: C1  │  1/8 scale                                              │
│   │ Stage 2: C2  │  1/16 scale                                             │
│   │ Stage 3: C3  │  1/32 scale                                             │
│   └──────┬───────┘                                                        │
│          ▼                                                                │
│   ┌──────────────────────┐                                                │
│   │  LightUNetDecoder     │  Conv-BN-ReLU blocks + skip connections        │
│   │  (Aux head optional)  │                                                │
│   └──────┬────────────────┘                                                │
│          ▼                                                                │
│   ┌─────────────────────────────────────────┐                             │
│   │        Segmentation Output              │                             │
│   │         (7, H, W) classes               │                             │
│   └─────────────────────────────────────────┘                             │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘
```

---

## VMamba Encoder (RGBEncoder)

**Source:** `UrbanMamba/encoders.py`

The encoder is VMamba with a 4-stage feature pyramid:

- **Patch embedding:** 4x downsampling to reach 1/4 resolution.
- **Stage 0–3:** VMamba blocks per stage with progressive downsampling.
- **Outputs:** 4 feature maps used as skip connections in the decoder.

Pretrained weights are loaded from `VMAMBA_WEIGHTS_DIR` using the map in
`UrbanMamba/config.py`.

### Variant Configuration

Select the VMamba model size in `UrbanMamba/config.py`:

```
VMAMBA_VARIANT: "tiny" | "small" | "base"
```

Default settings:

| Variant | Depths | Dims | Drop Path |
|---------|--------|------|-----------|
| tiny    | [2, 2, 9, 2] | [96, 192, 384, 768] | 0.2 |
| small   | [2, 2, 27, 2] | [96, 192, 384, 768] | 0.3 |
| base    | [2, 2, 27, 2] | [128, 256, 512, 1024] | 0.6 |

---

## Decoder (Light U-Net)

**Source:** `UrbanMamba/light_decoder.py`

The decoder is a compact U-Net style stack:

- **Bottleneck:** ConvBlock on C3 (1/32).
- **Up3:** Upsample 2x, concatenate C2, ConvBlock.
- **Up2:** Upsample 2x, concatenate C1, ConvBlock.
- **Up1:** Upsample 2x, concatenate C0, ConvBlock.
- **Head:** 1x1 conv on final features, bilinear upsample to input size.
- **Aux head:** Optional, uses 1/8 features (disabled by default).

Outputs:
- `main`: (N, 7, H, W)
- `aux`: (N, 7, H, W) when enabled

---

## Tensor Shapes (Example Input 3x640x640)

| Stage | Resolution | Channels (tiny/small) | Channels (base) |
|-------|------------|-----------------------|----------------|
| 0 | 1/4 | 96 | 128 |
| 1 | 1/8 | 192 | 256 |
| 2 | 1/16 | 384 | 512 |
| 3 | 1/32 | 768 | 1024 |
| Decoder bottleneck | 1/32 | 256 | 256 |
| Decoder up2 | 1/8 | 128 | 128 |
| Decoder up1 | 1/4 | 128 | 128 |
| Output | 1/1 | 7 | 7 |

---

## Loss Function

**Source:** `UrbanMamba/losses.py` (`TriBraidLoss`)

Combined loss:

- Lovasz-Softmax (1.0)
- Focal Loss (1.0, gamma=2.0)
- Boundary Loss (0.5)

---

## Training Pipeline

**Source:** `UrbanMamba/train.py`

- **Optimizer:** AdamW with differential LR
  - Backbone: `LR_BACKBONE`
  - Head: `LR_HEAD`
- **Scheduler:** Polynomial decay (`POLY_POWER`)
- **AMP:** Optional (`USE_AMP`)
- **Validation:** Every `VAL_INTERVAL` iterations
- **Checkpoints:** Saved at each validation step

---

## Data Pipeline

**Source:** `UrbanMamba/dataset.py`

- RandomCrop to `CROP_SIZE` during training
- HorizontalFlip / VerticalFlip / RandomRotate90
- Optional color jitter (RGB only)
- ImageNet normalization
- Label remapping:
  - Original 0 -> ignore (255)
  - 1..7 -> 0..6

---

## Summary

- RGB-only VMamba encoder for efficient training and inference.
- Variant selection via config for tiny/small/base backbones.
- Lightweight U-Net decoder with skip connections for spatial refinement.
- Iteration-based training with AMP + polynomial LR scheduling.
