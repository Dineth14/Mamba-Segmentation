# UrbanMamba Model Architecture

> **RGB-only semantic segmentation model for remote sensing**
>
> MambaVision backbone (tiny/small/base) with a lightweight U-Net decoder.

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
│   │  RGBEncoder  │  MambaVision-Tiny/Small/Base                           │
│   │  (MambaVision)│                                                      │
│   │ Stage 0: C0  │  1/4 scale                                             │
│   │ Stage 1: C1  │  1/8 scale                                             │
│   │ Stage 2: C2  │  1/16 scale                                            │
│   │ Stage 3: C3  │  1/32 scale                                            │
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

## MambaVision Variant Configuration

Select the MambaVision model size in `UrbanMamba/config.py`:

```
MAMBAVISION_VARIANT: "tiny" | "tiny2" | "small" | "base" | "base_21k" | "large" | "large_21k" | "large2" | "large2_512_21k" | "large3_256_21k" | "large3_512_21k"
```

Default settings:

| Variant | Depths | Dims | Drop Path |
|---------|--------|------|-----------|
| tiny    | [2, 2, 27, 2] | [96, 192, 384, 768] | 0.2 |
| small   | [2, 2, 27, 2] | [96, 192, 384, 768] | 0.3 |
| base    | [2, 2, 27, 2] | [128, 256, 512, 768] | 0.6 |

Pretrained weights are loaded from `MAMBAVISION_WEIGHTS_DIR` using the map in config.

---

## Decoder (Light U-Net)

**Source:** `light_decoder.py`

- Bottleneck Conv block
- 3 upsampling stages with skip concatenation
- Main head at 1/4 scale (upsampled to 1x)
- Single main head only

Outputs:
- `main`: (N, 7, H, W)

---

## Tensor Shapes (Example Input 3×640×640)

| Stage | Resolution | Channels (tiny) | Channels (small) | Channels (base) |
|-------|------------|-----------------|------------------|-----------------|
| 0 | 1/4 | 80 | 96 | 128 |
| 1 | 1/8 | 160 | 192 | 256 |
| 2 | 1/16 | 320 | 384 | 512 |
| 3 | 1/32 | 640 | 768 | 1024 |

---

## Summary

- RGB-only MambaVision encoder for efficient training and inference.
- Variant selection via config for tiny/small/base backbones.
- Lightweight U-Net decoder for segmentation output.
