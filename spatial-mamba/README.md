# 🌐 Spatial-Mamba — Spatially-Aware State-Space Backbone

> Part of [**Mamba-Segmentation**](../README.md): A Controlled Benchmark of Visual State-Space Backbones for Remote-Sensing Segmentation (IGRAAS 2026)

---

## 🎯 Role in the Benchmark

Spatial-Mamba serves as the **spatially-aware SSM backbone** in our controlled comparison. Derived from UrbanMamba's RGB pathway, it introduces explicit positional inductive biases into the state-space pathway, aiming to make SSM scanning order-aware of 2D spatial structure.

In our benchmark, Spatial-Mamba is plugged into the **same fixed decoder** (LightUNetDecoder, 256ch), trained with the **same TriBraid loss** (Lovász-Softmax + Focal + Boundary), and evaluated on the **same data splits** as every other backbone. The only variable is the encoder.

**Source:** [github.com/EdwardChasel/Spatial-Mamba](https://github.com/EdwardChasel/Spatial-Mamba)

---

## 🏗️ Architecture Overview

```
RGB Image (3ch)
    │
    ▼
┌─────────────────────────────────┐
│   Spatial-Mamba Encoder         │
│  (Spatially-aware SSM scanning) │
│                                 │
│  Stage 1 → F1 (stride 4)       │
│  Stage 2 → F2 (stride 8)       │
│  Stage 3 → F3 (stride 16)      │
│  Stage 4 → F4 (stride 32)      │
└─────────────────────────────────┘
    │ F1  │ F2  │ F3  │ F4
    ▼     ▼     ▼     ▼
┌─────────────────────────────────┐
│   LightUNetDecoder (256ch)      │
│   + Addition skip fusions       │
└─────────────────────────────────┘
    │
    ▼
  Predictions (C classes)
```

Key mechanisms:
- **Spatial-aware state-space modeling:** Embeds explicit 2D positional information into the SSM scanning process.
- **Multi-directional scanning:** Captures comprehensive spatial context across the feature map.
- **Hierarchical features:** `{F1, F2, F3, F4}` at strides `{4, 8, 16, 32}` feed into the shared decoder.

---

## 📐 Variant Table

| Variant | Dims (C1–C4) | Depths | Drop Path | d_state | MLP Ratio | Pretrained Weights |
|---|---|---|---|---|---|---|
| **Tiny** | 64, 128, 256, 512 | (2, 4, 8, 4) | 0.2 | 1 | 4.0 | `spatialmamba_tiny_224_1k.pth` |
| **Small** | 64, 128, 256, 512 | (2, 4, 21, 5) | 0.3 | 1 | 4.0 | `spatialmamba_small_224_1k.pth` |
| **Base** | 96, 192, 384, 768 | (2, 4, 21, 5) | 0.5 | 1 | 4.0 | `spatialmamba_base_224_1k.pth` |

All pretrained on ImageNet-1K. Weights directory: `spatial-mamba/weights/imageNet1K/`

---

## ⚙️ Setup

1. **Install dependencies** (from repo root):
   ```bash
   pip install torch torchvision mamba-ssm causal-conv1d timm
   ```

2. **Download pretrained weights** into `spatial-mamba/weights/imageNet1K/`.

3. **Set paths** in the config file:
   ```python
   # config.py (LoveDA) or config_icprs.py (ISPRS Potsdam)
   DATA_ROOT = "/path/to/dataset"
   SPATIALMAMBA_VARIANT = "base"   # tiny | small | base
   OUTPUT_DIR = "/path/to/output"
   ```

---

## 🚀 Training

### LoveDA (7-class, Urban + Rural)

```bash
# In train.py, ensure the import is:
# from config import Config

cd spatial-mamba
python train.py
```

### ISPRS Potsdam (6-class)

```bash
# In train.py, ensure the import is:
# from config_icprs import Config

cd spatial-mamba
python train.py
```

### Domain-Shift Evaluation

```bash
# Train on Urban only, evaluate on Rural (or vice versa)
python train_domain.py
```

---

## 📊 Results

All numbers from the paper — same decoder, same loss, same schedule across all backbones.

| Setting | mIoU |
|---|---:|
| **LoveDA (All→All)** | **48.03** |
| Urban → Rural | 35.23 |
| Rural → Urban | 46.55 |
| **ISPRS Potsdam** | **70.00** |

*Variant used for reported results: Spatial-Mamba-B (Base)*

### Comparison Context

| Type | Backbone | LoveDA mIoU | U→R | R→U | Potsdam mIoU |
|---|---|---:|---:|---:|---:|
| CNN | DeepLabv3+ | 43.01 | 30.36 | 39.98 | 75.09 |
| Transformer | UNetFormer | 48.61 | 34.56 | 44.84 | 74.99 |
| **SSM** | **VMamba-Small** | **55.66** | **40.62** | 53.52 | **77.59** |
| **SSM** | MambaVision-L | 55.25 | 38.53 | **54.01** | 77.07 |
| **SSM** | **Spatial-Mamba-B** | 48.03 | 35.23 | 46.55 | 70.00 |

---

## 💡 Distinctive Findings

> **Spatial-aware scanning alone is insufficient.** Spatial-Mamba-B scores 48.03 on LoveDA — competitive with UNetFormer (48.61) and well above DeepLabv3+ (43.01) — but falls 7.63 points behind VMamba-Small. Positional inductive biases help, but the cross-scan mechanism with global context aggregation matters more.

On ISPRS Potsdam, Spatial-Mamba-B achieves 70.00 — notably lower than both VMamba (77.59) and CNN DeepLabv3+ (75.09). The gap on Potsdam's high-resolution tiles suggests that explicit spatial biases may interfere with fine-grained boundary delineation at higher resolutions, a domain where pure cross-scan SSMs excel.

---

## 📁 File Reference

| File | Purpose |
|---|---|
| `config.py` | LoveDA configuration |
| `config_icprs.py` | ISPRS Potsdam configuration |
| `encoders.py` | Spatial-Mamba backbone wrapper (`RGBEncoder`) |
| `model.py` | Full model assembly (encoder + decoder) |
| `light_decoder.py` | Shared LightUNetDecoder |
| `losses.py` | TriBraid loss (Lovász + Focal + Boundary) |
| `train.py` | Iteration-based training loop |
| `train_domain.py` | Domain-shift training (single-domain) |
| `eval_domain.py` | Cross-domain evaluation |
| `dataset.py` | LoveDA dataset loader |

---

## 📜 Citation

```bibtex
@article{wasalathilaka2026controlledbenchmark,
  title={A Controlled Benchmark of Visual State-Space Backbones with
         Domain-Shift and Boundary Analysis for Remote-Sensing
         Segmentation},
  author={Wasalathilaka, Nichula and Perea, Dineth and Samarakoon,
          Oshadha and Wijenayake, Buddhi and Godaliyadda, Roshan and
          Herath, Vijitha and Ekanayake, Parakrama},
  journal={IGRAAS 2026},
  year={2026}
}
```

---

[← Back to main README](../README.md)
