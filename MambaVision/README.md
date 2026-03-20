# 🔬 MambaVision — Hybrid SSM + Self-Attention Backbone

> Part of [**Mamba-Segmentation**](../README.md): A Controlled Benchmark of Visual State-Space Backbones for Remote-Sensing Segmentation (IGRAAS 2026)

---

## 🎯 Role in the Benchmark

MambaVision serves as the **hybrid SSM backbone** in our controlled comparison. Developed by NVIDIA, it interleaves Mamba blocks in early stages with self-attention in later stages — combining linear-complexity sequence modeling with global receptive fields.

In our benchmark, MambaVision is plugged into the **same fixed decoder** (LightUNetDecoder, 256ch), trained with the **same TriBraid loss** (Lovász-Softmax + Focal + Boundary), and evaluated on the **same data splits** as every other backbone. The only variable is the encoder.

**Source:** [github.com/NVlabs/MambaVision](https://github.com/NVlabs/MambaVision)

---

## 🏗️ Architecture Overview

```
RGB Image (3ch)
    │
    ▼
┌─────────────────────────────┐
│   MambaVision Encoder       │
│  (Mamba blocks → Attention) │
│                             │
│  Stage 1 → F1 (stride 4)   │
│  Stage 2 → F2 (stride 8)   │
│  Stage 3 → F3 (stride 16)  │
│  Stage 4 → F4 (stride 32)  │
└─────────────────────────────┘
    │ F1  │ F2  │ F3  │ F4
    ▼     ▼     ▼     ▼
┌─────────────────────────────┐
│   LightUNetDecoder (256ch)  │
│   + Addition skip fusions   │
└─────────────────────────────┘
    │
    ▼
  Predictions (C classes)
```

The encoder outputs hierarchical features `{F1, F2, F3, F4}` at strides `{4, 8, 16, 32}`, which feed into the shared decoder via addition-based skip connections.

---

## 📐 Variant Table

| Variant | Dims (C1–C4) | Depths | Drop Path | Pretrained Weights |
|---|---|---|---|---|
| **Tiny** | 80, 160, 320, 640 | (1, 3, 8, 4) | 0.2 | `mambavision_tiny_1k.pth.tar` |
| **Tiny2** | 80, 160, 320, 640 | (1, 3, 11, 4) | 0.2 | `mambavision_tiny2_1k.pth.tar` |
| **Small** | 96, 192, 384, 768 | (3, 3, 7, 5) | 0.2 | `mambavision_small_1k.pth.tar` |
| **Base** | 128, 256, 512, 1024 | (3, 3, 10, 5) | 0.3 | `mambavision_base_1k.pth.tar` |
| **Large** | 196, 392, 784, 1568 | (3, 3, 10, 5) | 0.3 | `mambavision_large_1k.pth.tar` |
| **Large2** | 196, 392, 784, 1568 | (3, 3, 12, 5) | 0.3 | `mambavision_large2_1k.pth.tar` |

All pretrained on ImageNet-1K. Weights directory: `MambaVision/weights/1k/`

---

## ⚙️ Setup

1. **Install dependencies** (from repo root):
   ```bash
   pip install torch torchvision mamba-ssm causal-conv1d timm
   ```

2. **Download pretrained weights** into `MambaVision/weights/1k/`.

3. **Set paths** in the config file:
   ```python
   # config.py (LoveDA) or config_icprs.py (ISPRS Potsdam)
   DATA_ROOT = "/path/to/dataset"
   MAMBAVISION_VARIANT = "large"   # tiny | tiny2 | small | base | large | large2
   OUTPUT_DIR = "/path/to/output"
   ```

---

## 🚀 Training

### LoveDA (7-class, Urban + Rural)

```bash
# In train.py, ensure the import is:
# from config import Config

cd MambaVision
python train.py
```

### ISPRS Potsdam (6-class)

```bash
# In train.py, ensure the import is:
# from config_icprs import Config

cd MambaVision
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
| **LoveDA (All→All)** | **55.25** |
| Urban → Rural | 38.53 |
| Rural → Urban | **54.01** |
| **ISPRS Potsdam** | **77.07** |

*Variant used for reported results: MambaVision-L (Large)*

### Comparison Context

| Type | Backbone | LoveDA mIoU | U→R | R→U | Potsdam mIoU |
|---|---|---:|---:|---:|---:|
| CNN | DeepLabv3+ | 43.01 | 30.36 | 39.98 | 75.09 |
| Transformer | UNetFormer | 48.61 | 34.56 | 44.84 | 74.99 |
| **SSM** | **VMamba-Small** | **55.66** | **40.62** | 53.52 | **77.59** |
| **SSM** | **MambaVision-L** | 55.25 | 38.53 | **54.01** | 77.07 |
| SSM | Spatial-Mamba-B | 48.03 | 35.23 | 46.55 | 70.00 |

---

## 💡 Distinctive Findings

> **Bigger ≠ better under a fixed decoder.** MambaVision-L carries far more parameters than VMamba-Small yet scores 55.25 vs. 55.66 on LoveDA. Scaling the encoder past a threshold buys nothing when the decoder capacity stays constant — a key insight for practical RS deployment.

MambaVision-L achieves the **best Rural→Urban transfer** (54.01), suggesting the hybrid attention mechanism preserves urban structure priors better under domain shift. However, on ISPRS Potsdam it trails VMamba-Small (77.07 vs. 77.59), indicating that pure SSM scanning may generalize better to high-resolution grids.

---

## 📁 File Reference

| File | Purpose |
|---|---|
| `config.py` | LoveDA configuration |
| `config_icprs.py` | ISPRS Potsdam configuration |
| `encoders.py` | MambaVision backbone wrapper (`RGBEncoder`) |
| `model.py` | Full model assembly (encoder + decoder) |
| `light_decoder.py` | Shared LightUNetDecoder |
| `losses.py` | TriBraid loss (Lovász + Focal + Boundary) |
| `train.py` | Iteration-based training loop |
| `train_domain.py` | Domain-shift training (single-domain) |
| `eval_domain.py` | Cross-domain evaluation |
| `dataset.py` | LoveDA dataset loader |
| `dataset_isprs.py` | ISPRS Potsdam dataset loader |
| `MambaVision/` | Official NVIDIA MambaVision implementation |

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
