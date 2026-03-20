# ⚡ UNetFormer — Transformer Reference Baseline

> Part of [**Mamba-Segmentation**](../README.md): A Controlled Benchmark of Visual State-Space Backbones for Remote-Sensing Segmentation (IGRAAS 2026)

---

## 🎯 Role in the Benchmark

UNetFormer with ResNet-18 serves as the **Transformer reference baseline** in our controlled comparison. It combines a lightweight convolutional encoder with an efficient self-attention decoder, representing the Transformer family's approach to dense prediction for remote sensing.

In our benchmark, the ResNet-18 encoder is plugged into the **same fixed decoder** (LightUNetDecoder, 256ch), trained with the **same TriBraid loss** (Lovász-Softmax + Focal + Boundary), and evaluated on the **same data splits** as every other backbone. The only variable is the encoder.

---

## 🏗️ Architecture Overview

```
RGB Image (3ch)
    │
    ▼
┌──────────────────────────────────┐
│   ResNet-18 Encoder              │
│  (Efficient self-attention head) │
│                                  │
│  Stage 1 → F1 (stride 4)        │
│  Stage 2 → F2 (stride 8)        │
│  Stage 3 → F3 (stride 16)       │
│  Stage 4 → F4 (stride 32)       │
└──────────────────────────────────┘
    │ F1  │ F2  │ F3  │ F4
    ▼     ▼     ▼     ▼
┌──────────────────────────────────┐
│   LightUNetDecoder (256ch)       │
│   + Addition skip fusions        │
└──────────────────────────────────┘
    │
    ▼
  Predictions (C classes)
```

Key characteristics:
- **ResNet-18 backbone:** Lightweight residual network producing hierarchical features at strides {4, 8, 16, 32}.
- **Efficient self-attention:** UNetFormer's decoder design enables global context aggregation with reduced computational cost compared to full ViT-based approaches.
- **ImageNet-1K pretrained:** Uses `resnet18-f37072fd.pth`.

---

## 📐 Backbone Configuration

| Parameter | Value |
|---|---|
| Backbone | ResNet-18 |
| Feature Channels | 64, 64, 128, 256, 512 → projected to stride-matched dims |
| Pretrained | ImageNet-1K (`resnet18-f37072fd.pth`) |
| Decoder Channels | 256 |

Weights directory: `weights/imagenet/`

---

## ⚙️ Setup

1. **Install dependencies** (from repo root):
   ```bash
   pip install torch torchvision timm
   ```

2. **Download pretrained weights** into `weights/imagenet/`.

3. **Set paths** in the config file:
   ```python
   # config.py (LoveDA) or config_icprs.py (ISPRS Potsdam)
   DATA_ROOT = "/path/to/dataset"
   OUTPUT_DIR = "/path/to/output"
   ```

---

## 🚀 Training

### LoveDA (7-class, Urban + Rural)

```bash
# In train.py, ensure the import is:
# from config import Config

cd TransformerUNetFormer
python train.py
```

### ISPRS Potsdam (6-class)

```bash
# In train.py, ensure the import is:
# from config_icprs import Config

cd TransformerUNetFormer
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
| **LoveDA (All→All)** | **48.61** |
| Urban → Rural | 34.56 |
| Rural → Urban | 44.84 |
| **ISPRS Potsdam** | **74.99** |

*Backbone: ResNet-18*

### Comparison Context

| Type | Backbone | LoveDA mIoU | U→R | R→U | Potsdam mIoU |
|---|---|---:|---:|---:|---:|
| CNN | DeepLabv3+ | 43.01 | 30.36 | 39.98 | 75.09 |
| **Transformer** | **UNetFormer** | 48.61 | 34.56 | 44.84 | 74.99 |
| **SSM** | **VMamba-Small** | **55.66** | **40.62** | 53.52 | **77.59** |
| **SSM** | MambaVision-L | 55.25 | 38.53 | **54.01** | 77.07 |
| SSM | Spatial-Mamba-B | 48.03 | 35.23 | 46.55 | 70.00 |

---

## 💡 Distinctive Findings

> **Transformers beat CNNs — but SSMs beat Transformers.** UNetFormer scores 48.61 on LoveDA, a solid +5.60 over DeepLabv3+. But VMamba-Small surpasses it by +7.05, showing that self-attention's quadratic scaling isn't needed when linear SSM scanning captures global context more efficiently.

On ISPRS Potsdam, UNetFormer (74.99) sits essentially at parity with DeepLabv3+ (75.09) — both are outclassed by the SSM family. Under domain shift, UNetFormer's 34.56 Urban→Rural and 44.84 Rural→Urban show meaningful improvement over the CNN baseline, but still trail VMamba-Small by 6+ points in both directions. The Transformer provides a useful middle ground: better than local convolutions, but insufficient to match state-space global modeling.

---

## 📁 File Reference

| File | Purpose |
|---|---|
| `config.py` | LoveDA configuration |
| `config_icprs.py` | ISPRS Potsdam configuration |
| `encoders.py` | ResNet-18 backbone wrapper |
| `model.py` | Full model assembly (encoder + decoder) |
| `light_decoder.py` | Shared LightUNetDecoder |
| `losses.py` | TriBraid loss (Lovász + Focal + Boundary) |
| `train.py` | Iteration-based training loop |
| `train_domain.py` | Domain-shift training (single-domain) |
| `eval_domain.py` | Cross-domain evaluation |
| `dataset.py` | LoveDA dataset loader |
| `dataset_isprs.py` | ISPRS Potsdam dataset loader |

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
