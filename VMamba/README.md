# 🏆 VMamba — Cross-Scan 2D Selective State-Space Backbone

> Part of [**Mamba-Segmentation**](../README.md): A Controlled Benchmark of Visual State-Space Backbones for Remote-Sensing Segmentation (IGRAAS 2026)

---

## 🎯 Role in the Benchmark

VMamba is the **top-performing backbone** in our controlled comparison. It extends the selective state-space model (S6) to 2D vision via a cross-scan mechanism that processes image patches along multiple spatial directions, achieving global receptive fields with linear complexity.

In our benchmark, VMamba is plugged into the **same fixed decoder** (LightUNetDecoder, 256ch), trained with the **same TriBraid loss** (Lovász-Softmax + Focal + Boundary), and evaluated on the **same data splits** as every other backbone. The only variable is the encoder.

**Source:** [github.com/MzeroMiko/VMamba](https://github.com/MzeroMiko/VMamba)

---

## 🏗️ Architecture Overview

```
RGB Image (3ch)
    │
    ▼
┌───────────────────────────────┐
│   VMamba Encoder              │
│  (2D Selective Scan / SS2D)   │
│                               │
│  Stage 1 → F1 (stride 4)     │
│  Stage 2 → F2 (stride 8)     │
│  Stage 3 → F3 (stride 16)    │
│  Stage 4 → F4 (stride 32)    │
└───────────────────────────────┘
    │ F1  │ F2  │ F3  │ F4
    ▼     ▼     ▼     ▼
┌───────────────────────────────┐
│   LightUNetDecoder (256ch)    │
│   + Addition skip fusions     │
└───────────────────────────────┘
    │
    ▼
  Predictions (C classes)
```

Key mechanisms:
- **2D Selective Scan (SS2D):** Processes flattened 2D feature maps through SSM along four scanning directions (left→right, right→left, top→bottom, bottom→top), then merges results.
- **Linear complexity:** O(N) with respect to token count, vs. O(N²) for self-attention.
- **Hierarchical features:** `{F1, F2, F3, F4}` at strides `{4, 8, 16, 32}` feed into the shared decoder.

---

## 📐 Variant Table

| Variant | Dims (C1–C4) | Depths | Drop Path | Weight Set |
|---|---|---|---|---|
| **Tiny** | 96, 192, 384, 768 | (2, 2, 9, 2) | 0.2 | ImageNet-1K / ADE20K |
| **Small** | 96, 192, 384, 768 | (2, 2, 27, 2) | 0.3 | ImageNet-1K / ADE20K |
| **Base** | 128, 256, 512, 1024 | (2, 2, 27, 2) | 0.6 | ImageNet-1K / ADE20K |

Three weight sets are supported: `imagenet1k`, `ade20k`, and `vanilla_ade20k`. Weights directory: `VMamba/Vmamba_weights/`

---

## ⚙️ Setup

1. **Install dependencies** (from repo root):
   ```bash
   pip install torch torchvision mamba-ssm causal-conv1d timm
   ```

2. **Download pretrained weights** into `VMamba/Vmamba_weights/ImageNet-1K/` (or the corresponding ADE20K directory).

3. **Set paths** in the config file:
   ```python
   # config.py (LoveDA) or config_icprs.py (ISPRS Potsdam)
   DATA_ROOT = "/path/to/dataset"
   VMAMBA_VARIANT = "small"           # tiny | small | base
   VMAMBA_WEIGHT_SET = "imagenet1k"   # imagenet1k | ade20k | vanilla_ade20k
   OUTPUT_DIR = "/path/to/output"
   ```

---

## 🚀 Training

### LoveDA (7-class, Urban + Rural)

```bash
# In train.py, ensure the import is:
# from config import Config

cd VMamba
python train.py
```

### ISPRS Potsdam (6-class)

```bash
# In train.py, ensure the import is:
# from config_icprs import Config

cd VMamba
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
| **LoveDA (All→All)** | **55.66** 🏆 |
| Urban → Rural | **40.62** 🏆 |
| Rural → Urban | 53.52 |
| **ISPRS Potsdam** | **77.59** 🏆 |

*Variant used for reported results: VMamba-Small*

### Comparison Context

| Type | Backbone | LoveDA mIoU | U→R | R→U | Potsdam mIoU |
|---|---|---:|---:|---:|---:|
| CNN | DeepLabv3+ | 43.01 | 30.36 | 39.98 | 75.09 |
| Transformer | UNetFormer | 48.61 | 34.56 | 44.84 | 74.99 |
| **SSM** | **VMamba-Small** | **55.66** | **40.62** | 53.52 | **77.59** |
| **SSM** | MambaVision-L | 55.25 | 38.53 | **54.01** | 77.07 |
| SSM | Spatial-Mamba-B | 48.03 | 35.23 | 46.55 | 70.00 |

---

## 💡 Distinctive Findings

> **VMamba-Small wins the controlled fight.** 55.66 mIoU — +7.05 over the best Transformer, +12.65 over the best CNN. Same decoder, same training, no tricks. The cross-scan SSM provides global spatial context that neither local convolutions nor quadratic attention can match at this efficiency point.

VMamba-Small also achieves the strongest Urban→Rural transfer (40.62), suggesting its multi-directional scanning captures transferable spatial patterns across domain boundaries. On ISPRS Potsdam, it leads with 77.59 — confirming that the advantage holds on high-resolution, single-domain data as well.

---

## 📁 File Reference

| File | Purpose |
|---|---|
| `config.py` | LoveDA configuration |
| `config_icprs.py` | ISPRS Potsdam configuration |
| `encoders.py` | VMamba backbone wrapper (`RGBEncoder`) |
| `model.py` | Full model assembly (encoder + decoder) |
| `light_decoder.py` | Shared LightUNetDecoder |
| `losses.py` | TriBraid loss (Lovász + Focal + Boundary) |
| `train.py` | Iteration-based training loop |
| `train_domain.py` | Domain-shift training (single-domain) |
| `eval_domain.py` | Cross-domain evaluation |
| `dataset.py` | LoveDA dataset loader |
| `VMamba/` | Official VMamba implementation |

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
