# 🚀 Mamba-Segmentation

**Controlled Visual State-Space Backbone Benchmark with Domain-Shift & Boundary Analysis for Remote-Sensing Segmentation**

### 🏆 The First Fair-Fight Benchmark for SSM vs. CNN vs. Transformer Backbones in Remote Sensing 🏆

[![📄 Paper](https://img.shields.io/badge/📄_IGRAAS_2026-Paper-blue)](https://doi.org/PLACEHOLDER)
[![🏆 Venue](https://img.shields.io/badge/🏆_IGRAAS_2026-Accepted-brightgreen)](https://doi.org/PLACEHOLDER)
[![🐍 Python](https://img.shields.io/badge/🐍_Python-3.9-3776AB)](https://www.python.org/)
[![🔥 PyTorch](https://img.shields.io/badge/🔥_PyTorch-2.0+-EE4C2C)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

One pipeline. One decoder. One loss. One schedule. **Five backbone families.** The only variable is the encoder — so the results finally mean something. SSMs dominate, scaling plateaus early, domain transfer is asymmetric, and boundaries are where every model breaks.

Ready to see which backbone actually wins a fair fight? Let's go.

---

[🔥 Updates](#-updates) • [🔭 Overview](#-overview) • [✨ Why Controlled?](#-why-controlled-benchmarking-matters) • [🧠 Pipeline](#-the-controlled-pipeline) • [⚡ Quick Start](#-quick-start) • [🗂 Data](#-data-preparation) • [🚀 Train & Eval](#-train--evaluation) • [🔬 Analysis](#-analysis-scripts) • [📊 Results](#-results) • [🙏 Acknowledgements](#-acknowledgements) • [📜 Cite](#-citation)

---

## 🔥🔥 Updates

| Date | Update |
|---|---|
| **Mar 2026** | Checkpoints + analysis notebooks released — pretrained weights for all five backbone families available |
| **Mar 2026** | **Paper Accepted** — IGRAAS 2026 (camera-ready submitted) |
| **Feb 2026** | Code released — full controlled training pipeline with per-backbone configs |
| **Jan 2026** | Analysis suite released — boundary, cross-domain, and rotation diagnostics |

---

## 🔭 Overview

Remote-sensing segmentation benchmarks have a fatal flaw: they change the backbone **and** the decoder **and** the loss **and** the schedule **and** the augmentations — all at once. The resulting numbers tell you who tuned harder, not which backbone is better.

**Mamba-Segmentation fixes this:**

- **Fixed lightweight U-Net decoder** → identical decoder across all experiments
- **Fixed TriBraid loss** (Lovász + Focal + Boundary) → same optimization objective for every backbone
- **Fixed training protocol** → 50k iterations, AdamW, poly LR, 512×512 crops, same augmentations
- **Standardized feature interface** → {F1, F2, F3, F4} at strides {4, 8, 16, 32}
- **Five backbone families** → VMamba, MambaVision, Spatial-Mamba, CNN (DeepLabv3), Transformer (UNetFormer)

**Outcome:** differences in results reflect backbone behavior. Nothing else.

<p align="center">
  <img src="IGARSS%202026/Architecture.png" alt="Controlled Pipeline Architecture" width="100%">
</p>
<p align="center"><i>Lock the pipeline. Swap the backbone. Read the truth. Three SSM families (Spatial-Mamba, MambaVision, VMamba) share a single U-Net decoder and standardized feature interface {F1–F4}.</i></p>

---

## ✨ Why Controlled Benchmarking Matters

Every backbone paper ships its own decoder, its own training recipe, its own augmentation policy. You compare "Method A" to "Method B" — but you're really comparing two *entire pipelines*.

Mamba-Segmentation isolates the **one variable that matters:**

| What | Status |
|---|---|
| Encoder backbone | 🔀 **Swapped** per experiment — the ONLY variable |
| Decoder architecture | 🔒 Fixed (lightweight U-Net, 256ch, MambaBlock2d) |
| Loss function | 🔒 Fixed (Lovász-Softmax + Focal + Boundary) |
| Training schedule | 🔒 Fixed (50k iters, AdamW, poly decay) |
| Augmentations | 🔒 Fixed (random crop, flip, color jitter) |
| Input resolution | 🔒 Fixed (512×512) |
| Feature interface | 🔒 Fixed ({F1–F4} at strides {4, 8, 16, 32}) |

When the results differ, you know *exactly* why.

---

## 🧠 The Controlled Pipeline

```
Encoder:     swapped per experiment — the ONLY variable
Decoder:     fixed lightweight U-Net (256ch, MambaBlock2d, addition skips)
Interface:   {F1, F2, F3, F4} at strides {4, 8, 16, 32}
Training:    50k iters · AdamW · poly LR decay · 512×512 crops · fixed augmentations
Loss:        L = L_lovász + L_focal + 0.5 × L_boundary
               ├─ Lovász-Softmax   → direct IoU optimization
               ├─ Focal (γ=2.0)    → class imbalance handling
               └─ Boundary (2px)   → edge penalty with warmup
```

**Backbone families tested:**

| Family | Backbones | Type |
|---|---|---|
| **VMamba** | Tiny, Small, Base | SSM — cross-scan 2D selective state-space |
| **MambaVision** | Tiny, Small, Base, Large, Large2 | SSM/Hybrid — Mamba + self-attention |
| **Spatial-Mamba** | Tiny, Small, Base | SSM — spatially-aware scanning |
| **DeepLabv3+** | ResNet-50 | CNN baseline |
| **UNetFormer** | ResNet-18 | Transformer baseline |

**Datasets:**
- **LoveDA** → All→All, Urban→Rural, Rural→Urban (source-only, zero adaptation)
- **ISPRS Potsdam** → high-resolution urban parsing (6-class)

---

## ⚡ Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/YOUR_USERNAME/Mamba-Segmentation
cd Mamba-Segmentation

conda create -n mamba-seg python=3.9 -y
conda activate mamba-seg

cd MambaVision && pip install -r requirements.txt
```

### 2. Grab Pre-trained Backbone Weights

| Backbone | Source | Location |
|---|---|---|
| VMamba (Tiny/Small/Base) | [VMamba repo](https://github.com/MzeroMiko/VMamba) | `VMamba/Vmamba_weights/ImageNet-1K/` |
| MambaVision (Tiny→Large2) | [NVIDIA MambaVision](https://github.com/NVlabs/MambaVision) | `MambaVision/weights/1k/` |
| Spatial-Mamba (Tiny/Small/Base) | [Spatial-Mamba repo](https://github.com/EdwardChaworworrachat/SpatialMamba) | `spatial-mamba/weights/imageNet1K/` |
| ResNet-50 / ResNet-18 | [torchvision](https://pytorch.org/vision/stable/models.html) | `weights/imagenet/` |

Set the weights path in each backbone's `config.py` — that's it.

### 3. Configure Your Experiment

Each backbone family has its own directory with a standardized interface:

```
<ModelFamily>/
├── config.py          # ← edit DATA_ROOT, OUTPUT_DIR, variant
├── config_icprs.py    # ← for ISPRS Potsdam experiments
├── train.py           # ← same training loop across all families
├── model.py
├── encoders.py
├── light_decoder.py   # ← THE fixed decoder (identical everywhere)
├── losses.py          # ← THE fixed loss (identical everywhere)
└── utils.py
```

---

## 🗂 Data Preparation

Plug-and-play support for **LoveDA** and **ISPRS Potsdam**.

<details>
<summary>📁 <b>LoveDA Layout</b></summary>

```
DATA_ROOT/
├── Train/
│   ├── Urban/
│   │   ├── images_png/
│   │   └── masks_png/
│   └── Rural/
│       ├── images_png/
│       └── masks_png/
├── Val/
│   ├── Urban/
│   │   ├── images_png/
│   │   └── masks_png/
│   └── Rural/
│       ├── images_png/
│       └── masks_png/
└── Test/
```

- **7 classes:** Background, Building, Road, Water, Barren, Forest, Agricultural
- **Resolution:** 1024×1024 (cropped to 512×512 during training)
- **Domains:** Urban and Rural — used for cross-domain evaluation

</details>

<details>
<summary>📁 <b>ISPRS Potsdam Layout</b></summary>

```
DATA_ROOT/
├── Images/
├── Labels/
└── splits/
    ├── train.txt
    ├── val.txt
    └── test.txt
```

- **6 classes:** Impervious, Building, Low Vegetation, Tree, Car, Clutter
- **Resolution:** 6000×6000 tiles (cropped to 512×512)

</details>

**Must-do:** Set `DATA_ROOT` in `config.py` (LoveDA) or `config_icprs.py` (Potsdam) to your local dataset path.

---

## 🚀 Train & Evaluation

YAML-free, config-driven — clean and reproducible.

### Train

```bash
# LoveDA — pick any backbone family
cd MambaVision                          # or VMamba/, spatial-mamba/, CNN_DeepLabv3p/, etc.
# → edit config.py: set DATA_ROOT, OUTPUT_DIR, and backbone variant
python train.py

# ISPRS Potsdam
cd VMamba
# → edit config_icprs.py: set DATA_ROOT and OUTPUT_DIR
python train.py
```

Checkpoints + TensorBoard logs land in `Comparison_Experiments/<experiment_name>/`.

### Efficiency Profiling

```bash
# Single model benchmark (FPS + peak VRAM)
python tools/benchmark_fps_mem.py \
  --model mambavision --variant base --device cuda:0

# Full sweep across all families
python tools/benchmark_fps_mem_total.py \
  --device cuda:0 --batch_size 1
```

---

## 🔬 Analysis Scripts

Three diagnostic scripts that reproduce every analytical claim in the paper:

| Script | What It Measures | What It Tells You |
|---|---|---|
| `analysis/boundary_analysis.py` | Boundary vs. interior mIoU under domain shift | Boundary degradation is the dominant failure mode — not interior misclassification |
| `analysis/cross_domain_analysis.py` | U→R and R→U metrics for all families | Domain transfer asymmetry is backbone-agnostic — it's a data property |
| `analysis/rotation_analysis.py` | Prediction stability under 90°/180°/270° rotations | Tests whether SSM scan-order introduces orientation artifacts |

```bash
python analysis/boundary_analysis.py \
  --device cuda:0 --use_pretrained 1

python analysis/cross_domain_analysis.py \
  --device cuda:0 --use_pretrained 1

python analysis/rotation_analysis.py \
  --device cuda:0 --use_pretrained 1 \
  --pack_rotations 1 \
  --families mambavision,vmamba,spatialmamba
```

Results land in `analysis_outputs/` as CSV files ready for plotting.

---

## 📊 Results

Straight from the paper — reproducible out of the box.

Every row shares the same decoder, loss, optimizer, schedule, augmentations, and data splits. **The only variable is the encoder backbone.**

| Type | Backbone | LoveDA mIoU | U→R | R→U | Potsdam mIoU |
|---|---|---:|---:|---:|---:|
| CNN | DeepLabv3 (controlled) | 43.01 | 30.36 | 39.98 | 75.09 |
| Transformer | UNetFormer (controlled) | 48.61 | 34.56 | 44.84 | 74.99 |
| **SSM** 🔥 | **VMamba-Small** | **55.66** | **40.62** | 53.52 | **77.59** |
| **SSM** 🔥 | **MambaVision-L** | 55.25 | 38.53 | **54.01** | 77.07 |
| SSM | Spatial-Mamba-B | 48.03 | 35.23 | 46.55 | 70.00 |

> 🏆 **VMamba-Small. 55.66 mIoU. +7.05 over the best Transformer. +12.65 over the best CNN. Same decoder. Same training. No tricks.**

### Accuracy vs. Throughput

<p align="center">
  <img src="IGARSS%202026/fps_vs_miou.png" alt="mIoU vs Inference Throughput" width="60%">
</p>
<p align="center"><i>mIoU (%) vs. inference throughput (FPS) for all SSM variants. VMamba holds near-peak accuracy across all sizes. MambaVision trades speed for capacity with diminishing returns. Spatial-Mamba sits in the lower tier.</i></p>

### Key Takeaways

🔥 **SSMs dominate the fair fight.** VMamba-Small beats UNetFormer by +7.05 and DeepLabv3 by +12.65 on LoveDA — under identical conditions. This is the backbone, not the pipeline.

📏 **Bigger ≠ better under a fixed decoder.** MambaVision-L carries far more parameters than VMamba-Small yet scores 55.25 vs. 55.66. Scaling the encoder past a threshold buys nothing when the decoder stays constant.

🔄 **Domain transfer is asymmetric — and backbone-agnostic.** Rural→Urban outperforms Urban→Rural by 10–15 points across every family. VMamba-Small: 53.52 R→U vs. 40.62 U→R. This is a data distribution property, not a model property.

🧱 **Boundaries are the unsolved failure mode.** Under domain shift, interior accuracy holds. Boundary accuracy collapses. Every backbone, every family, same story. Whoever cracks boundary sensitivity under distribution shift wins the next round.

### Qualitative Results — LoveDA

<p align="center">
  <img src="IGARSS%202026/loveda_qualitative_detailed_enhanced.png" alt="LoveDA Qualitative Results" width="85%">
</p>
<p align="center"><i>Predictions + error maps (magenta = false positive, dark green = false negative) on LoveDA Urban and Rural scenes. VMamba-S and VMamba-B produce the cleanest boundaries; Spatial-Mamba-B shows the most false positives at class transitions.</i></p>

### Qualitative Results — ISPRS Potsdam

<p align="center">
  <img src="IGARSS%202026/potsdam_qualitative_detailed_enhanced.png" alt="ISPRS Potsdam Qualitative Results" width="85%">
</p>
<p align="center"><i>Predictions + error maps on ISPRS Potsdam. All SSM variants handle large homogeneous regions well; errors concentrate at fine-grained boundaries (cars, narrow roads) — consistent with the boundary analysis findings.</i></p>

---

## 🧬 Backbone Overview

| Backbone | Architecture | Key Idea | RS Segmentation Impact |
|---|---|---|---|
| **VMamba** | Cross-scan 2D selective SSM | Global spatial context with linear complexity via multi-directional scanning | 🥇 Top performer: 55.66 LoveDA mIoU, strongest domain transfer |
| **MambaVision** | Hybrid Mamba + self-attention | Interleaves Mamba blocks (early stages) with attention (late stages) | Matches VMamba on Potsdam, but extra capacity doesn't help on LoveDA |
| **Spatial-Mamba** | Spatially-aware SSM | Explicit positional inductive biases in the state-space pathway | Beats CNN baseline, but scan-order alone insufficient without global modeling |
| **DeepLabv3+** | CNN (ResNet-50) | Atrous convolutions + ASPP for multi-scale context | Controlled CNN reference — 43.01 mIoU baseline |
| **UNetFormer** | Transformer (ResNet-18) | Efficient self-attention decoder for dense prediction | Controlled Transformer reference — 48.61 mIoU baseline |

---

## 🙏 Acknowledgements

This work builds on prior advances in visual state-space models and remote-sensing segmentation. We gratefully acknowledge:

- **[VMamba](https://github.com/MzeroMiko/VMamba)** — Visual State Space Model backbone
- **[MambaVision](https://github.com/NVlabs/MambaVision)** — NVIDIA's hybrid Mamba-Transformer architecture
- **[Spatial-Mamba](https://github.com/EdwardChaworworrachat/SpatialMamba)** — Spatially-aware Mamba variant
- **[LoveDA](https://github.com/Junjue-Wang/LoveDA)** and **[ISPRS Potsdam](https://www.isprs.org/education/benchmarks/UrbanSemLab/)** dataset creators

---

## 📜 Citation

If Mamba-Segmentation fuels your research, please cite:

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

🌍🛰️ Built at the **University of Peradeniya**. Got inspired? Give us a ⭐