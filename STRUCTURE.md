# 🗂 Mamba-Segmentation Structure
## Controlled Backbone Benchmark Layout (Paper-Aligned)

This file describes the repository organization in the same style as the project README.

`🔭 Overview` • `🏗 Core Modules` • `🧪 Experiments` • `⚡ Efficiency Tools` • `📊 Analysis` • `🚀 Workflow`

## 🔭 Overview

The repository is organized around a single principle:

**Keep the segmentation pipeline fixed, vary only the encoder backbone.**

This allows fair comparison across visual SSM, CNN, and Transformer backbones.

## 🏗 Core Modules

```text
Mamba-Segmentation/
├── MambaVision/
├── VMamba/
├── VisionMamba/
├── spatial-mamba/
├── CNN_DeepLabv3p/
├── CNN_UNet/
├── Swin-Transformer/
├── TransformerSwinTiny/
└── TransformerUNetFormer/
```

### Typical model directory interface

```text
<ModelFamily>/
├── train.py
├── config.py
├── config_icprs.py
├── dataset.py
├── dataset_isprs.py
├── model.py
├── encoders.py
├── light_decoder.py
├── losses.py
└── utils.py
```

## 🧪 Experiment Roots

```text
Mamba-Segmentation/
├── Comparison_Experiments/                  # LoveDA experiments
└── Comparison_Experiments_ICPRS_potsdam/    # ISPRS Potsdam experiments
```

Typical experiment folder:

```text
<experiment_name>/
├── checkpoints/
├── logs/
├── tensorboard/
├── val_preds/
└── *.log
```

## ⚡ Efficiency Tools

```text
tools/
├── benchmark_fps_mem.py
├── benchmark_fps_mem_total.py
├── benchmark_allall_isolated.py
├── eval_loveda_urban_rural.py
└── plot_loveda_throughput_miou.py
```

Used to report Params/FLOPs/FPS/peak memory under consistent settings.

## 📊 Analysis Modules

```text
analysis/
├── boundary_analysis.py
├── cross_domain_analysis.py
├── rotation_analysis.py
└── analysis_utils.py

analysis_outputs/
├── boundary_vs_interior.csv
├── cross_domain_groups.csv
└── rotation_robustness.csv
```

These modules support the paper's diagnostic claims on boundary sensitivity and domain-shift asymmetry.

## 🚀 Standard Workflow

1. Pick backbone family + variant.
2. Configure data/output paths.
3. Train under fixed protocol.
4. Evaluate All->All and cross-domain (U->R, R->U).
5. Run boundary and robustness analysis.
6. Profile efficiency metrics.

## 📤 Publication Note

- Submitted to IGRAAS 2026
- Pending acceptance

## 🔗 Related Docs

- `README.md` - benchmark narrative and quick start
- `INDEX.md` - project navigation
- `CONTRIBUTING.md` - extension guidelines
- `tools/README.md` - utility script details
- `analysis/README.md` - analysis command reference
