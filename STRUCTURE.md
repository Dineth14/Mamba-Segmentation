# 🗂 Mamba-Segmentation Structure
## Controlled Backbone Benchmark Layout (Paper-Aligned)

This file describes the repository organization in the same style as the project README.

`🔭 Overview` • `🏗 Core Modules` • `🧪 Experiments` • `⚡ Efficiency Tools` • `📊 Analysis` • `🧭 Naming Rules` • `🚀 Workflow`

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

## 🧭 Naming Rules

To keep the repository consistent and maintainable:

1. Use stable folder names in `snake_case` or established model names without spaces.
2. Keep model family folders at repository root (for example `VMamba/`, `MambaVision/`, `TransformerSwinTiny/`).
3. Store generated outputs only in experiment or output folders, not source directories.
4. Keep per-folder `README.md` files updated when adding scripts, configs, or assets.
5. Use lowercase variant names in experiment folder names (for example `vmamba_small_512`).

Current legacy names kept for backward compatibility:

- `Qualitative Analysis/` (space in name)
- Mixed historical prefixes in experiments (for example `Vmamb_*` and `VisionMamba_*`)

Use normalized names for new experiments and new folders.

## 🚀 Standard Workflow

1. Pick backbone family + variant.
2. Configure data/output paths.
3. Train under fixed protocol.
4. Evaluate All->All and cross-domain (U->R, R->U).
5. Run boundary and robustness analysis.
6. Profile efficiency metrics.

## 📤 Publication Note

- Accepted at IGRAAS 2026

## 🔗 Related Docs

- `README.md` - benchmark narrative and quick start
- `INDEX.md` - project navigation
- `CONTRIBUTING.md` - extension guidelines
- `tools/README.md` - utility script details
- `analysis/README.md` - analysis command reference
- `analysis_outputs/README.md` - generated analysis artifacts
- `TransformerSwinTiny/README.md` - Transformer Swin-Tiny baseline guide
- `Qualitative Analysis/README.md` - qualitative notebooks and assets
