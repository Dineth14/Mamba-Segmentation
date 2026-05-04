# 🗂 Mamba-Segmentation Structure
## Controlled Backbone Benchmark Layout (Paper-Aligned)

This file describes the repository organization.

`🔭 Overview` • `🏗 Core Modules` • `🔌 Backbones` • `⚙️ Configs` • `🧪 Experiments` • `⚡ Efficiency Tools` • `📊 Analysis` • `🧭 Naming Rules` • `🚀 Workflow`

## 🔭 Overview

The repository is organized around a single principle:

**Keep the segmentation pipeline fixed, vary only the encoder backbone.**

This allows fair comparison across visual SSM, CNN, and Transformer backbones.

## 🏗 Repository Layout

```text
Mamba-Segmentation/
├── train.py                        # Unified training entry point
├── eval_domain.py                  # Unified domain evaluation
├── setup_backbones.sh              # Clone external backbone repos
│
├── core/                           # Shared model code
│   ├── model.py                    # SegmentationModel (encoder + decoder)
│   ├── config_loader.py            # YAML config loader with env var + CLI overrides
│   ├── light_decoder.py            # LightUNetDecoder
│   ├── dataset.py                  # LoveDA dataset loader
│   ├── dataset_isprs.py            # ISPRS Potsdam dataset loader
│   ├── losses.py
│   ├── parts.py
│   └── utils.py
│
├── backbones/                      # Thin encoder wrappers (one per backbone family)
│   ├── mambavision/encoders.py
│   ├── vmamba/encoders.py
│   ├── visionmamba/encoders.py
│   ├── spatialmamba/encoders.py
│   ├── swintransformer/encoders.py
│   └── cnn/encoders.py
│
├── configs/                        # YAML configs (dataset paths via env vars)
│   ├── base.yaml                   # Shared defaults
│   ├── vmamba.yaml
│   ├── vmamba_potsdam.yaml
│   ├── mambavision.yaml
│   ├── mambavision_potsdam.yaml
│   ├── visionmamba.yaml
│   ├── spatialmamba.yaml
│   ├── cnn_deeplabv3p.yaml
│   ├── cnn_unet.yaml
│   ├── transformer_swintiny.yaml
│   └── transformer_unetformer.yaml
│
├── analysis/                       # Analysis scripts
├── analysis_outputs/               # Generated CSVs, PNGs, logs (gitignored)
├── Comparison_Experiments/         # LoveDA experiment checkpoints (gitignored)
├── Comparison_Experiments_ICPRS_potsdam/  # Potsdam checkpoints (gitignored)
├── tools/                          # Benchmarking utilities
├── weights/                        # Pre-trained backbone weights (gitignored)
└── Qualitative Analysis/           # Qualitative result assets
```

## 🔌 Backbone Wrappers

Each backbone family has a thin wrapper in `backbones/<name>/encoders.py` that exposes a single `RGBEncoder` class with an `out_channels` attribute. The wrapper adds the cloned backbone repo to `sys.path` at import time.

External backbone repos are **not** included in this repo. Clone them with:

```bash
bash setup_backbones.sh              # all backbones
bash setup_backbones.sh vmamba       # specific backbone only
```

Expected clone locations:

| Backbone | Clone path |
|----------|------------|
| MambaVision | `MambaVision/MambaVision/` |
| VMamba | `VMamba/VMamba/` |
| Vision Mamba (Vim) | `VisionMamba/Vim/` |
| Spatial-Mamba | `spatial-mamba/Spatial-Mamba/` |
| Swin Transformer | `Swin-Transformer/` |

## ⚙️ Config System

All configs use YAML with:
- `extends: base.yaml` for inheritance
- `${ENV_VAR:default}` for dataset paths (set `LOVEDA_ROOT` or `POTSDAM_ROOT`)
- CLI `key=value` overrides at runtime

```bash
export LOVEDA_ROOT=/path/to/LoveDA
python train.py --config configs/vmamba.yaml batch_size=4
```

## 🧪 Experiment Roots

Experiment directories are gitignored (checkpoints are large). Download weights from HuggingFace:

```text
Comparison_Experiments/              # LoveDA experiments  (gitignored)
Comparison_Experiments_ICPRS_potsdam/  # Potsdam experiments (gitignored)
```

Typical experiment folder layout:

```text
<experiment_name>/
├── checkpoints/best.pth
├── logs/
├── tensorboard/
└── val_preds/
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

analysis_outputs/    ← generated outputs (gitignored)
├── boundary_vs_interior.csv
├── cross_domain_groups.csv
└── rotation_robustness.csv
```

These modules support the paper's diagnostic claims on boundary sensitivity and domain-shift asymmetry.

## 🧭 Naming Rules

1. Use `snake_case` for new backbone config names and experiment folders.
2. Backbone wrappers live in `backbones/<lowercase_name>/`.
3. Store generated outputs only in gitignored dirs (`analysis_outputs/`, `Comparison_Experiments/`).
4. Use lowercase variant names in experiment folder names (e.g. `vmamba_small_512`).

## 🚀 Standard Workflow

1. Clone repo and run `bash setup_backbones.sh <backbone>`.
2. Set `LOVEDA_ROOT` or `POTSDAM_ROOT` environment variable.
3. Run `python train.py --config configs/<backbone>.yaml`.
4. Evaluate with `python eval_domain.py --config configs/<backbone>.yaml --ckpt <path>`.
5. Run analysis scripts and efficiency profiling.

## 📤 Publication Note

- Accepted at IGARSS 2026

## 🔗 Related Docs

- `README.md` — benchmark narrative and quick start
- `INDEX.md` — project navigation
- `tools/` — utility script details
- `analysis/` — analysis command reference
