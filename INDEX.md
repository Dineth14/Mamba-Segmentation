# Mamba-Segmentation Repository Index

Quick reference for navigating and using the Mamba-Segmentation repository.

## 📖 Documentation Files

| File | Purpose |
|------|---------|
| [README.md](README.md) | Main documentation and quick start |
| [STRUCTURE.md](STRUCTURE.md) | Detailed repository organization guide |
| [INDEX.md](INDEX.md) | This file — navigation reference |

## 🗂 Repository Layout

```
Mamba-Segmentation/
├── train.py                 ← unified training entry point
├── eval_domain.py           ← domain evaluation (All/Urban/Rural)
├── setup_backbones.sh       ← clone external backbone repos
│
├── core/                    ← shared model code
│   ├── model.py             ← SegmentationModel
│   ├── config_loader.py     ← YAML loader (env vars + CLI overrides)
│   ├── light_decoder.py
│   ├── dataset.py           ← LoveDA loader
│   ├── dataset_isprs.py     ← ISPRS Potsdam loader
│   ├── losses.py
│   ├── parts.py
│   └── utils.py
│
├── backbones/               ← thin RGBEncoder wrappers
│   ├── mambavision/
│   ├── vmamba/
│   ├── visionmamba/
│   ├── spatialmamba/
│   ├── swintransformer/
│   └── cnn/
│
├── configs/                 ← YAML configs
│   ├── base.yaml            ← shared defaults
│   ├── vmamba.yaml
│   ├── vmamba_potsdam.yaml
│   ├── mambavision.yaml
│   ├── mambavision_potsdam.yaml
│   ├── visionmamba.yaml
│   ├── visionmamba_potsdam.yaml
│   ├── spatialmamba.yaml
│   ├── spatialmamba_potsdam.yaml
│   ├── cnn_deeplabv3p.yaml
│   ├── cnn_unet.yaml
│   ├── transformer_swintiny.yaml
│   └── transformer_unetformer.yaml
│
├── analysis/                ← boundary/domain/rotation analysis scripts
├── analysis_outputs/        ← generated CSVs + plots (gitignored)
├── Comparison_Experiments/  ← LoveDA checkpoints (gitignored)
├── Comparison_Experiments_ICPRS_potsdam/  ← Potsdam checkpoints (gitignored)
├── tools/                   ← FPS/memory benchmarking utilities
└── weights/                 ← pre-trained backbone weights (gitignored)
```

## 🔌 Backbones

External backbone repos are cloned by `setup_backbones.sh` — they are **not** bundled in this repo.

| Backbone | Config | Clone path |
|----------|--------|-----------|
| MambaVision | `configs/mambavision.yaml` | `MambaVision/MambaVision/` |
| VMamba | `configs/vmamba.yaml` | `VMamba/VMamba/` |
| Vision Mamba (Vim) | `configs/visionmamba.yaml` | `VisionMamba/Vim/` |
| Spatial-Mamba | `configs/spatialmamba.yaml` | `spatial-mamba/Spatial-Mamba/` |
| Swin Transformer | `configs/transformer_swintiny.yaml` | `Swin-Transformer/` |
| CNN (ResNet-50) | `configs/cnn_deeplabv3p.yaml` | — (torchvision) |

## 📊 Datasets

| Dataset | Env var | Default path |
|---------|---------|-------------|
| LoveDA | `LOVEDA_ROOT` | `data/LoveDA` |
| ISPRS Potsdam | `POTSDAM_ROOT` | `data/Potsdam` |

## 🚀 Quick Workflows

### Setup
```bash
git clone https://github.com/Dineth14/Mamba-Segmentation
cd Mamba-Segmentation
bash setup_backbones.sh vmamba      # clone only what you need
conda create -n mamba-seg python=3.9 -y && conda activate mamba-seg
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install tensorboard tqdm pyyaml timm
```

### Train
```bash
export LOVEDA_ROOT=/path/to/LoveDA
python train.py --config configs/vmamba.yaml
python train.py --config configs/mambavision.yaml variant=large batch_size=2
```

### Evaluate
```bash
python eval_domain.py --config configs/vmamba.yaml --ckpt path/to/best.pth --domain rural
python eval_domain.py --config configs/vmamba.yaml --ckpt path/to/best.pth --domain all --append_csv results.csv
```

### Profile efficiency
```bash
python tools/benchmark_fps_mem.py --model vmamba --variant base
python tools/benchmark_fps_mem_total.py --device cuda:0
```

### Run analysis
```bash
python analysis/rotation_analysis.py --device cuda:0
python analysis/boundary_analysis.py --device cuda:0
```

## 📦 Pre-trained Weights

Checkpoints are hosted on HuggingFace: [dineth18/Mamba-Segmentation](https://huggingface.co/dineth18/Mamba-Segmentation)

## 🛠️ Tools

| Script | Purpose |
|--------|---------|
| `tools/benchmark_fps_mem.py` | Single model FPS + memory |
| `tools/benchmark_fps_mem_total.py` | All models comparison table |
| `tools/benchmark_allall_isolated.py` | Isolated benchmarking |
| `tools/eval_loveda_urban_rural.py` | Domain-split evaluation |
| `tools/plot_loveda_throughput_miou.py` | FPS vs mIoU plots |

## 🐛 Troubleshooting

| Issue | Solution |
|-------|----------|
| Dataset not found | Set `LOVEDA_ROOT` or `POTSDAM_ROOT` env var |
| Import error for backbone | Run `bash setup_backbones.sh <backbone>` |
| CUDA out of memory | Add `batch_size=2` CLI override |
| Wrong weights loaded | Check `weights_path` in config or pass `--ckpt` explicitly |

## 📝 Citation

**Paper:** Accepted at IGARSS 2026

---

**Last Updated:** May 2026

