# Mamba-Segmentation Repository Index

Quick reference guide for navigating and using the Mamba-Segmentation repository.

## 📖 Documentation Files

| File | Purpose |
|------|---------|
| [README.md](README.md) | Main repository documentation and quick start |
| [STRUCTURE.md](STRUCTURE.md) | Detailed repository organization guide |
| [CONTRIBUTING.md](CONTRIBUTING.md) | Guidelines for adding models and experiments |
| [INDEX.md](INDEX.md) | This file - navigation reference |

## 🔬 Model Implementations

### Mamba-Based Architectures

| Model | Directory | Description | Variants |
|-------|-----------|-------------|----------|
| **MambaVision** | [MambaVision/](MambaVision/) | Hybrid Mamba-Transformer from NVIDIA | tiny, small, base, large |
| **VMamba** | [VMamba/](VMamba/) | Visual state space with 2D-selective scanning | tiny, small, base |
| **VisionMamba (Vim)** | [VisionMamba/](VisionMamba/) | Bidirectional Mamba for vision | tiny, small, base |
| **SpatialMamba** | [spatial-mamba/](spatial-mamba/) | Spatially-aware Mamba variant | tiny, small, base |

**Start here:** Choose a model and read its [README.md](MambaVision/README.md) for architecture details.

### Baseline Architectures (CNN + Transformer)

| Model | Directory | Description |
|-------|-----------|-------------|
| **CNN-DeepLabv3+** | [CNN_DeepLabv3p/](CNN_DeepLabv3p/) | ResNet-50 + DeepLabv3+ decoder |
| **CNN-UNet** | [CNN_UNet/](CNN_UNet/) | ResNet-50 + U-Net decoder |
| **Swin Transformer** | [Swin-Transformer/](Swin-Transformer/) | Vision Transformer baseline |
| **TransformerSwinTiny** | [TransformerSwinTiny/](TransformerSwinTiny/) | Lightweight Swin variant |

## 📊 Datasets Supported

### LoveDA (Land-cOVEr Domain Adaptive)

- **Type:** RGB remote sensing imagery
- **Resolution:** 512×512 patches
- **Classes:** 7 semantic classes
- **Domains:** Urban and Rural scenes
- **Results:** [Comparison_Experiments/](Comparison_Experiments/README.md)
- **Config File:** `config.py` in each model directory

### ISPRS Potsdam (2D Semantic Labeling)

- **Type:** RGB-Infrared (IRRG) imagery
- **Resolution:** 512×512 patches
- **Classes:** 6 semantic classes
- **Results:** [Comparison_Experiments_ICPRS_potsdam/](Comparison_Experiments_ICPRS_potsdam/README.md)
- **Config File:** `config_icprs.py` in each model directory

## 🚀 Getting Started

### 1. **Quick Setup**
```bash
cd Mamba-Segmentation
conda create -n mamba-seg python=3.9
conda activate mamba-seg
```

### 2. **Prepare Dataset**
- Download [LoveDA](http://loveda.rsvision.org/) or [ISPRS Potsdam](http://www2.isprs.org/commissions/comm3/wg4/potsdam-2d-semantic-labeling.html)
- Extract and note the path

### 3. **Choose a Model**
```bash
cd MambaVision  # or VMamba, VisionMamba, etc.
pip install -r requirements.txt
```

### 4. **Configure Training**
- Edit `config.py` (LoveDA) or `config_icprs.py` (Potsdam)
- Set `DATA_ROOT` to your dataset path
- Set `OUTPUT_DIR` for results

### 5. **Train**
```bash
python train.py
```

**See:** [README.md](README.md) for detailed quick start

## 🛠️ Tools & Analysis

| Tool | File | Purpose |
|------|------|---------|
| **Benchmarking** | [tools/](tools/README.md) | FPS, memory, and efficiency metrics |
| **Analysis Scripts** | [analysis/](analysis/README.md) | Boundary/rotation/domain analysis |
| **Evaluation** | [tools/eval_loveda_urban_rural.py](tools/) | Domain-specific evaluation |
| **Visualization** | [tools/plot_loveda_throughput_miou.py](tools/) | Performance plots |

**Quick command:**
```bash
python tools/benchmark_fps_mem.py --model mambavision --variant base
```

## 📈 Experiment Results

### LoveDA Results
- Directory: [Comparison_Experiments/](Comparison_Experiments/)
- Documentation: [Comparison_Experiments/README.md](Comparison_Experiments/README.md)
- Metrics: CSV files with FPS, memory, and IoU
- Naming: `[model]_[variant]_[resolution]`

### ISPRS Potsdam Results
- Directory: [Comparison_Experiments_ICPRS_potsdam/](Comparison_Experiments_ICPRS_potsdam/)
- Documentation: [Comparison_Experiments_ICPRS_potsdam/README.md](Comparison_Experiments_ICPRS_potsdam/README.md)
- Results: Per-experiment directories with metrics

### Analysis Outputs
- Directory: [analysis_outputs/](analysis_outputs/)
- Contains: CSV results and PNG visualizations from analysis scripts

## 📦 Weights

### Pre-trained Weights
- **Location:** [weights/](weights/README.md)
- **ImageNet:** ResNet-18, ResNet-50
- **Model-specific:** Auto-downloaded during training
- **Fine-tuned:** Stored in experiment directories

## 🤝 Contributing

**Want to add a new model?** → See [CONTRIBUTING.md](CONTRIBUTING.md)

Quick checklist:
1. Create `[ModelName]/` directory
2. Add required files (train.py, config.py, model.py, etc.)
3. Write [ModelName]/README.md
4. Test on LoveDA and/or Potsdam
5. Document results in CSV format

## 📝 Citation

If using this repository, cite:
- The respective Mamba architecture papers
- LoveDA and ISPRS Potsdam dataset papers
- This benchmarking framework

**Paper Status:** Submitted to IGRAAS 2026 (Pending Acceptance)

## 🐛 Troubleshooting

| Issue | Solution |
|-------|----------|
| Dataset not found | Check `DATA_ROOT` in config file |
| CUDA out of memory | Reduce `BATCH_SIZE` in config |
| Import errors | Install requirements: `pip install -r requirements.txt` |
| Model not downloading | Check internet connection and official repo access |

See individual model READMEs for model-specific issues.

## 📚 Full Documentation Map

```
Repository Root/
├── README.md ..................... Main entry point
├── STRUCTURE.md .................. Repository organization (detailed)
├── CONTRIBUTING.md ............... Adding new models
├── INDEX.md (this file) .......... Navigation reference
│
├── MambaVision/README.md ......... MambaVision specifics
├── VMamba/README.md .............. VMamba specifics
├── VisionMamba/README.md ......... VisionMamba specifics
├── spatial-mamba/README.md ....... SpatialMamba specifics
│
├── Comparison_Experiments/README.md ........ LoveDA results guide
├── Comparison_Experiments_ICPRS_potsdam/README.md .. Potsdam results
├── analysis/README.md ................. Analysis scripts guide
├── tools/README.md ................... Benchmarking tools
└── weights/README.md ................. Pre-trained weights
```

## 🔍 Common Workflows

### Training MambaVision on LoveDA
```bash
cd MambaVision
# Edit config.py: set DATA_ROOT
python train.py
```

### Training VMamba on Potsdam
```bash
cd VMamba
# Edit config_icprs.py: set DATA_ROOT
python train.py
```

### Benchmark All Models
```bash
python tools/benchmark_fps_mem_total.py --device cuda:0
```

### Analyze Robustness
```bash
cd analysis
python rotation_analysis.py --device cuda:0
```

### Compare FPS vs mIoU
```bash
python tools/plot_loveda_throughput_miou.py --csv_path Comparison_Experiments/fps_mem_allall.csv
```

## 📞 Quick Links

- **Download LoveDA:** http://loveda.rsvision.org/
- **Download ISPRS Potsdam:** http://www2.isprs.org/commissions/comm3/wg4/potsdam-2d-semantic-labeling.html
- **MambaVision Official:** https://github.com/NVlabs/MambaVision
- **VMamba Official:** https://github.com/MzeroMiko/VMamba
- **VisionMamba Official:** https://github.com/hustvl/Vim

## 💡 Tips

- Start with a smaller model variant (tiny) to test your setup
- Set `BATCH_SIZE = 1` if running on limited GPU memory
- Use `--data_root` argument to override config file dataset path
- Save the `.gitignore` exclusions - don't commit large output directories
- Check `training_log.txt` in experiment directories for detailed logs

---

**Last Updated:** February 2026  
**Paper Status:** Submitted to IGRAAS 2026 (Pending Acceptance)

