# Mamba-Segmentation

A comprehensive comparison of Mamba-based architectures for semantic segmentation on remote sensing datasets. This repository benchmarks multiple Mamba variants (MambaVision, VMamba, VisionMamba, and SpatialMamba) on LoveDA and ISPRS Potsdam datasets.

## 🏗️ Architecture Overview

This repository implements semantic segmentation using various Mamba-style backbones paired with a lightweight U-Net decoder:

### Model Backbones

- **[MambaVision](MambaVision/)** - Hybrid Mamba-Transformer architecture with efficient vision processing
- **[VMamba](VMamba/)** - Visual State Space Model with 2D-selective scanning
- **[VisionMamba](VisionMamba/)** - Bidirectional Mamba for vision tasks with domain adaptation support
- **[spatial-mamba](spatial-mamba/)** - Spatial-aware Mamba architecture (UrbanMamba RGB variant)

Each backbone is integrated with a LightUNetDecoder for multi-scale feature fusion and dense prediction.

## 📊 Datasets

### LoveDA (Land-cOVEr Domain Adaptive)
- **Type:** RGB remote sensing imagery
- **Task:** Multi-class semantic segmentation
- **Resolution:** 512×512 patches
- **Domains:** Urban and Rural scenes
- **Configuration:** `config.py` in each model directory

### ISPRS Potsdam
- **Type:** Urban aerial imagery (RGB + Infrared)
- **Task:** 6-class semantic segmentation
- **Resolution:** 512×512 patches (from 6000×6000 tiles)
- **Classes:** Impervious surfaces, Building, Low vegetation, Tree, Car, Clutter
- **Configuration:** `config_icprs.py` in each model directory

### Dataset Structure

**LoveDA:**
```
DATA_ROOT/
  Train/
    Urban/
      images_png/
      masks_png/
    Rural/
      images_png/
      masks_png/
  Val/
    ...
  Test/
    ...
```

**ISPRS Potsdam:**
```
DATA_ROOT/
  Images/
  Labels/
  splits/
    train.txt
    val.txt
    test.txt
```

## 🚀 Quick Start

### Prerequisites
```bash
# Create environment
conda create -n mamba-seg python=3.9
conda activate mamba-seg

# Install dependencies (example for MambaVision)
cd MambaVision
pip install -r requirements.txt
```

### Training

1. **Choose a model architecture:**
   ```bash
   cd MambaVision  # or VMamba, VisionMamba, spatial-mamba
   ```

2. **Configure dataset and training:**
   - For LoveDA: Edit `config.py`
   - For ISPRS Potsdam: Edit `config_icprs.py`
   
   Update key parameters:
   ```python
   DATA_ROOT = '/path/to/dataset'
   OUTPUT_DIR = '../Comparison_Experiments/mambavision_base_512'
   VARIANT = 'base'  # or 'tiny', 'small', 'large'
   ```

3. **Run training:**
   ```bash
   python train.py
   ```

### Example Workflows

**Train MambaVision-Base on LoveDA:**
```bash
cd MambaVision
# Edit config.py: set DATA_ROOT and OUTPUT_DIR
python train.py
```

**Train VMamba-Small on ISPRS Potsdam:**
```bash
cd VMamba
# Edit config_icprs.py: set DATA_ROOT and OUTPUT_DIR
python train.py
```

**Domain-specific training (Rural only):**
```bash
# Edit config to set TRAIN_ON_RURAL=True, TRAIN_ON_URBAN=False
python train.py
```

## 📁 Repository Structure

```
Mamba-Segmentation/
├── MambaVision/              # MambaVision implementation
│   ├── train.py
│   ├── config.py             # LoveDA config
│   ├── config_icprs.py       # Potsdam config
│   ├── dataset.py            # LoveDA dataloader
│   ├── dataset_isprs.py      # Potsdam dataloader
│   └── ...
├── VMamba/                   # VMamba implementation
├── VisionMamba/              # VisionMamba implementation  
├── spatial-mamba/            # SpatialMamba implementation
├── Comparison_Experiments/   # LoveDA experiment results
│   ├── README.md             # Detailed experiment documentation
│   ├── mambavision_base_512/
│   ├── vmamba_small_512/
│   └── ...
├── Comparison_Experiments_ICPRS_potsdam/  # Potsdam results
│   ├── README.md
│   ├── mambavision_large_512/
│   └── ...
├── .gitignore                # Excludes checkpoints/logs
└── README.md                 # This file
```

## 🧪 Experiments

This repository contains extensive experimental comparisons across:
- **4 architectures** (MambaVision, VMamba, VisionMamba, SpatialMamba)
- **4 model sizes** (tiny, small, base, large)
- **3 training strategies** (full dataset, rural-only, urban-only)
- **2 datasets** (LoveDA, ISPRS Potsdam)

See [Comparison_Experiments/README.md](Comparison_Experiments/README.md) and [Comparison_Experiments_ICPRS_potsdam/README.md](Comparison_Experiments_ICPRS_potsdam/README.md) for detailed experiment documentation and model configurations.

## 📈 Results and Outputs

Each experiment produces:
- **Checkpoints** (`checkpoints/`) - Model weights at each epoch
- **Logs** (`logs/`) - Detailed training logs
- **TensorBoard** (`tensorboard/`) - Training curves and metrics
- **Predictions** (`val_preds/`) - Validation set predictions

**Note:** Result directories are gitignored to keep the repository lightweight. Only code and configurations are tracked.

## 🔧 Configuration

Key configuration parameters in `config.py` / `config_icprs.py`:

```python
# Dataset
DATA_ROOT = '/path/to/dataset'
BATCH_SIZE = 8
NUM_WORKERS = 4

# Model
VARIANT = 'base'  # 'tiny', 'small', 'base', 'large', 'large2', 'tiny2'
IN_CHANNELS = 3   # 3 for RGB, 4 for RGBIR

# Training
EPOCHS = 100
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4

# Output
OUTPUT_DIR = '../Comparison_Experiments/model_name'

# Domain-specific (LoveDA only)
TRAIN_ON_URBAN = True
TRAIN_ON_RURAL = True
```

## 🛠️ Development

### Adding New Models
1. Create model directory with `train.py`, `config.py`, and dataset loaders
2. Implement backbone integration with decoder
3. Add experiment output directory to `.gitignore`

### Git Workflow
```bash
# Results are automatically ignored
git add .
git commit -m "Add new experiment configuration"
git push

# To include submodules
git submodule update --init --recursive
```

## 📝 Citation

If you use this code in your research, please cite the respective papers for:
- MambaVision
- VMamba  
- VisionMamba (Vim)
- SpatialMamba / UrbanMamba
- LoveDA dataset
- ISPRS Potsdam dataset

## 📄 License

See individual model directories for specific licenses. This wrapper code is provided as-is for research purposes.

## 🤝 Contributing

Contributions welcome! Please:
1. Keep result directories out of commits (use .gitignore)
2. Document new models in this README
3. Add experiment documentation to Comparison_Experiments/README.md
4. Follow existing code structure and naming conventions

## 🐛 Issues

For bugs or questions:
1. Check existing issues
2. Provide full error trace and configuration
3. Specify dataset, model variant, and environment details

