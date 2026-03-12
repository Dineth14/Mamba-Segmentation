# 🚀 Mamba-Segmentation
## A Controlled Benchmark of Visual State-Space Backbones with Domain-Shift and Boundary Analysis for Remote-Sensing Segmentation

### 🏆 Controlled, Reproducible, Deployment-Oriented Benchmark for RS Segmentation 🏆

**Submitted to:** IGRAAS 2026  
**Status:** Pending Acceptance

**Authors:** Nichula Wasalathilaka, Dineth Perea, Oshadha Samarakoon, Buddhi Wijenayake, Roshan Godaliyadda, Vijitha Herath, Parakrama Ekanayake  
**Affiliation:** University of Peradeniya, Sri Lanka  
**Contact:** `{e20425,e21291,e21345,e19445,roshang,vijitha,mpbe}@eng.pdn.ac.lk`

Visual State-Space backbones benchmarked under a **strictly controlled segmentation pipeline** where only the encoder changes. The repository reports in-domain accuracy, source-only cross-domain robustness, boundary sensitivity, and practical efficiency.

`🔥 Updates` • `🔭 Overview` • `✨ Why Controlled Benchmarking?` • `🧠 Method` • `⚡ Quick Start` • `🗂 Data` • `🚀 Train & Eval` • `🧪 Analysis` • `📊 Results` • `🙏 Acknowledgements` • `📜 Cite`

## 🔥🔥 Updates

- **Mar 2026 - Documentation Refresh**  
  Repository docs reorganized and aligned with paper-style benchmark reporting.
- **Feb 2026 - Submission Update**  
  Manuscript submitted to **IGRAAS 2026** and currently **pending acceptance**.
- **Jan 2026 - Controlled Benchmark Pipeline Finalized**  
  Unified 4-stage encoder interface, fixed decoder, and standardized efficiency protocol completed.

Ready to compare backbones fairly under real domain shift? Let’s go.

## 🔭 Overview

Remote-sensing semantic segmentation must handle:

- Fine boundaries (roads, building edges, thin structures)
- Large-scene context
- Illumination and appearance variability
- Urban-rural domain shift

This repository isolates encoder effects by fixing everything else.

### Backbone families benchmarked

- **Visual SSMs:** `VMamba`, `MambaVision`, `Spatial-Mamba`
- **References:** CNN and Transformer encoders under the same decoder/protocol

### Key evaluation settings

- LoveDA: All->All, Urban->Rural, Rural->Urban (source-only, no adaptation)
- ISPRS Potsdam: controlled high-resolution urban parsing
- Unified efficiency protocol: Params/FLOPs/FPS/peak memory

## ✨ Why Controlled Benchmarking?

Many comparisons change multiple variables at once (decoder, training schedule, augmentations, interfaces), making conclusions unreliable.

This benchmark enforces:

- Same decoder for all backbones
- Same optimization schedule and augmentations
- Same input resolution and metric pipeline
- Same output feature interface `{F1,F2,F3,F4}` with strides `{4,8,16,32}`

So differences in results reflect **backbone behavior**, not pipeline drift.

## 🧠 Method in ~30 Seconds

1. Extract 4-stage features from each encoder (or project to unified interface).
2. Feed into the same lightweight U-Net style decoder.
3. Train with fixed optimization and augmentations.
4. Evaluate in-domain + cross-domain + boundary diagnostics.
5. Profile efficiency with a unified runtime/memory measurement setup.

### Loss

`L = L_lovasz + L_focal + 0.5 * L_boundary`

- Lovasz-Softmax: IoU-oriented optimization
- Focal: class imbalance mitigation
- Boundary term: edge-focused penalty (2-pixel neighborhood)

## ⚡ Quick Start

### 1. Environment

```bash
cd Mamba-Segmentation
conda create -n mamba-seg python=3.9 -y
conda activate mamba-seg
```

### 2. Install dependencies (example)

```bash
cd MambaVision
pip install -r requirements.txt
```

### 3. Configure and train

Edit `config.py` (LoveDA) or `config_icprs.py` (Potsdam):

- `DATA_ROOT`
- `OUTPUT_DIR`
- model variant / domain flags (if available)

Run:

```bash
python train.py
```

## 🗂 Data Preparation

### LoveDA

```text
DATA_ROOT/
├── Train/
│   ├── Urban/
│   │   ├── images_png/
│   │   └── masks_png/
│   └── Rural/
│       ├── images_png/
│       └── masks_png/
├── Val/
└── Test/
```

### ISPRS Potsdam

```text
DATA_ROOT/
├── Images/
├── Labels/
└── splits/
    ├── train.txt
    ├── val.txt
    └── test.txt
```

## 🚀 Train & Evaluation

### Example: LoveDA

```bash
cd MambaVision
# edit config.py
python train.py
```

### Example: ISPRS Potsdam

```bash
cd VMamba
# edit config_icprs.py
python train.py
```

### Efficiency profiling

```bash
cd ..
python tools/benchmark_fps_mem.py --model mambavision --variant base --device cuda:0
python tools/benchmark_fps_mem_total.py --device cuda:0 --batch_size 1
```

Outputs are saved under:

- `Comparison_Experiments/`
- `Comparison_Experiments_ICPRS_potsdam/`
- `analysis_outputs/`

## 🧪 Diagnostic Analysis

Run paper-aligned analysis scripts:

```bash
python analysis/boundary_analysis.py --device cuda:0 --use_pretrained 1
python analysis/cross_domain_analysis.py --device cuda:0 --use_pretrained 1
python analysis/rotation_analysis.py --device cuda:0 --use_pretrained 1 --pack_rotations 1 --families mambavision,vmamba,spatialmamba
```

These scripts support boundary-vs-interior diagnostics and robustness profiling.

## 📊 Results (from manuscript)

### Main findings

- Scaling inside a backbone family gives limited gains under fixed decoder constraints.
- Cross-domain robustness is asymmetric (Rural->Urban generally stronger than Urban->Rural).
- Boundary sensitivity is the dominant failure mode under domain shift.

### Representative LoveDA/Potsdam trends

| Type | Backbone | LoveDA mIoU | U->R | R->U | Potsdam mIoU |
| --- | --- | ---: | ---: | ---: | ---: |
| CNN | DeepLabv3 encoder (controlled) | 43.01 | 30.36 | 39.98 | 75.09 |
| Transformer | UNetFormer encoder (controlled) | 48.61 | 34.56 | 44.84 | 74.99 |
| SSM | VMamba-Small | 55.66 | 40.62 | 53.52 | 77.59 |
| SSM | MambaVision-L | 55.25 | 38.53 | 54.01 | 77.07 |
| SSM | Spatial-Mamba-B | 48.03 | 35.23 | 46.55 | 70.00 |

## 🙏 Acknowledgements

This benchmark builds on the broader open-source ecosystem and prior work in:

- Visual SSM backbones (VMamba, MambaVision, Spatial-Mamba)
- CNN and Transformer segmentation baselines
- LoveDA and ISPRS Potsdam datasets

## 📜 Citation

If this benchmark helps your research, please cite:

```bibtex
@article{wasalathilaka2026controlledbenchmark,
  title={A Controlled Benchmark of Visual State-Space Backbones with Domain-Shift and Boundary Analysis for Remote-Sensing Segmentation},
  author={Wasalathilaka, Nichula and Perea, Dineth and Samarakoon, Oshadha and Wijenayake, Buddhi and Godaliyadda, Roshan and Herath, Vijitha and Ekanayake, Parakrama},
  journal={Submitted to IGRAAS 2026},
  year={2026}
}
```

## 🌍🛰️ Inspired by fair benchmarking?

If this repository is useful for your work, please give it a STAR.
