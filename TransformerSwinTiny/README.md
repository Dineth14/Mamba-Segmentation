# TransformerSwinTiny

Controlled Transformer baseline for remote-sensing semantic segmentation in this repository.

## Purpose

This module provides a Swin-Tiny based segmentation baseline under the same controlled setup used across all backbones:

- fixed lightweight decoder
- fixed loss function (TriBraid loss)
- fixed training schedule and augmentations
- fixed feature interface for fair comparison

## Key Files

- `train.py` - standard training loop
- `train_domain.py` - Urban->Rural and Rural->Urban domain-shift training wrapper
- `config.py` - LoveDA settings
- `config_icprs.py` - ISPRS Potsdam settings
- `model.py` - model assembly
- `light_decoder.py` - fixed decoder implementation
- `losses.py` - controlled loss stack

## Quick Start

```bash
cd TransformerSwinTiny
# Edit config.py or config_icprs.py
python train.py
```

## Expected Outputs

Training outputs are written to experiment folders under:

- `Comparison_Experiments/` for LoveDA
- `Comparison_Experiments_ICPRS_potsdam/` for Potsdam
