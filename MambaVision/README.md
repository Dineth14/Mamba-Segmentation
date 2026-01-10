# MambaVision (UrbanMamba RGB)

Production-grade RGB-only semantic segmentation model using a MambaVision backbone and a lightweight U-Net decoder.

## Architecture

```
RGB Image (3ch) -> MambaVision Encoder -> LightUNetDecoder -> Predictions
```

## Key Files

- `config.py` - Loveda configuration (RGB dataset).
- `config_icprs.py` - ISPRS Potsdam configuration.
- `dataset.py` - Loveda dataset loader.
- `dataset_isprs.py` - ISPRS Potsdam dataset loader.
- `encoders.py` - MambaVision RGB encoder.
- `light_decoder.py` - Light U-Net decoder.
- `model.py` - Full model assembly.
- `train.py` - Iteration-based training loop.

## Training

The training script imports `config_icprs.py` by default. To use Loveda instead, swap the import in `train.py` to:

```
from config import Config
```

Then adjust `DATA_ROOT`, `OUTPUT_DIR`, and `MAMBAVISION_VARIANT` in the chosen config file.

Run training:

```
python train.py
```

## Results

Outputs are written to the configured `OUTPUT_DIR` (checkpoints, logs, tensorboard, val_preds). These folders are gitignored by the root `.gitignore`.
