# VisionMamba (UrbanMamba RGB)

RGB-only semantic segmentation model using a VisionMamba backbone and a lightweight U-Net decoder.

## Architecture

```
RGB Image (3ch) -> VisionMamba Encoder -> LightUNetDecoder -> Predictions
```

## Key Files

- `config.py` - Loveda configuration (RGB dataset).
- `config_icprs.py` - ISPRS Potsdam configuration.
- `dataset.py` - Loveda dataset loader.
- `encoders.py` - VisionMamba RGB encoder.
- `light_decoder.py` - Light U-Net decoder.
- `model.py` - Full model assembly.
- `train.py` - Iteration-based training loop.
- `train_domain.py` / `eval_domain.py` - Domain-specific training and evaluation helpers.

## Training

The training script imports `config_icprs.py` by default. To use Loveda instead, swap the import in `train.py` to:

```
from config import Config
```

Then update `DATA_ROOT`, `OUTPUT_DIR`, and `VISIONMAMBA_VARIANT` in the selected config file.

Run training:

```
python train.py
```

## Results

Outputs are written to the configured `OUTPUT_DIR` (checkpoints, logs, tensorboard, val_preds). These folders are gitignored by the root `.gitignore`.
