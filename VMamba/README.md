# VMamba (UrbanMamba RGB)

RGB-only semantic segmentation model using a VMamba backbone and a lightweight U-Net decoder.

## Architecture

```
RGB Image (3ch) -> VMamba Encoder -> LightUNetDecoder -> Predictions
```

## Key Files

- `config.py` - Loveda configuration (RGB dataset).
- `config_icprs.py` - ISPRS Potsdam configuration.
- `dataset.py` - Loveda dataset loader.
- `encoders.py` - VMamba RGB encoder.
- `light_decoder.py` - Light U-Net decoder.
- `model.py` - Full model assembly.
- `train.py` - Iteration-based training loop.

## Training

The training script imports `config_icprs.py` by default. To use Loveda instead, swap the import in `train.py` to:

```
from config import Config
```

Then update `DATA_ROOT`, `OUTPUT_DIR`, and `VMAMBA_VARIANT` in the selected config file.

Run training:

```
python train.py
```

## Results

Outputs are written to the configured `OUTPUT_DIR` (checkpoints, logs, tensorboard, val_preds). These folders are gitignored by the root `.gitignore`.
