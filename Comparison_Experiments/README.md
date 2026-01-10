# Comparison Experiments - LoveDA Dataset

This directory contains experimental results for various Mamba-based segmentation models trained on the **LoveDA** remote sensing dataset.

## Dataset
- **Dataset:** LoveDA (Land-cOVEr Domain Adaptive semantic segmentation)
- **Task:** Semantic segmentation of RGB remote sensing imagery
- **Resolution:** 512×512 patches
- **Training Variants:** 
  - Standard training (all data)
  - Rural-only training
  - Urban-only training

## Models

### MambaVision Models
- **mambavision_tiny_512** - Smallest MambaVision variant (standard training)
- **mambavision_tiny_ruraltrain_512** - Tiny variant trained on rural subset
- **mambavision_tiny_urbantrain_512** - Tiny variant trained on urban subset
- **mambavision_tiny2_512** - Alternative tiny configuration
- **mambavision_tiny2_ruraltrain_512** - Alternative tiny (rural)
- **mambavision_tiny2_urbantrain_512** - Alternative tiny (urban)
- **mambavision_small_512** - Small MambaVision variant
- **mambavision_small_ruraltrain_512** - Small (rural)
- **mambavision_small_urbantrain_512** - Small (urban)
- **mambavision_base_512** - Base MambaVision variant
- **mambavision_base_ruraltrain_512** - Base (rural)
- **mambavision_base_urbantrain_512** - Base (urban)
- **mambavision_large_512** - Large MambaVision variant
- **mambavision_large_ruraltrain_512** - Large (rural)
- **mambavision_large_urbantrain_512** - Large (urban)
- **mambavision_large2_512** - Alternative large configuration
- **mambavision_large2_ruraltrain_512** - Alternative large (rural)
- **mambavision_large2_urbantrain_512** - Alternative large (urban)

### VMamba Models
- **vmamba_tiny_ruraltrain_512** - VMamba tiny (rural training)
- **vmamba_tiny_urbantrain_512** - VMamba tiny (urban training)
- **vmamba_small_ruraltrain_512** - VMamba small (rural training)
- **vmamba_small_urbantrain_512** - VMamba small (urban training)
- **vmamba_base_ruraltrain_512** - VMamba base (rural training)
- **vmamba_base_urbantrain_512** - VMamba base (urban training)
- **Vmamb_tiny_512** - VMamba tiny (standard training)
- **Vmamb_small_512** - VMamba small (standard training)
- **Vmamb_base_512** - VMamba base (standard training)

### VisionMamba Models
- **VisionMamba_tiny_512** - VisionMamba tiny (standard training)
- **VisionMamba_small_512** - VisionMamba small (standard training)
- **VisionMamba_base_512** - VisionMamba base (standard training)
- **visionmamba_tiny_ruraltrain_512** - VisionMamba tiny (rural)
- **visionmamba_tiny_urbantrain_512** - VisionMamba tiny (urban)
- **visionmamba_small_ruraltrain_512** - VisionMamba small (rural)
- **visionmamba_small_urbantrain_512** - VisionMamba small (urban)
- **visionmamba_base_ruraltrain_512** - VisionMamba base (rural)
- **visionmamba_base_urbantrain_512** - VisionMamba base (urban)

### SpatialMamba Models
- **spatialmamba_tiny_512** - SpatialMamba tiny (standard training)
- **spatialmamba_tiny_ruraltrain_512** - SpatialMamba tiny (rural)
- **spatialmamba_tiny_urbantrain_512** - SpatialMamba tiny (urban)
- **spatialmamba_small_512** - SpatialMamba small (standard training)
- **spatialmamba_small_ruraltrain_512** - SpatialMamba small (rural)
- **spatialmamba_small_urbantrain_512** - SpatialMamba small (urban)
- **spatialmamba_base_512** - SpatialMamba base (standard training)
- **spatialmamba_base_ruraltrain_512** - SpatialMamba base (rural)
- **spatialmamba_base_urbantrain_512** - SpatialMamba base (urban)

## Directory Structure

Each experiment directory contains:
- `checkpoints/` - Saved model weights (gitignored)
- `logs/` - Training logs (gitignored)
- `tensorboard/` - TensorBoard event files (gitignored)
- `val_preds/` - Validation predictions (gitignored)
- `*.log` - Training log files (gitignored)

## Results

Results are stored locally in each experiment directory but are excluded from git tracking. To reproduce results, run the training scripts with appropriate configurations.

## Usage

To analyze or continue training from any experiment:
1. Navigate to the model directory (e.g., `MambaVision/`, `VMamba/`, etc.)
2. Update the config file to point to the desired experiment output directory
3. Run the training script with appropriate parameters

## Notes

- All experiments use 512×512 input resolution
- Training variants (rural/urban) allow domain-specific model evaluation
- Standard training uses the full LoveDA training set
