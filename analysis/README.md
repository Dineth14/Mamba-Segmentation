# Analysis Experiments

This folder contains analysis-only scripts for boundary/interior IoU, rotation robustness, and cross-domain group IoU on LoveDA.

## Run All Analyses

From the `Mamba-Segmentation` root:

```bash
python run_all_analysis.py --device cuda:0 --batch 1 --num_workers 4 --use_pretrained 1 --pack_rotations 1 --families mambavision,vmamba,spatialmamba
```

## Run Individually

```bash
python analysis/boundary_analysis.py --device cuda:0 --use_pretrained 1
python analysis/rotation_analysis.py --device cuda:0 --use_pretrained 1 --pack_rotations 1 --families mambavision,vmamba,spatialmamba
python analysis/cross_domain_analysis.py --device cuda:0 --use_pretrained 1
```

## Outputs

All CSVs and plots are written to `analysis_outputs/` in the repository root:

- `boundary_vs_interior.csv`, `boundary_interior_iou.png`
- `rotation_robustness.csv`, `rotation_robustness.png`
- `cross_domain_groups.csv`, `cross_domain_groups.png`
- Log files per analysis run (e.g., `boundaryanalysis_YYYYMMDD_HHMMSS.log`)

## Notes

- Use `--data_root /path/to/LoveDA` to override dataset location.
- Scripts use the existing dataloaders/evaluators and respect `ignore_index=255`.
- Set `--use_pretrained 0` to skip loading pretrained encoder weights before loading `best.pth`.
- Set `--pack_rotations 0` if the rotation analysis OOMs; `1` is faster but uses 4x batch.
- Use `--families mambavision,vmamba,spatialmamba` to skip VisionMamba runs.
