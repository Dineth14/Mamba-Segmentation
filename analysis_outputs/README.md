# Analysis Outputs

This folder stores generated artifacts from the analysis scripts in [analysis/](../analysis/):

- `boundary_vs_interior.csv` and `boundary_interior_iou.png`
- `cross_domain_groups.csv` and `cross_domain_groups.png`
- `rotation_robustness.csv` and `rotation_robustness.png`
- timestamped `*.log` files from analysis runs

## How to Regenerate

Run the scripts below from the repository root:

```bash
python analysis/boundary_analysis.py --device cuda:0 --use_pretrained 1
python analysis/cross_domain_analysis.py --device cuda:0 --use_pretrained 1
python analysis/rotation_analysis.py --device cuda:0 --use_pretrained 1 --pack_rotations 1 --families mambavision,vmamba,spatialmamba
```

## Notes

- These are reproducible outputs, not source code.
- Large intermediate artifacts should not be committed unless they are required for paper-ready figures.
