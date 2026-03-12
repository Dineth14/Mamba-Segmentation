# Tools Directory

This directory contains utility scripts for benchmarking, evaluation, and analysis of Mamba-based segmentation models.

## Benchmarking Tools

### Performance Benchmarking

**`benchmark_fps_mem.py`** - Measures FPS (frames per second) and memory consumption for model inference
- Records frames per second and peak GPU memory
- Useful for efficiency comparisons across architectures
- Output: CSV files with FPS and memory metrics

**`benchmark_fps_mem_total.py`** - Comprehensive benchmarking script
- Tests all model variants across datasets
- Produces aggregate statistics across families
- Outputs to: `Comparison_Experiments/fps_mem_total_allall.csv`

**`benchmark_allall_isolated.py`** - Isolated benchmarking for individual models
- Runs each model in isolation to prevent memory carryover
- More accurate memory measurements
- Outputs to: `Comparison_Experiments/fps_mem_allall_isolated.csv`

**`bench_worker.py`** - Worker module for parallel benchmarking
- Supports distributed benchmarking across multiple GPUs/nodes
- Used by other benchmark scripts for parallelization

## Evaluation Tools

**`eval_loveda_urban_rural.py`** - Domain-specific evaluation for LoveDA dataset
- Evaluates model performance separately on Urban and Rural domains
- Computes per-domain IoU and mIoU metrics
- Useful for analyzing domain adaptation and generalization

## Visualization Tools

**`plot_loveda_throughput_miou.py`** - Performance vs Accuracy visualization
- Plots models on 2D space: FPS (throughput) vs mIoU (accuracy)
- Creates Pareto frontier visualization
- Helps identify efficiency trade-offs between architectures

## Usage Examples

### Run Full Benchmark Suite
```bash
cd Mamba-Segmentation
python tools/benchmark_fps_mem_total.py --device cuda:0 --batch_size 1
```

### Benchmark Individual Model
```bash
python tools/benchmark_fps_mem.py --model mambavision --variant base --device cuda:0
```

### Domain-Specific Evaluation
```bash
python tools/eval_loveda_urban_rural.py \
  --model_path Comparison_Experiments/mambavision_base_512/best.pth \
  --data_root /path/to/loveda
```

### Generate Performance Plot
```bash
python tools/plot_loveda_throughput_miou.py \
  --csv_path Comparison_Experiments/fps_mem_total_allall.csv
```

## Output Formats

The tools generate results in the following locations:
- **CSV Results:** `Comparison_Experiments/fps_mem_*.csv`
- **Data:** Domain-specific results stored in experiment directories
- **Plots:** PNG/PDF visualizations for publication

## Notes

- Some benchmarking tools may require pre-trained model weights (see `weights/` directory)
- Batch size and device can be configured for each tool
- Results can be aggregated across multiple runs for statistical reliability
- See main README for dataset preparation before running evaluation scripts

