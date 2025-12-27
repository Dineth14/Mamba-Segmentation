#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./run_nsst_batch.sh /path/to/loveda /path/to/output_mat
#
# Output layout mirrors:
#   <output_mat>/Train/Urban/images_png
#   <output_mat>/Train/Rural/images_png
#   <output_mat>/Val/Urban/images_png
#   <output_mat>/Val/Rural/images_png

LOVEDA_ROOT="${1:?Loveda root required}"
OUT_ROOT="${2:?Output root required}"

MATLAB_CMD=${MATLAB_CMD:-matlab}

declare -a SUBSETS=(
  "Train/Train/Urban/images_png:Train/Urban/images_png"
  "Train/Train/Rural/images_png:Train/Rural/images_png"
  "Val/Val/Urban/images_png:Val/Urban/images_png"
  "Val/Val/Rural/images_png:Val/Rural/images_png"
)

for pair in "${SUBSETS[@]}"; do
  IN_REL="${pair%%:*}"
  OUT_REL="${pair##*:}"
  IN_DIR="${LOVEDA_ROOT}/${IN_REL}"
  OUT_DIR="${OUT_ROOT}/${OUT_REL}"

  echo "Processing: ${IN_DIR} -> ${OUT_DIR}"
  "${MATLAB_CMD}" -batch "matlab.internal.webservices.disableConnector; addpath('$(pwd)'); compute_nsst_batch('${IN_DIR}', '${OUT_DIR}');"
done

echo "All subsets processed."
