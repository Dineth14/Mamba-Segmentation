#!/usr/bin/env bash
# setup_backbones.sh
# Clone external backbone repositories into the expected locations.
# Run this once from the repo root before training.
#
# Usage:
#   bash setup_backbones.sh           # clone all backbones
#   bash setup_backbones.sh vmamba    # clone a specific backbone
#
# Supported backbone names:
#   mambavision  vmamba  visionmamba  spatialmamba  swintransformer

set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

BACKBONE="${1:-all}"

clone_if_missing() {
    local dest="$1"
    local url="$2"
    if [ -d "$dest" ] && [ -n "$(ls -A "$dest" 2>/dev/null)" ]; then
        echo "[skip] $dest already exists"
    else
        mkdir -p "$(dirname "$dest")"
        echo "[clone] $url -> $dest"
        git clone "$url" "$dest"
    fi
}

if [[ "$BACKBONE" == "all" || "$BACKBONE" == "mambavision" ]]; then
    clone_if_missing MambaVision/MambaVision https://github.com/NVlabs/MambaVision
    echo "[install] MambaVision dependencies"
    pip install -e MambaVision/MambaVision --quiet
fi

if [[ "$BACKBONE" == "all" || "$BACKBONE" == "vmamba" ]]; then
    clone_if_missing VMamba/VMamba https://github.com/MzeroMiko/VMamba
    if [ -f VMamba/VMamba/requirements.txt ]; then
        echo "[install] VMamba dependencies"
        pip install -r VMamba/VMamba/requirements.txt --quiet
    fi
fi

if [[ "$BACKBONE" == "all" || "$BACKBONE" == "visionmamba" ]]; then
    clone_if_missing VisionMamba/Vim https://github.com/hustvl/Vim
fi

if [[ "$BACKBONE" == "all" || "$BACKBONE" == "spatialmamba" ]]; then
    clone_if_missing spatial-mamba/Spatial-Mamba https://github.com/EdwardChasel/Spatial-Mamba
    if [ -f spatial-mamba/Spatial-Mamba/requirements.txt ]; then
        echo "[install] Spatial-Mamba dependencies"
        pip install -r spatial-mamba/Spatial-Mamba/requirements.txt --quiet
    fi
fi

if [[ "$BACKBONE" == "all" || "$BACKBONE" == "swintransformer" ]]; then
    clone_if_missing Swin-Transformer https://github.com/microsoft/Swin-Transformer
fi

echo ""
echo "Done. Backbone repos are ready."
echo "Download pre-trained weights and place them under the corresponding weights/ directories."
echo "See each backbone's README.md for weight download links."
