"""
Verify RGB-only forward behavior for UrbanMamba.

Runs a single batch through the model and logs feature stats.
"""

import argparse
from typing import Dict, List

import torch

from config import Config
from dataset import build_dataloaders
from model import NSSTMamba


def _tensor_stats(x: torch.Tensor) -> Dict[str, float]:
    return {
        "mean": float(x.mean().item()),
        "std": float(x.std().item()),
        "min": float(x.min().item()),
        "max": float(x.max().item()),
        "abs_mean": float(x.abs().mean().item()),
    }


def _log_feature_stats(name: str, feats: List[torch.Tensor]) -> None:
    print(f"\n{name}:")
    for i, f in enumerate(feats):
        stats = _tensor_stats(f)
        print(
            f"  [{i}] shape={tuple(f.shape)} "
            f"mean={stats['mean']:.4f} std={stats['std']:.4f} "
            f"min={stats['min']:.4f} max={stats['max']:.4f}"
        )
        if torch.isnan(f).any() or torch.isinf(f).any():
            raise ValueError(f"{name}[{i}] contains NaN/Inf")


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify RGB-only UrbanMamba forward pass.")
    parser.add_argument("--checkpoint", default=None, help="Optional checkpoint path")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    cfg = Config()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    train_loader, _ = build_dataloaders(cfg)
    batch = next(iter(train_loader))
    rgb = batch["rgb"].to(device, non_blocking=True)

    print("Input stats:")
    print(f"  rgb shape={tuple(rgb.shape)} stats={_tensor_stats(rgb)}")

    model = NSSTMamba(
        num_classes=cfg.NUM_CLASSES,
        encoder_dims=cfg.VMAMBA_DIMS,
        encoder_depths=cfg.VMAMBA_DEPTHS,
        decoder_channels=cfg.DECODER_CHANNELS,
        drop_path_rate=cfg.VMAMBA_DROP_PATH,
        pretrained_rgb=False,
        weights_path=None,
    ).to(device)

    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location="cpu")
        model.load_state_dict(ckpt.get("model_state_dict", ckpt), strict=False)

    model.eval()
    with torch.no_grad():
        main_out, aux_out, feats = model(rgb, return_features=True)

    print("\nLogits:", tuple(main_out.shape))
    if aux_out is not None:
        print("Aux logits:", tuple(aux_out.shape))

    _log_feature_stats("RGB features", feats["rgb_features"])
    _log_feature_stats("Fused features", feats["fused_features"])

    print("\nVerification complete.")


if __name__ == "__main__":
    main()
