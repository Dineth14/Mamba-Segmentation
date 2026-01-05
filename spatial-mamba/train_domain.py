"""
Example usage:
  cd /storage2/ChangeDetection/NSST-mamba/Mamba-Segmentation/spatial-mamba
  python train_domain.py --variant tiny  --domain urban --gpu 0 --amp 1
  python train_domain.py --variant small --domain rural --gpu 1 --amp 1
"""

import argparse
import os
import sys

# Ensure we import from the local config.py, not from Spatial-Mamba/classification/config.py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import Config
import train as train_module


DATA_ROOT = "/storage2/ChangeDetection/Datasets/Loveda"


def _domain_paths(domain: str) -> dict:
    domain_cap = "Urban" if domain == "urban" else "Rural"
    return {
        "train_img": f"Train/Train/{domain_cap}/images_png",
        "train_mask": f"Train/Train/{domain_cap}/masks_png",
        "val_img": f"Val/Val/{domain_cap}/images_png",
        "val_mask": f"Val/Val/{domain_cap}/masks_png",
    }


def _build_output_dir(out_root: str, family: str, variant: str, domain: str, crop: int, tag: str) -> str:
    suffix = f"_{tag}" if tag else ""
    run_name = f"{family}_{variant}_{domain}train_{crop}{suffix}"
    return os.path.join(out_root, run_name)


def main() -> None:
    parser = argparse.ArgumentParser(description="Domain-specific training wrapper for Spatial-Mamba UrbanMamba.")
    parser.add_argument("--domain", choices=["urban", "rural"], required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--amp", type=int, choices=[0, 1], default=None)
    parser.add_argument("--iters", type=int, default=50000)
    parser.add_argument("--crop", type=int, default=512)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument(
        "--out_root",
        default="/storage2/ChangeDetection/NSST-mamba/Mamba-Segmentation/Comparison_Experiments",
    )
    parser.add_argument("--tag", default="")
    args = parser.parse_args()

    paths = _domain_paths(args.domain)
    output_dir = _build_output_dir(args.out_root, "spatialmamba", args.variant, args.domain, args.crop, args.tag)

    cfg = Config(
        DATA_ROOT=DATA_ROOT,
        TRAIN_IMG_DIR=[paths["train_img"]],
        TRAIN_MASK_DIR=[paths["train_mask"]],
        VAL_IMG_DIR=[paths["val_img"]],
        VAL_MASK_DIR=[paths["val_mask"]],
        OUTPUT_DIR=output_dir,
        GPU_ID=args.gpu,
        BATCH_SIZE=args.batch,
        CROP_SIZE=args.crop,
        MAX_ITERS=args.iters,
        WEIGHTS_PATH="auto",
        SPATIALMAMBA_VARIANT=args.variant,
    )
    if args.amp is not None:
        cfg.USE_AMP = bool(args.amp)

    train_img_full = os.path.join(cfg.DATA_ROOT, paths["train_img"])
    train_mask_full = os.path.join(cfg.DATA_ROOT, paths["train_mask"])
    val_img_full = os.path.join(cfg.DATA_ROOT, paths["val_img"])
    val_mask_full = os.path.join(cfg.DATA_ROOT, paths["val_mask"])

    print(f"[train_domain] Domain: {args.domain}")
    print(f"[train_domain] Train images: {train_img_full}")
    print(f"[train_domain] Train masks:  {train_mask_full}")
    print(f"[train_domain] Val images:   {val_img_full}")
    print(f"[train_domain] Val masks:    {val_mask_full}")
    print(f"[train_domain] Output dir:   {cfg.OUTPUT_DIR}")

    if hasattr(train_module, "_configure_runtime"):
        train_module._configure_runtime()
    resume_path = cfg.RESUME_PATH if cfg.RESUME_PATH else None
    train_module.train(cfg, resume_path=resume_path)


if __name__ == "__main__":
    main()
