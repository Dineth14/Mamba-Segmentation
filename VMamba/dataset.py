"""
UrbanMamba Dataset
==================
LovedaDataset with synchronized augmentations for RGB and masks.
Uses albumentations for consistent spatial transforms.
"""

import os
import glob
from typing import Dict, List, Tuple, Callable, Any, Optional
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader

try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    HAS_ALBUMENTATIONS = True
except ImportError:
    HAS_ALBUMENTATIONS = False
    print("Warning: albumentations not installed. Using basic transforms.")

from config import Config


ICPRS_COLOR_MAP = {
    (255, 255, 255): 0,  # Impervious surfaces
    (0, 0, 255): 1,      # Building
    (0, 255, 255): 2,    # Low vegetation
    (0, 255, 0): 3,      # Tree
    (255, 255, 0): 4,    # Car
    (255, 0, 0): 5,      # Clutter/background
}


class LovedaDataset(Dataset):
    """
    LOVEDA Dataset with paired RGB and Mask loading.
    
    Features:
    - Synchronized augmentations (RGB and mask get same spatial transforms)
    - Proper label remapping (LOVEDA: 0->ignore, 1-7->0-6)
    - ImageNet normalization for RGB
    """
    
    def __init__(
        self,
        img_dirs: List[str],
        mask_dirs: List[str],
        crop_size: int = 640,
        is_train: bool = True,
        rgb_mean: Tuple[float, ...] = (0.485, 0.456, 0.406),
        rgb_std: Tuple[float, ...] = (0.229, 0.224, 0.225),
        ignore_index: int = 255
    ):
        super().__init__()
        
        self.crop_size = crop_size
        self.is_train = is_train
        self.rgb_mean = np.array(rgb_mean, dtype=np.float32)
        self.rgb_std = np.array(rgb_std, dtype=np.float32)
        self.ignore_index = ignore_index
        
        # Collect all samples
        self.samples = []
        
        for img_dir, mask_dir in zip(img_dirs, mask_dirs):
            # Find all images
            img_patterns = ['*.png', '*.jpg', '*.jpeg', '*.tif', '*.tiff']
            img_files = []
            for pattern in img_patterns:
                img_files.extend(glob.glob(os.path.join(img_dir, pattern)))
            
            for img_path in img_files:
                # Get filename stem
                stem = os.path.splitext(os.path.basename(img_path))[0]
                
                # Find corresponding mask
                mask_path = None
                for ext in ['.png', '.tif', '.tiff']:
                    candidate = os.path.join(mask_dir, stem + ext)
                    if os.path.exists(candidate):
                        mask_path = candidate
                        break
                
                if mask_path is None:
                    continue
                
                self.samples.append({
                    'img_path': img_path,
                    'mask_path': mask_path,
                    'stem': stem
                })
        
        print(f"Found {len(self.samples)} samples (train={is_train})")
        
        # Build augmentation pipeline
        self.transform = self._build_transforms()
    
    def _build_transforms(self) -> Optional[Callable]:
        """Build albumentations transform pipeline."""
        if not HAS_ALBUMENTATIONS:
            return None
        
        if self.is_train:
            # Training augmentations
            # IMPORTANT: Color augmentations apply to RGB only.
            # We handle this by doing spatial transforms first, then color on RGB only.
            
            # Spatial transforms (apply to RGB and mask)
            spatial_transform = A.Compose([
                A.RandomCrop(height=self.crop_size, width=self.crop_size, p=1.0),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
            ])
            
            # Color transforms (RGB only, applied separately)
            self.color_transform = A.Compose([
                A.OneOf([
                    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1.0),
                    A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=1.0),
                ], p=0.5),
                
                # Blur (RGB only)
                A.OneOf([
                    A.GaussianBlur(blur_limit=(3, 5), p=1.0),
                    A.MedianBlur(blur_limit=3, p=1.0),
                ], p=0.2),
            ])
            
            return spatial_transform
        else:
            # Validation: no augmentation, just pass through
            # For full-resolution validation, we don't crop
            return None
    
    def _load_rgb(self, path: str) -> np.ndarray:
        """Load RGB image as HWC uint8."""
        img = Image.open(path).convert('RGB')
        return np.array(img)
    
    def _load_mask(self, path: str) -> np.ndarray:
        """Load mask and apply label remapping."""
        mask = np.array(Image.open(path))
        
        # LOVEDA label remapping:
        # Original: 0=NoData, 1=Background, 2=Building, ..., 7=Agricultural
        # Target: 0=Background, 1=Building, ..., 6=Agricultural, 255=ignore
        
        mask = mask.astype(np.int64)
        mask = np.clip(mask, 0, 7)
        
        # Shift: 0->-1, 1->0, 2->1, ..., 7->6
        mask = mask - 1
        
        # Map -1 (original 0) to ignore_index
        mask[mask == -1] = self.ignore_index
        
        return mask.astype(np.int64)
    
    def _normalize_rgb(self, img: np.ndarray) -> np.ndarray:
        """Normalize RGB with ImageNet stats."""
        img = img.astype(np.float32) / 255.0
        img = (img - self.rgb_mean) / self.rgb_std
        return img
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        
        # Load data
        img = self._load_rgb(sample['img_path'])
        mask = self._load_mask(sample['mask_path'])
        # Apply synchronized spatial augmentations (to RGB and mask)
        if self.transform is not None:
            transformed = self.transform(image=img, mask=mask)
            img = transformed['image']
            mask = transformed['mask']
        
        # Apply color augmentations to RGB only (training only)
        if self.is_train and hasattr(self, 'color_transform') and self.color_transform is not None:
            color_transformed = self.color_transform(image=img)
            img = color_transformed['image']
        
        # Normalize
        img = self._normalize_rgb(img)
        
        # Convert to tensors (HWC -> CHW)
        img = torch.from_numpy(img.transpose(2, 0, 1)).float()
        mask = torch.from_numpy(mask).long()
        
        batch = {
            'rgb': img,
            'mask': mask,
            'stem': sample['stem']
        }
        return batch


class ICPRSDataset(Dataset):
    """
    ICPRS (ISPRS) dataset loader for Potsdam/Vaihingen-style RGB + color-mask pairs.

    Supports RGB masks with ISPRS color encoding and optional train/val splits.
    """

    def __init__(
        self,
        img_dirs: Optional[List[str]] = None,
        mask_dirs: Optional[List[str]] = None,
        samples: Optional[List[Dict[str, str]]] = None,
        crop_size: int = 640,
        is_train: bool = True,
        rgb_mean: Tuple[float, ...] = (0.485, 0.456, 0.406),
        rgb_std: Tuple[float, ...] = (0.229, 0.224, 0.225),
        ignore_index: int = 255
    ):
        super().__init__()

        self.crop_size = crop_size
        self.is_train = is_train
        self.rgb_mean = np.array(rgb_mean, dtype=np.float32)
        self.rgb_std = np.array(rgb_std, dtype=np.float32)
        self.ignore_index = ignore_index

        if samples is not None:
            self.samples = samples
        else:
            self.samples = self._collect_samples(img_dirs or [], mask_dirs or [])

        print(f"Found {len(self.samples)} ICPRS samples (train={is_train})")
        self.transform = self._build_transforms()

    def _collect_samples(self, img_dirs: List[str], mask_dirs: List[str]) -> List[Dict[str, str]]:
        samples: List[Dict[str, str]] = []
        img_patterns = ['*.png', '*.jpg', '*.jpeg', '*.tif', '*.tiff']
        mask_exts = ['.png', '.tif', '.tiff']

        for img_dir, mask_dir in zip(img_dirs, mask_dirs):
            img_files: List[str] = []
            for pattern in img_patterns:
                img_files.extend(glob.glob(os.path.join(img_dir, pattern)))
            img_files = sorted(img_files)

            for img_path in img_files:
                stem = os.path.splitext(os.path.basename(img_path))[0]
                candidates = [
                    stem,
                    stem.replace("_RGB", ""),
                    stem + "_label",
                    stem.replace("_RGB", "") + "_label",
                ]
                mask_path = None
                for cand in candidates:
                    for ext in mask_exts:
                        candidate = os.path.join(mask_dir, cand + ext)
                        if os.path.exists(candidate):
                            mask_path = candidate
                            break
                    if mask_path:
                        break
                if not mask_path:
                    continue

                samples.append({
                    'img_path': img_path,
                    'mask_path': mask_path,
                    'stem': stem
                })
        return samples

    def _build_transforms(self) -> Optional[Callable]:
        if not HAS_ALBUMENTATIONS:
            return None
        if self.is_train:
            spatial_transform = A.Compose([
                A.RandomCrop(height=self.crop_size, width=self.crop_size, p=1.0),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
            ])
            self.color_transform = A.Compose([
                A.OneOf([
                    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1.0),
                    A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=1.0),
                ], p=0.5),
                A.OneOf([
                    A.GaussianBlur(blur_limit=(3, 5), p=1.0),
                    A.MedianBlur(blur_limit=3, p=1.0),
                ], p=0.2),
            ])
            return spatial_transform
        return None

    def _load_rgb(self, path: str) -> np.ndarray:
        img = Image.open(path).convert('RGB')
        return np.array(img)

    def _load_mask(self, path: str) -> np.ndarray:
        mask = np.array(Image.open(path))
        if mask.ndim == 2:
            mask = mask.astype(np.int64)
            mask[mask == 255] = self.ignore_index
            invalid = (mask < 0) | (mask > 5)
            mask[invalid] = self.ignore_index
            return mask

        h, w, _ = mask.shape
        flat = mask.reshape(-1, 3)
        out = np.full((flat.shape[0],), self.ignore_index, dtype=np.int64)
        for color, idx in ICPRS_COLOR_MAP.items():
            matches = np.all(flat == np.array(color, dtype=np.uint8), axis=1)
            out[matches] = idx
        return out.reshape(h, w)

    def _normalize_rgb(self, img: np.ndarray) -> np.ndarray:
        img = img.astype(np.float32) / 255.0
        img = (img - self.rgb_mean) / self.rgb_std
        return img

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]

        img = self._load_rgb(sample['img_path'])
        mask = self._load_mask(sample['mask_path'])

        if self.transform is not None:
            transformed = self.transform(image=img, mask=mask)
            img = transformed['image']
            mask = transformed['mask']

        if self.is_train and hasattr(self, 'color_transform') and self.color_transform is not None:
            color_transformed = self.color_transform(image=img)
            img = color_transformed['image']

        img = self._normalize_rgb(img)
        img = torch.from_numpy(img.transpose(2, 0, 1)).float()
        mask = torch.from_numpy(mask).long()

        return {'rgb': img, 'mask': mask, 'stem': sample['stem']}


def build_dataloaders(cfg: Config) -> Tuple[DataLoader, DataLoader]:
    """Build training and validation dataloaders."""
    dataset_name = getattr(cfg, "DATASET", "loveda").lower()
    img_dirs_train, mask_dirs_train = cfg.get_train_paths()
    img_dirs_val, mask_dirs_val = cfg.get_val_paths()

    if dataset_name == "icprs":
        if img_dirs_train == img_dirs_val and mask_dirs_train == mask_dirs_val:
            samples = ICPRSDataset(
                img_dirs=img_dirs_train,
                mask_dirs=mask_dirs_train,
                crop_size=cfg.CROP_SIZE,
                is_train=True,
                rgb_mean=cfg.RGB_MEAN,
                rgb_std=cfg.RGB_STD,
                ignore_index=cfg.IGNORE_INDEX
            ).samples
            rng = np.random.RandomState(getattr(cfg, "ICPRS_SEED", 42))
            rng.shuffle(samples)
            val_split = float(getattr(cfg, "ICPRS_VAL_SPLIT", 0.2))
            val_count = max(1, int(len(samples) * val_split))
            val_samples = samples[:val_count]
            train_samples = samples[val_count:]
        else:
            train_samples = None
            val_samples = None

        train_dataset = ICPRSDataset(
            img_dirs=img_dirs_train,
            mask_dirs=mask_dirs_train,
            samples=train_samples,
            crop_size=cfg.CROP_SIZE,
            is_train=True,
            rgb_mean=cfg.RGB_MEAN,
            rgb_std=cfg.RGB_STD,
            ignore_index=cfg.IGNORE_INDEX
        )
        val_dataset = ICPRSDataset(
            img_dirs=img_dirs_val,
            mask_dirs=mask_dirs_val,
            samples=val_samples,
            crop_size=cfg.CROP_SIZE,
            is_train=False,
            rgb_mean=cfg.RGB_MEAN,
            rgb_std=cfg.RGB_STD,
            ignore_index=cfg.IGNORE_INDEX
        )
    else:
        train_dataset = LovedaDataset(
            img_dirs=img_dirs_train,
            mask_dirs=mask_dirs_train,
            crop_size=cfg.CROP_SIZE,
            is_train=True,
            rgb_mean=cfg.RGB_MEAN,
            rgb_std=cfg.RGB_STD,
            ignore_index=cfg.IGNORE_INDEX
        )
        val_dataset = LovedaDataset(
            img_dirs=img_dirs_val,
            mask_dirs=mask_dirs_val,
            crop_size=cfg.CROP_SIZE,
            is_train=False,
            rgb_mean=cfg.RGB_MEAN,
            rgb_std=cfg.RGB_STD,
            ignore_index=cfg.IGNORE_INDEX
        )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=True,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=cfg.PIN_MEMORY,
        drop_last=True,
        persistent_workers=cfg.PERSISTENT_WORKERS and cfg.NUM_WORKERS > 0,
        prefetch_factor=cfg.PREFETCH_FACTOR if cfg.NUM_WORKERS > 0 else None
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.BATCH_SIZE // 2,  # Smaller batch for validation
        shuffle=False,
        num_workers=cfg.NUM_WORKERS // 2,
        pin_memory=cfg.PIN_MEMORY,
        persistent_workers=cfg.PERSISTENT_WORKERS and cfg.NUM_WORKERS > 0,
        prefetch_factor=cfg.PREFETCH_FACTOR if cfg.NUM_WORKERS > 0 else None
    )
    
    return train_loader, val_loader


if __name__ == "__main__":
    # Test dataset
    from config import cfg
    
    print("Testing LovedaDataset...")
    print(f"Data root: {cfg.DATA_ROOT}")
    
    # Get paths
    img_dirs, mask_dirs = cfg.get_train_paths()
    print(f"\nImage dirs: {img_dirs}")
    print(f"Mask dirs: {mask_dirs}")
    
    # Create dataset
    dataset = LovedaDataset(
        img_dirs=img_dirs,
        mask_dirs=mask_dirs,
        crop_size=640,
        is_train=True
    )
    
    print(f"\nDataset size: {len(dataset)}")
    
    if len(dataset) > 0:
        # Load a sample
        sample = dataset[0]
        print(f"\nSample keys: {sample.keys()}")
        print(f"RGB shape: {sample['rgb'].shape}")
        print(f"Mask shape: {sample['mask'].shape}")
        print(f"Mask unique values: {torch.unique(sample['mask'])}")
        print(f"Stem: {sample['stem']}")
        
        # Test dataloader
        loader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0)
        batch = next(iter(loader))
        print(f"\nBatch RGB shape: {batch['rgb'].shape}")
        print(f"Batch Mask shape: {batch['mask'].shape}")
    
    print("\nDataset test complete!")
