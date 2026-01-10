# ===== Potsdam (ISPRS) Dataset + Loaders =====
from __future__ import annotations

import os
import glob
import random
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

try:
    import rasterio
    _HAS_RASTERIO = True
except Exception:
    _HAS_RASTERIO = False

try:
    import tifffile
    _HAS_TIFFFILE = True
except Exception:
    _HAS_TIFFFILE = False

try:
    import albumentations as A
    _HAS_ALBUMENTATIONS = True
except Exception:
    _HAS_ALBUMENTATIONS = False


POTSDAM_COLOR_MAP: Dict[Tuple[int, int, int], int] = {
    (255, 255, 255): 0,  # impervious
    (0, 0, 255): 1,      # building
    (0, 255, 255): 2,    # low vegetation
    (0, 255, 0): 3,      # tree
    (255, 255, 0): 4,    # car
    (255, 0, 0): 5,      # clutter/background
}


def read_geotiff_rgb(path: str) -> np.ndarray:
    """Read GeoTIFF RGB as HxWx3 uint8. Prefers rasterio; falls back to tifffile."""
    if _HAS_RASTERIO:
        with rasterio.open(path) as src:
            data = src.read()
            # data shape: (C, H, W)
            if data.ndim != 3 or data.shape[0] < 3:
                raise ValueError(f"Expected >=3 bands in {path}, got shape {data.shape}.")
            data = data[:3].transpose(1, 2, 0)
            return data.astype(np.uint8)
    if _HAS_TIFFFILE:
        data = tifffile.imread(path)
        # allow CHW or HWC
        if data.ndim == 3 and data.shape[0] in (3, 4):
            data = data[:3].transpose(1, 2, 0)
        if data.ndim != 3 or data.shape[2] < 3:
            raise ValueError(f"Expected HxWx3 in {path}, got shape {data.shape}.")
        return data[:, :, :3].astype(np.uint8)
    raise ImportError("Neither rasterio nor tifffile is available for GeoTIFF reading.")


def read_label_rgb(path: str) -> np.ndarray:
    """Read label GeoTIFF RGB as HxWx3 uint8."""
    return read_geotiff_rgb(path)


def rgb_label_to_mask(lbl_rgb: np.ndarray, ignore_index: int = 255) -> np.ndarray:
    """Convert RGB label image to class-id mask (0..5, ignore_index for unknown)."""
    if lbl_rgb.ndim != 3 or lbl_rgb.shape[2] != 3:
        raise ValueError(f"Expected HxWx3 label, got shape {lbl_rgb.shape}")
    flat = lbl_rgb.reshape(-1, 3)
    mask = np.full((flat.shape[0],), ignore_index, dtype=np.uint8)
    for color, idx in POTSDAM_COLOR_MAP.items():
        matches = np.all(flat == np.array(color, dtype=np.uint8), axis=1)
        mask[matches] = idx
    return mask.reshape(lbl_rgb.shape[0], lbl_rgb.shape[1])


def _normalize_tile_id(tile_id: str) -> str:
    """Normalize tile_id from split file to base id like '2_10'."""
    tid = tile_id.strip()
    tid = os.path.splitext(tid)[0]
    if tid.startswith("top_potsdam_"):
        tid = tid.replace("top_potsdam_", "")
    if tid.endswith("_RGB"):
        tid = tid[: -len("_RGB")]
    return tid


def resolve_image_path(root: str, tile_id: str) -> str:
    """Resolve Potsdam RGB image path for a tile id."""
    tid = _normalize_tile_id(tile_id)
    candidates = [
        os.path.join(root, "Images", f"top_potsdam_{tid}_RGB.tif"),
        os.path.join(root, "Images", f"{tid}.tif"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    # fallback: glob any matching RGB
    pattern = os.path.join(root, "Images", f"*{tid}*RGB*.tif")
    matches = sorted(glob.glob(pattern))
    if matches:
        return matches[0]
    raise FileNotFoundError(f"No RGB image found for tile_id='{tile_id}' under {root}/Images")


def resolve_label_path(root: str, tile_id: str) -> Optional[str]:
    """Resolve Potsdam label path for a tile id with robust matching."""
    tid = _normalize_tile_id(tile_id)
    patterns = [
        f"top_potsdam_{tid}_label*.tif",
        f"top_potsdam_{tid}_GT*.tif",
        f"top_potsdam_{tid}_*label*.tif",
        f"*{tid}*label*.tif",
        f"*{tid}*GT*.tif",
    ]
    for pattern in patterns:
        candidates = sorted(glob.glob(os.path.join(root, "Labels", pattern)))
        if candidates:
            return candidates[0]
    return None


def _read_split_list(path: str) -> List[str]:
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f.readlines()]
    return [ln for ln in lines if ln]


def _list_tile_ids_from_images(root: str) -> List[str]:
    image_dir = os.path.join(root, "Images")
    patterns = ["*_RGB.tif", "*_RGB.tiff", "*.tif", "*.tiff"]
    img_files: List[str] = []
    for pattern in patterns:
        img_files.extend(glob.glob(os.path.join(image_dir, pattern)))
    img_files = sorted(set(img_files))
    tile_ids = [_normalize_tile_id(os.path.basename(p)) for p in img_files]
    return [tid for tid in tile_ids if tid]


def _cover_positions(length: int, patch: int, stride: int) -> List[int]:
    if length <= patch:
        return [0]
    positions = list(range(0, length - patch + 1, stride))
    last = length - patch
    if positions[-1] != last:
        positions.append(last)
    return positions


def _make_cached_reader(cache_tiles: bool) -> Callable[[str], np.ndarray]:
    if not cache_tiles:
        return read_geotiff_rgb

    @lru_cache(maxsize=16)
    def _cached(path: str) -> np.ndarray:
        return read_geotiff_rgb(path)

    return _cached


def _seed_worker(worker_id: int) -> None:
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)


class PotsdamSegmentationDataset(Dataset):
    """
    Potsdam dataset loader for RGB GeoTIFFs and RGB color labels.

    TRAIN: returns {"rgb": img_tensor, "mask": mask_tensor, "stem": tile_id}
    VAL:   returns {"rgb": img_tensor, "mask": mask_tensor, "meta": meta, "stem": tile_id}
    TEST:  returns {"rgb": img_tensor, "mask": None, "meta": meta, "stem": tile_id}
    """

    def __init__(
        self,
        root: str,
        split: str,
        tile_ids: Optional[List[str]] = None,
        patch_size: int = 512,
        stride: int = 512,
        train_mode: str = "random_crop",  # "random_crop" or "grid"
        augment: bool = True,
        ignore_index: int = 255,
        normalize_mean: Optional[Tuple[float, float, float]] = None,
        normalize_std: Optional[Tuple[float, float, float]] = None,
        cache_tiles: bool = False,
        max_ignore_ratio: float = 1.0,
        seed: int = 0,
    ) -> None:
        super().__init__()
        self.root = root
        self.split = split
        self.patch_size = patch_size
        self.stride = stride
        self.train_mode = train_mode
        self.augment = augment
        self.ignore_index = ignore_index
        self.normalize_mean = normalize_mean
        self.normalize_std = normalize_std
        self.cache_tiles = cache_tiles
        self.max_ignore_ratio = max_ignore_ratio
        self.seed = seed

        if tile_ids is None:
            split_path = os.path.join(root, "splits", f"{split}.txt")
            tile_ids = _read_split_list(split_path)
            if not tile_ids:
                tile_ids = _list_tile_ids_from_images(root)
        if not tile_ids:
            raise FileNotFoundError(
                f"No tiles found for split='{split}'. "
                f"Expected split file under {os.path.join(root, 'splits')} "
                f"or RGB tiles under {os.path.join(root, 'Images')}."
            )

        self.tile_ids = [_normalize_tile_id(tid) for tid in tile_ids]
        self.is_train = split == "train"
        self.is_test = split == "test"

        self._read_rgb = _make_cached_reader(cache_tiles)
        self._read_lbl = _make_cached_reader(cache_tiles)

        self._index: List[Tuple[str, int, int]] = []
        if self.is_train and self.train_mode == "grid":
            self._index = self._build_grid_index()
        elif not self.is_train:
            self._index = self._build_grid_index()

        self._rng = np.random.RandomState(seed)

        self._spatial_aug = None
        self._color_aug = None
        if _HAS_ALBUMENTATIONS and self.augment and self.is_train:
            self._spatial_aug = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
            ])
            self._color_aug = A.Compose([
                A.OneOf([
                    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1.0),
                    A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=10, p=1.0),
                ], p=0.3),
            ])

    def _build_grid_index(self) -> List[Tuple[str, int, int]]:
        index: List[Tuple[str, int, int]] = []
        for tid in self.tile_ids:
            img_path = resolve_image_path(self.root, tid)
            img = self._read_rgb(img_path)
            h, w = img.shape[:2]
            xs = _cover_positions(w, self.patch_size, self.stride)
            ys = _cover_positions(h, self.patch_size, self.stride)
            for y in ys:
                for x in xs:
                    index.append((tid, x, y))
        return index

    def __len__(self) -> int:
        if self.is_train and self.train_mode == "random_crop":
            return len(self.tile_ids)
        return len(self._index)

    def _load_pair(self, tile_id: str) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        img_path = resolve_image_path(self.root, tile_id)
        img = self._read_rgb(img_path)
        label_path = resolve_label_path(self.root, tile_id)
        if label_path is None:
            if self.is_test:
                return img, None
            raise FileNotFoundError(
                f"Label not found for tile_id='{tile_id}' under {self.root}/Labels"
            )
        lbl_rgb = self._read_lbl(label_path)
        return img, lbl_rgb

    def _random_crop(self, img: np.ndarray, mask: Optional[np.ndarray]) -> Tuple[np.ndarray, Optional[np.ndarray], int, int]:
        h, w = img.shape[:2]
        ph = pw = self.patch_size

        if h < ph or w < pw:
            pad_h = max(0, ph - h)
            pad_w = max(0, pw - w)
            img = np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)), mode="constant")
            if mask is not None:
                mask = np.pad(mask, ((0, pad_h), (0, pad_w), (0, 0)), mode="constant", constant_values=0)
            h, w = img.shape[:2]

        for _ in range(10):
            x = self._rng.randint(0, w - pw + 1)
            y = self._rng.randint(0, h - ph + 1)
            img_patch = img[y:y + ph, x:x + pw]
            if mask is None:
                return img_patch, None, x, y
            mask_patch = mask[y:y + ph, x:x + pw]
            if self.max_ignore_ratio >= 1.0:
                return img_patch, mask_patch, x, y
            mask_ids = rgb_label_to_mask(mask_patch, ignore_index=self.ignore_index)
            ignore_ratio = float(np.mean(mask_ids == self.ignore_index))
            if ignore_ratio <= self.max_ignore_ratio:
                return img_patch, mask_patch, x, y
        return img_patch, mask_patch, x, y

    def _grid_crop(self, img: np.ndarray, mask: Optional[np.ndarray], x: int, y: int) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        ph = pw = self.patch_size
        img_patch = img[y:y + ph, x:x + pw]
        if mask is None:
            return img_patch, None
        mask_patch = mask[y:y + ph, x:x + pw]
        return img_patch, mask_patch

    def _apply_aug(self, img: np.ndarray, mask_ids: Optional[np.ndarray]) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        if self._spatial_aug is not None:
            if mask_ids is None:
                aug = self._spatial_aug(image=img)
                img = aug["image"]
            else:
                aug = self._spatial_aug(image=img, mask=mask_ids)
                img = aug["image"]
                mask_ids = aug["mask"]
        if self._color_aug is not None:
            img = self._color_aug(image=img)["image"]
        if not _HAS_ALBUMENTATIONS and self.augment and self.is_train:
            if random.random() < 0.5:
                img = np.flip(img, axis=1)
                if mask_ids is not None:
                    mask_ids = np.flip(mask_ids, axis=1)
            if random.random() < 0.5:
                img = np.flip(img, axis=0)
                if mask_ids is not None:
                    mask_ids = np.flip(mask_ids, axis=0)
            k = random.randint(0, 3)
            if k:
                img = np.rot90(img, k)
                if mask_ids is not None:
                    mask_ids = np.rot90(mask_ids, k)
        return img, mask_ids

    def _to_tensor(self, img: np.ndarray, mask_ids: Optional[np.ndarray]) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        img = img.astype(np.float32) / 255.0
        if self.normalize_mean is not None and self.normalize_std is not None:
            mean = np.array(self.normalize_mean, dtype=np.float32)
            std = np.array(self.normalize_std, dtype=np.float32)
            img = (img - mean) / std
        img = np.transpose(img, (2, 0, 1))
        img_t = torch.from_numpy(img).float()

        if mask_ids is None:
            return img_t, None
        mask_t = torch.from_numpy(mask_ids.astype(np.int64))
        return img_t, mask_t

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if self.is_train and self.train_mode == "random_crop":
            tile_id = self.tile_ids[idx]
            img, lbl_rgb = self._load_pair(tile_id)
            img_patch, lbl_patch, x, y = self._random_crop(img, lbl_rgb)
        else:
            tile_id, x, y = self._index[idx]
            img, lbl_rgb = self._load_pair(tile_id)
            img_patch, lbl_patch = self._grid_crop(img, lbl_rgb, x, y)

        if lbl_patch is not None:
            mask_ids = rgb_label_to_mask(lbl_patch, ignore_index=self.ignore_index)
        else:
            mask_ids = None

        if self.is_train:
            img_patch, mask_ids = self._apply_aug(img_patch, mask_ids)

        img_t, mask_t = self._to_tensor(img_patch, mask_ids)

        meta = {
            "tile_id": tile_id,
            "x": int(x),
            "y": int(y),
            "original_tile_hw": (int(img.shape[0]), int(img.shape[1])),
            "patch_size": int(self.patch_size),
            "stride": int(self.stride),
        }
        batch: Dict[str, Any] = {"rgb": img_t, "mask": mask_t, "stem": tile_id}
        if not self.is_train:
            batch["meta"] = meta
        return batch


class _EmptyDataset(Dataset):
    def __len__(self) -> int:
        return 0

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        raise IndexError("Empty dataset")


def build_potsdam_loaders(
    root: str,
    patch_size: int = 512,
    train_stride: int = 512,
    val_stride: int = 512,
    val_split: float = 0.2,
    test_split: float = 0.0,
    batch_size: int = 4,
    num_workers: int = 4,
    pin_memory: bool = True,
    persistent_workers: bool = True,
    normalize_mean: Optional[Tuple[float, float, float]] = None,
    normalize_std: Optional[Tuple[float, float, float]] = None,
    ignore_index: int = 255,
    train_mode: str = "random_crop",
    augment: bool = True,
    cache_tiles: bool = False,
    seed: int = 0,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Build Potsdam train/val/test DataLoaders."""
    split_dir = os.path.join(root, "splits")
    has_splits = all(
        os.path.exists(os.path.join(split_dir, f"{name}.txt"))
        for name in ("train", "val")
    )
    has_test_split = os.path.exists(os.path.join(split_dir, "test.txt"))
    train_ids = None
    val_ids = None
    test_ids = None
    if not has_splits:
        tile_ids = _list_tile_ids_from_images(root)
        if not tile_ids:
            raise FileNotFoundError(
                f"No tiles found under {os.path.join(root, 'Images')}. "
                "Cannot build splits."
            )
        rng = np.random.RandomState(seed)
        rng.shuffle(tile_ids)
        val_count = max(1, int(len(tile_ids) * val_split)) if val_split > 0 else 0
        test_count = int(len(tile_ids) * test_split) if test_split > 0 else 0
        val_ids = tile_ids[:val_count]
        test_ids = tile_ids[val_count:val_count + test_count] if test_count else []
        train_ids = tile_ids[val_count + test_count:]

    train_ds = PotsdamSegmentationDataset(
        root=root,
        split="train",
        tile_ids=train_ids,
        patch_size=patch_size,
        stride=train_stride,
        train_mode=train_mode,
        augment=augment,
        ignore_index=ignore_index,
        normalize_mean=normalize_mean,
        normalize_std=normalize_std,
        cache_tiles=cache_tiles,
        seed=seed,
    )
    val_ds = PotsdamSegmentationDataset(
        root=root,
        split="val",
        tile_ids=val_ids,
        patch_size=patch_size,
        stride=val_stride,
        train_mode="grid",
        augment=False,
        ignore_index=ignore_index,
        normalize_mean=normalize_mean,
        normalize_std=normalize_std,
        cache_tiles=cache_tiles,
        seed=seed,
    )
    test_ds: Dataset
    if (has_splits and not has_test_split) or (test_ids is not None and len(test_ids) == 0):
        test_ds = _EmptyDataset()
    else:
        test_ds = PotsdamSegmentationDataset(
            root=root,
            split="test",
            tile_ids=test_ids,
            patch_size=patch_size,
            stride=val_stride,
            train_mode="grid",
            augment=False,
            ignore_index=ignore_index,
            normalize_mean=normalize_mean,
            normalize_std=normalize_std,
            cache_tiles=cache_tiles,
            seed=seed,
        )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
        persistent_workers=persistent_workers if num_workers > 0 else False,
        worker_init_fn=_seed_worker,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        persistent_workers=persistent_workers if num_workers > 0 else False,
        worker_init_fn=_seed_worker,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0 if isinstance(test_ds, _EmptyDataset) else num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        persistent_workers=False if isinstance(test_ds, _EmptyDataset) else (persistent_workers if num_workers > 0 else False),
        worker_init_fn=_seed_worker if not isinstance(test_ds, _EmptyDataset) else None,
    )
    return train_loader, val_loader, test_loader


def sanity_check(root: str) -> None:
    """Sanity check: load one tile, convert labels, and validate class ids."""
    ds = PotsdamSegmentationDataset(root=root, split="train", patch_size=512, seed=0)
    batch = ds[0]
    img_t = batch["rgb"]
    mask_t = batch["mask"]
    lbl_rgb = read_label_rgb(resolve_label_path(root, ds.tile_ids[0]) or "")
    mask = rgb_label_to_mask(lbl_rgb)
    unique_colors = np.unique(lbl_rgb.reshape(-1, 3), axis=0)
    unique_ids = np.unique(mask)
    print("Unique label colors:", unique_colors.tolist())
    print("Unique class ids:", unique_ids.tolist())
    allowed = set([0, 1, 2, 3, 4, 5, 255])
    assert set(unique_ids.tolist()).issubset(allowed), "Unexpected class ids detected."
    print("Sanity check passed.")


if __name__ == "__main__":
    # Example usage
    root = "/storage2/ChangeDetection/Datasets/ICPRS/Potsdam"
    train_loader, val_loader, test_loader = build_potsdam_loaders(
        root=root,
        patch_size=512,
        train_stride=512,
        val_stride=512,
        batch_size=2,
        num_workers=2,
        normalize_mean=(0.485, 0.456, 0.406),
        normalize_std=(0.229, 0.224, 0.225),
        train_mode="random_crop",
        augment=True,
        cache_tiles=False,
        seed=0,
    )
    batch = next(iter(train_loader))
    imgs = batch["rgb"]
    masks = batch["mask"]
    print("Train batch imgs:", imgs.shape, "masks:", masks.shape, "mask unique:", torch.unique(masks))
    sanity_check(root)
