"""Dataset and preprocessing utilities for time-frequency segmentation."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


DEFAULT_MULTICLASS_NAMES = (
    "background",
    "BPSK",
    "QPSK",
    "8PSK",
    "16QAM",
    "64QAM",
    "FM",
    "AM-DSB",
    "AM-SSB",
    "MSK",
)


DEFAULT_PALETTE = np.array(
    [
        [0, 0, 0],
        [255, 255, 255],
        [0, 255, 0],
        [0, 0, 255],
        [255, 255, 0],
        [128, 0, 128],
        [255, 165, 0],
        [0, 255, 255],
        [255, 0, 255],
        [128, 128, 128],
    ],
    dtype=np.uint8,
)


class SignalSegmentationDataset(Dataset):
    """Image/mask dataset.

    Expected layout by default:

    data_root/
      images/train/*.jpg
      masks/train/*.png
      images/val/*.jpg
      masks/val/*.png

    The original mmseg-style layout is also accepted by passing
    image_dir="img_dir/train" and mask_dir="ann_dir/train".
    """

    def __init__(
        self,
        root: Union[str, Path],
        split: str = "train",
        image_dir: Optional[str] = None,
        mask_dir: Optional[str] = None,
        image_suffixes: Sequence[str] = (".jpg", ".jpeg", ".png"),
        mask_suffix: str = ".png",
        image_size: tuple[int, int] = (512, 512),
        mean: Sequence[float] = (0.485, 0.456, 0.406),
        std: Sequence[float] = (0.229, 0.224, 0.225),
        palette: np.ndarray = DEFAULT_PALETTE,
    ):
        self.root = Path(root)
        self.split = split
        self.image_dir = self.root / (image_dir or f"images/{split}")
        self.mask_dir = self.root / (mask_dir or f"masks/{split}")
        self.mask_suffix = mask_suffix
        self.image_size = image_size
        self.mean = torch.tensor(mean, dtype=torch.float32).view(3, 1, 1)
        self.std = torch.tensor(std, dtype=torch.float32).view(3, 1, 1)
        self.palette = palette
        self.images = []
        for suffix in image_suffixes:
            self.images.extend(sorted(self.image_dir.glob(f"*{suffix}")))
        if not self.images:
            raise FileNotFoundError(f"No images found in {self.image_dir}")

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> dict[str, object]:
        image_path = self.images[index]
        mask_path = self.mask_dir / f"{image_path.stem}{self.mask_suffix}"
        if not mask_path.exists():
            raise FileNotFoundError(f"Missing mask for {image_path.name}: {mask_path}")
        image = Image.open(image_path).convert("RGB").resize(self.image_size, Image.BILINEAR)
        mask = Image.open(mask_path).resize(self.image_size, Image.NEAREST)
        image_tensor = torch.from_numpy(np.asarray(image, dtype=np.float32) / 255.0).permute(2, 0, 1)
        image_tensor = (image_tensor - self.mean) / self.std
        mask_array = mask_to_labels(mask, self.palette)
        return {
            "image": image_tensor,
            "mask": torch.from_numpy(mask_array.astype(np.int64)),
            "image_path": str(image_path),
        }


def mask_to_labels(mask: Image.Image, palette: np.ndarray = DEFAULT_PALETTE) -> np.ndarray:
    arr = np.asarray(mask)
    if arr.ndim == 2:
        return arr.astype(np.int64)
    labels = np.zeros(arr.shape[:2], dtype=np.int64)
    for label, color in enumerate(palette):
        labels[np.all(arr[:, :, :3] == color, axis=-1)] = label
    return labels


def labels_to_color(mask: np.ndarray, palette: np.ndarray = DEFAULT_PALETTE) -> np.ndarray:
    mask = np.asarray(mask, dtype=np.int64)
    out = np.zeros((*mask.shape, 3), dtype=np.uint8)
    valid = (mask >= 0) & (mask < len(palette))
    out[valid] = palette[mask[valid]]
    return out
