"""Dataset indexing and loading for MVTec-style anomaly detection datasets.

Expected layout (used by MVTec AD, and by VisA after the official 1cls split
conversion; PCB categories are pcb1..pcb4 in VisA):

    root/<category>/train/good/*.png
    root/<category>/test/good/*.png
    root/<category>/test/<defect_type>/*.png
    root/<category>/ground_truth/<defect_type>/<name>_mask.png   (optional)

Design note: indexing (pure Python, no torch) is separated from the torch
Dataset so that file discovery can be tested without heavy dependencies.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
from PIL import Image

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class Sample:
    image_path: Path
    label: int                  # 0 = good, 1 = defective
    defect_type: str            # "good" or the defect folder name
    mask_path: Optional[Path]   # None for good images or if no mask exists


def _is_image(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in IMG_EXTS


def _find_mask(root: Path, defect_type: str, image_path: Path) -> Optional[Path]:
    """Mask named <stem>_mask.* (MVTec) or <stem>.* (VisA); extension case-insensitive."""
    gt_dir = root / "ground_truth" / defect_type
    if not gt_dir.is_dir():
        return None
    by_stem = {}
    for f in sorted(gt_dir.iterdir()):
        if _is_image(f):
            by_stem.setdefault(f.stem, f)
    for name in (f"{image_path.stem}_mask", image_path.stem):
        if name in by_stem:
            return by_stem[name]
    return None


def index_split(root: str | Path, split: str) -> List[Sample]:
    """List all samples of a split ('train' or 'test') for one category folder."""
    root = Path(root)
    if split not in ("train", "test"):
        raise ValueError(f"split must be 'train' or 'test', got {split!r}")
    split_dir = root / split
    if not split_dir.is_dir():
        raise FileNotFoundError(f"Split directory not found: {split_dir}")

    samples: List[Sample] = []
    for type_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
        defect = type_dir.name
        is_good = defect == "good"
        if split == "train" and not is_good:
            raise ValueError(
                f"Training folder must contain only 'good' images, found {defect!r}"
            )
        for img in sorted(p for p in type_dir.iterdir() if _is_image(p)):
            mask = None if is_good else _find_mask(root, defect, img)
            samples.append(Sample(img, 0 if is_good else 1, defect, mask))
    if not samples:
        raise ValueError(f"No images found under {split_dir}")
    return samples


def _to_rgb_pil(img) -> Image.Image:
    """Accept a PIL image or a uint8 numpy array (HxW, HxWx1/3/4) -> RGB PIL image."""
    if isinstance(img, Image.Image):
        pil = img
    elif isinstance(img, np.ndarray):
        if img.dtype != np.uint8:
            raise ValueError(f"array images must be uint8, got {img.dtype}")
        if img.ndim == 3 and img.shape[2] == 1:
            img = img[:, :, 0]
        if img.ndim not in (2, 3) or (img.ndim == 3 and img.shape[2] not in (3, 4)):
            raise ValueError(f"unsupported array shape {img.shape}")
        if img.size == 0:
            raise ValueError("empty image array")
        pil = Image.fromarray(img)
    else:
        raise TypeError(f"unsupported image type {type(img).__name__}")
    return pil.convert("RGB")


def prepare_image(img, size: int) -> np.ndarray:
    """PIL/ndarray image -> float32 CHW in [0, 1], resized to (size, size).

    Handles grayscale, palette and RGBA inputs of any resolution.
    """
    pil = _to_rgb_pil(img).resize((size, size), Image.BILINEAR)
    return np.asarray(pil, dtype=np.float32).transpose(2, 0, 1) / 255.0


def load_image(path: Path, size: int) -> np.ndarray:
    """Load an image file as float32 CHW in [0, 1]; corrupt files raise ValueError."""
    try:
        with Image.open(path) as im:
            im.load()
            return prepare_image(im, size)
    except OSError as e:  # includes UnidentifiedImageError and truncated files
        raise ValueError(f"Cannot read image {path}: {e}") from e


def load_mask(sample: Sample, size: int) -> np.ndarray:
    """Binary ground-truth mask (size, size) as uint8; zeros for good images."""
    if sample.mask_path is None:
        return np.zeros((size, size), dtype=np.uint8)
    m = Image.open(sample.mask_path).convert("L").resize((size, size), Image.NEAREST)
    return (np.asarray(m) > 127).astype(np.uint8)


def normalize(x: np.ndarray) -> np.ndarray:
    mean = np.array(IMAGENET_MEAN, dtype=np.float32)[:, None, None]
    std = np.array(IMAGENET_STD, dtype=np.float32)[:, None, None]
    return (x - mean) / std


class AnomalyDataset:
    """Torch-compatible dataset (duck-typed so it works with DataLoader).

    Returns a dict: image (3,H,W) float tensor, mask (H,W), label, path.
    torch is imported lazily so indexing/tests work without it.
    """

    def __init__(self, root: str | Path, split: str, image_size: int = 256):
        self.samples = index_split(root, split)
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int) -> dict:
        import torch  # lazy

        s = self.samples[i]
        img = normalize(load_image(s.image_path, self.image_size))
        return {
            "image": torch.from_numpy(img),
            "mask": torch.from_numpy(load_mask(s, self.image_size)),
            "label": s.label,
            "has_mask": bool(s.label == 0 or s.mask_path is not None),
            "defect_type": s.defect_type,
            "path": str(s.image_path),
        }
