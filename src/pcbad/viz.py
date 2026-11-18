"""Heatmap visualisation (numpy + Pillow only)."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

from .data import IMAGENET_MEAN, IMAGENET_STD


def denormalize(x: np.ndarray) -> np.ndarray:
    """Normalized CHW float image -> HWC uint8."""
    mean = np.array(IMAGENET_MEAN, dtype=np.float32)[:, None, None]
    std = np.array(IMAGENET_STD, dtype=np.float32)[:, None, None]
    img = np.clip(x * std + mean, 0.0, 1.0)
    return (img.transpose(1, 2, 0) * 255.0 + 0.5).astype(np.uint8)


def colorize(v: np.ndarray) -> np.ndarray:
    """Map values in [0, 1] to a jet-like RGB uint8 image."""
    v = np.clip(v, 0.0, 1.0)
    r = np.clip(1.5 - np.abs(4 * v - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4 * v - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * v - 1), 0, 1)
    return (np.stack([r, g, b], axis=-1) * 255).astype(np.uint8)


def overlay_heatmap(
    image: np.ndarray, amap: np.ndarray, alpha: float = 0.5,
    vmin: Optional[float] = None, vmax: Optional[float] = None,
) -> np.ndarray:
    """Blend a colorised anomaly map onto an HWC uint8 image."""
    if image.shape[:2] != amap.shape:
        raise ValueError(f"image {image.shape[:2]} and map {amap.shape} differ in size")
    finite = amap[np.isfinite(amap)]
    lo = (float(finite.min()) if finite.size else 0.0) if vmin is None else vmin
    hi = (float(finite.max()) if finite.size else 1.0) if vmax is None else vmax
    amap = np.nan_to_num(amap, nan=lo, posinf=hi, neginf=lo)
    norm = (amap - lo) / (hi - lo + 1e-12)
    heat = colorize(norm).astype(np.float32)
    out = (1 - alpha) * image.astype(np.float32) + alpha * heat
    return np.clip(out, 0, 255).astype(np.uint8)


def save_panel(
    path: str | Path, image: np.ndarray, amap: np.ndarray,
    mask: Optional[np.ndarray] = None,
    vmin: Optional[float] = None, vmax: Optional[float] = None,
) -> None:
    """Save [image | heatmap overlay | ground-truth mask] side by side."""
    parts = [image, overlay_heatmap(image, amap, vmin=vmin, vmax=vmax)]
    if mask is not None:
        m = (np.asarray(mask) > 0).astype(np.uint8) * 255
        parts.append(np.stack([m, m, m], axis=-1))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.concatenate(parts, axis=1)).save(path)
