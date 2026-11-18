"""Image perturbations that mimic real production-line conditions.

All functions take/return HxWx3 uint8 arrays and never modify the input.
Severity is 1 (mild) .. 3 (strong).
"""
from __future__ import annotations

import io
from typing import Callable, Dict, Optional

import numpy as np
from PIL import Image, ImageFilter

SEVERITIES = (1, 2, 3)

_NOISE_SIGMA = {1: 5.0, 2: 15.0, 3: 30.0}        # camera sensor noise (0-255 units)
_BRIGHTEN_GAIN = {1: 1.25, 2: 1.5, 3: 2.0}        # over-exposure / strong lighting
_DARKEN_GAIN = {1: 0.8, 2: 0.6, 3: 0.4}           # under-exposure / dim lighting
_CONTRAST = {1: 0.8, 2: 0.6, 3: 0.4}              # hazy / low-contrast optics
_BLUR_RADIUS = {1: 1.0, 2: 2.0, 3: 4.0}           # defocus
_JPEG_QUALITY = {1: 70, 2: 40, 3: 15}             # compression artefacts


def _check(img: np.ndarray) -> np.ndarray:
    if img.dtype != np.uint8 or img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(f"expected HxWx3 uint8 image, got {img.dtype} {img.shape}")
    return img


def gaussian_noise(img, severity, rng):
    x = _check(img).astype(np.float32)
    x += rng.normal(0.0, _NOISE_SIGMA[severity], size=x.shape)
    return np.clip(x + 0.5, 0, 255).astype(np.uint8)


def _gain(table):
    def f(img, severity, rng):
        x = _check(img).astype(np.float32) * table[severity]
        return np.clip(x + 0.5, 0, 255).astype(np.uint8)
    return f


def contrast(img, severity, rng):
    x = _check(img).astype(np.float32)
    m = x.mean()
    return np.clip((x - m) * _CONTRAST[severity] + m + 0.5, 0, 255).astype(np.uint8)


def blur(img, severity, rng):
    pil = Image.fromarray(_check(img)).filter(ImageFilter.GaussianBlur(_BLUR_RADIUS[severity]))
    return np.asarray(pil, dtype=np.uint8).copy()


def jpeg(img, severity, rng):
    buf = io.BytesIO()
    Image.fromarray(_check(img)).save(buf, format="JPEG", quality=_JPEG_QUALITY[severity])
    buf.seek(0)
    return np.asarray(Image.open(buf).convert("RGB"), dtype=np.uint8).copy()


PERTURBATIONS: Dict[str, Callable] = {
    "noise": gaussian_noise,
    "brighten": _gain(_BRIGHTEN_GAIN),
    "darken": _gain(_DARKEN_GAIN),
    "contrast": contrast,
    "blur": blur,
    "jpeg": jpeg,
}


def apply_perturbation(
    img: np.ndarray, name: str, severity: int, rng: Optional[np.random.Generator] = None
) -> np.ndarray:
    """Apply a named perturbation. `rng` only matters for stochastic ones (noise)."""
    if name not in PERTURBATIONS:
        raise ValueError(f"unknown perturbation {name!r}; choose from {sorted(PERTURBATIONS)}")
    if severity not in SEVERITIES:
        raise ValueError(f"severity must be one of {SEVERITIES}, got {severity}")
    rng = rng if rng is not None else np.random.default_rng(0)
    return PERTURBATIONS[name](img, severity, rng)
