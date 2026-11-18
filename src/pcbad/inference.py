"""Single-image inference used by the demo (numpy-only, easy to unit-test).

`predict_np(batch) -> (scores (B,), maps (B,H,W))` takes a float32 numpy batch
of normalized images; wrap torch models with `factory.to_numpy_predictor`.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from .data import normalize, prepare_image
from .evaluate import _np
from .viz import overlay_heatmap


@dataclass(frozen=True)
class Calibration:
    """Decision threshold and colour range for heatmaps."""
    threshold: float
    vmin: float
    vmax: float


@dataclass(frozen=True)
class Detection:
    score: float
    threshold: float
    is_anomalous: bool
    overlay: np.ndarray      # (size, size, 3) uint8
    heatmap: np.ndarray      # (size, size) float32 raw anomaly map


def load_calibration(results_json: str | Path, model: str) -> Calibration:
    """Use the best-F1 threshold written by scripts/evaluate.py.

    That threshold was picked on the test set, so treat it as a starting point
    and tune it on your own good images before real deployment.
    """
    data = json.loads(Path(results_json).read_text(encoding="utf-8"))
    if model not in data:
        raise ValueError(f"model {model!r} not in {results_json}; found {sorted(data)}")
    thr = float(data[model]["threshold"])
    if not math.isfinite(thr) or thr <= 0:
        raise ValueError(f"invalid threshold {thr} for model {model!r}")
    return Calibration(threshold=thr, vmin=0.0, vmax=2.0 * thr)


def detect(
    image, predict_np: Callable, size: int, threshold: float,
    vmin: Optional[float] = None, vmax: Optional[float] = None, alpha: float = 0.5,
) -> Detection:
    """Run one image (PIL or uint8 ndarray) through the model."""
    if image is None:
        raise ValueError("No image provided")
    if not math.isfinite(threshold) or threshold < 0:
        raise ValueError(f"threshold must be a finite number >= 0, got {threshold}")

    chw = prepare_image(image, size)                       # float32 [0,1]
    base = (chw.transpose(1, 2, 0) * 255.0 + 0.5).astype(np.uint8)
    batch = normalize(chw)[None].astype(np.float32)

    scores, maps = predict_np(batch)
    scores, maps = _np(scores).reshape(-1), _np(maps).astype(np.float32)
    if maps.shape != (1, size, size) or scores.shape != (1,):
        raise ValueError(f"model returned maps {maps.shape} / scores {scores.shape}, "
                         f"expected (1, {size}, {size}) / (1,)")
    score = float(scores[0])
    return Detection(
        score=score, threshold=float(threshold), is_anomalous=score > threshold,
        overlay=overlay_heatmap(base, maps[0], alpha=alpha, vmin=vmin, vmax=vmax),
        heatmap=maps[0],
    )


def describe(d: Detection) -> str:
    verdict = "DEFECT SUSPECTED" if d.is_anomalous else "Looks normal"
    return f"{verdict}  (anomaly score {d.score:.4f} vs threshold {d.threshold:.4f})"
