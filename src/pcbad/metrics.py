"""Evaluation metrics (numpy / scikit-learn only, no torch)."""
from __future__ import annotations

import time
from typing import Callable, Dict, Optional, Sequence

import numpy as np
from sklearn.metrics import precision_recall_curve, roc_auc_score


def _require_both_classes(y: np.ndarray, what: str) -> None:
    present = np.unique(y)
    if len(present) < 2:
        raise ValueError(f"{what} needs both classes, found only {present.tolist()}")


def image_auroc(labels: Sequence[int], scores: Sequence[float]) -> float:
    """Image-level AUROC (1 = defective, higher score = more anomalous)."""
    y = np.asarray(labels).astype(int)
    s = np.asarray(scores, dtype=np.float64)
    if y.shape != s.shape:
        raise ValueError(f"labels {y.shape} and scores {s.shape} differ in shape")
    _require_both_classes(y, "image AUROC")
    return float(roc_auc_score(y, s))


def pixel_auroc(masks: np.ndarray, maps: np.ndarray) -> float:
    """Pixel-level AUROC over all pixels of all images."""
    masks, maps = np.asarray(masks), np.asarray(maps, dtype=np.float64)
    if masks.shape != maps.shape:
        raise ValueError(f"masks {masks.shape} and maps {maps.shape} differ in shape")
    y = (masks > 0).astype(int).ravel()
    _require_both_classes(y, "pixel AUROC")
    return float(roc_auc_score(y, maps.ravel()))


def best_f1_threshold(labels: Sequence[int], scores: Sequence[float]) -> Dict[str, float]:
    """Threshold maximising F1 on the given data.

    NOTE: choosing the threshold on the test set is optimistic. It is reported
    as an upper bound of what a well-calibrated threshold could reach.
    """
    y = np.asarray(labels).astype(int)
    s = np.asarray(scores, dtype=np.float64)
    _require_both_classes(y, "F1 threshold search")
    p, r, t = precision_recall_curve(y, s)
    p, r = p[:-1], r[:-1]  # align with thresholds
    f1 = 2 * p * r / np.maximum(p + r, 1e-12)
    i = int(np.argmax(f1))
    return {"threshold": float(t[i]), "precision": float(p[i]),
            "recall": float(r[i]), "f1": float(f1[i])}


def per_defect_auroc(
    labels: Sequence[int], scores: Sequence[float], defect_types: Sequence[str]
) -> Dict[str, float]:
    """AUROC of each defect type against all good images."""
    y = np.asarray(labels).astype(int)
    s = np.asarray(scores, dtype=np.float64)
    types = np.asarray(defect_types)
    good = y == 0
    if not good.any():
        raise ValueError("per-defect AUROC needs at least one good image")
    out: Dict[str, float] = {}
    for t in sorted(set(types[y == 1].tolist())):
        sel = good | (types == t)
        out[t] = float(roc_auc_score((y[sel] == 1).astype(int), s[sel]))
    return out


def measure_latency(
    fn: Callable[[], object], warmup: int = 3, runs: int = 20,
    sync: Optional[Callable[[], None]] = None,
) -> Dict[str, float]:
    """Wall-clock latency of `fn()` in milliseconds.

    Pass `sync` (e.g. torch.cuda.synchronize) so asynchronous GPU work is counted.
    """
    if runs < 1 or warmup < 0:
        raise ValueError("runs must be >= 1 and warmup >= 0")
    for _ in range(warmup):
        fn()
    if sync:
        sync()
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        if sync:
            sync()
        times.append((time.perf_counter() - t0) * 1000.0)
    a = np.array(times)
    return {"mean_ms": float(a.mean()), "p50_ms": float(np.percentile(a, 50)),
            "p95_ms": float(np.percentile(a, 95))}
