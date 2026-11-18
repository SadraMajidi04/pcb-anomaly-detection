"""Model-agnostic evaluation loop.

`predict_fn(images) -> (image_scores (B,), anomaly_maps (B,H,W))` may return
torch tensors or numpy arrays, so the same code evaluates the autoencoder,
PatchCore, or any future model (and can be unit-tested without torch).
"""
from __future__ import annotations

from typing import Callable, Dict, Iterable, Tuple

import numpy as np

from .metrics import best_f1_threshold, image_auroc, per_defect_auroc, pixel_auroc


def _np(x) -> np.ndarray:
    if hasattr(x, "detach"):  # torch tensor
        x = x.detach().cpu().numpy()
    return np.asarray(x)


def evaluate_model(
    predict_fn: Callable, loader: Iterable[dict]
) -> Tuple[Dict, Dict]:
    """Run the model on every batch and compute all metrics.

    Returns (metrics, raw) where raw holds labels/scores/maps/masks/paths for
    plotting. Pixel AUROC only uses images that have a ground-truth mask.
    """
    labels, scores, maps, masks, has_mask, types, paths = [], [], [], [], [], [], []
    for batch in loader:
        s, m = predict_fn(batch["image"])
        s = _np(s).astype(np.float64).reshape(-1)
        m = _np(m).astype(np.float32)
        gt = _np(batch["mask"])
        if m.shape != gt.shape:
            raise ValueError(f"anomaly map {m.shape} does not match mask {gt.shape}")
        y = _np(batch["label"]).astype(int).reshape(-1)
        if not (len(s) == len(y) == len(m)):
            raise ValueError("batch size mismatch between scores, maps and labels")
        hm = batch.get("has_mask")
        labels.append(y)
        scores.append(s)
        maps.append(m)
        masks.append(gt.astype(np.uint8))
        has_mask.append(np.ones(len(y), bool) if hm is None else _np(hm).astype(bool).reshape(-1))
        types.extend(list(batch["defect_type"]))
        paths.extend(list(batch["path"]))
    if not labels:
        raise ValueError("Empty loader: nothing to evaluate")

    y = np.concatenate(labels)
    sc = np.concatenate(scores)
    mp = np.concatenate(maps)
    mk = np.concatenate(masks)
    hm = np.concatenate(has_mask)

    metrics: Dict = {
        "n_images": int(len(y)),
        "n_defective": int(y.sum()),
        "image_auroc": image_auroc(y, sc),
        "per_defect_auroc": per_defect_auroc(y, sc, types),
        **best_f1_threshold(y, sc),
    }
    sel = hm
    metrics["pixel_auroc"] = (
        pixel_auroc(mk[sel], mp[sel]) if sel.any() and mk[sel].any() else None
    )
    raw = {"labels": y, "scores": sc, "maps": mp, "masks": mk,
           "has_mask": hm, "defect_types": types, "paths": paths}
    return metrics, raw
