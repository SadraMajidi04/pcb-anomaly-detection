"""Build `predict(x) -> (image_scores, anomaly_maps)` callables from checkpoints."""
from __future__ import annotations

from typing import Callable, Optional, Tuple

import torch

from .autoencoder import anomaly_map, image_score, load_autoencoder
from .patchcore import PatchCore


def make_autoencoder(ckpt: str, device: Optional[str], _image_size: int = 0) -> Tuple[Callable, int]:
    model, size = load_autoencoder(ckpt, device)
    dev = next(model.parameters()).device

    def predict(x: torch.Tensor):
        amap = anomaly_map(model, x.to(dev))
        return image_score(amap), amap

    return predict, size


def make_patchcore(ckpt: str, device: Optional[str], image_size: int) -> Tuple[Callable, int]:
    return PatchCore.load(ckpt, device).predict, image_size


def to_numpy_predictor(predict: Callable) -> Callable:
    """Wrap a torch predict(x) so it takes/returns numpy arrays."""
    def predict_np(x):
        with torch.no_grad():
            scores, maps = predict(torch.from_numpy(x))
        return scores.cpu().numpy(), maps.cpu().numpy()
    return predict_np
