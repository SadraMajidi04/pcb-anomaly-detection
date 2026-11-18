"""Small shared helpers."""
from __future__ import annotations

import random

import numpy as np


def set_seed(seed: int) -> None:
    """Seed python, numpy and torch for reproducible runs."""
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(prefer: str | None = None):
    """Return the requested device, or the best available one (cuda > mps > cpu)."""
    import torch

    if prefer:
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
