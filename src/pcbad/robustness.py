"""Robustness benchmark: how much does image AUROC drop under perturbations?

numpy-only: `predict_fn` receives numpy batches, so torch models are wrapped by
the caller (see scripts/robustness.py).
"""
from __future__ import annotations

from typing import Callable, Dict, Iterator, List, Optional, Sequence

import numpy as np

from .data import Sample, load_image, load_mask, normalize
from .evaluate import evaluate_model
from .perturb import PERTURBATIONS, SEVERITIES, apply_perturbation


def iter_batches(
    samples: Sequence[Sample], size: int, batch_size: int = 8,
    perturb: Optional[Callable[[np.ndarray], np.ndarray]] = None,
) -> Iterator[dict]:
    """Yield numpy batches; `perturb` acts on the uint8 image at model resolution."""
    for i in range(0, len(samples), batch_size):
        chunk = samples[i:i + batch_size]
        imgs = []
        for s in chunk:
            x = load_image(s.image_path, size)
            if perturb is not None:
                u8 = (x.transpose(1, 2, 0) * 255.0 + 0.5).astype(np.uint8)
                x = perturb(u8).transpose(2, 0, 1).astype(np.float32) / 255.0
            imgs.append(normalize(x))
        yield {
            "image": np.stack(imgs).astype(np.float32),
            "mask": np.stack([load_mask(s, size) for s in chunk]),
            "label": np.array([s.label for s in chunk]),
            "has_mask": np.array([s.label == 0 or s.mask_path is not None for s in chunk]),
            "defect_type": [s.defect_type for s in chunk],
            "path": [str(s.image_path) for s in chunk],
        }


def robustness_table(
    predict_fn: Callable, samples: Sequence[Sample], size: int, batch_size: int = 8,
    names: Optional[Sequence[str]] = None, severities: Sequence[int] = SEVERITIES,
    seed: int = 0,
) -> List[Dict]:
    """Image AUROC on clean data and under each perturbation x severity."""
    names = list(names) if names is not None else list(PERTURBATIONS)

    def auroc(perturb):
        m, _ = evaluate_model(predict_fn, iter_batches(samples, size, batch_size, perturb))
        return m["image_auroc"]

    clean = auroc(None)
    rows = [{"perturbation": "clean", "severity": 0, "image_auroc": clean, "delta": 0.0}]
    for name in names:
        for sev in severities:
            rng = np.random.default_rng(seed)  # same noise stream every run -> reproducible
            a = auroc(lambda u8, n=name, s=sev, r=rng: apply_perturbation(u8, n, s, r))
            rows.append({"perturbation": name, "severity": sev,
                         "image_auroc": a, "delta": a - clean})
    return rows


def robustness_to_markdown(rows: List[Dict]) -> str:
    out = ["| Perturbation | Severity | Image AUROC | Δ vs clean |", "|---|---|---|---|"]
    for r in rows:
        out.append(f"| {r['perturbation']} | {r['severity']} | "
                   f"{r['image_auroc']:.3f} | {r['delta']:+.3f} |")
    return "\n".join(out) + "\n"
