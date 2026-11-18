"""Evaluate trained models on a category's test split and write a comparison.

Writes outputs/results.json, outputs/results.md and heatmap panels under
outputs/figures/. Models whose checkpoint is missing are skipped.

Example:
    python scripts/evaluate.py --root data/VisA_pytorch/1cls/pcb1
"""
import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from pcbad.data import AnomalyDataset
from pcbad.evaluate import evaluate_model
from pcbad.factory import make_autoencoder, make_patchcore
from pcbad.metrics import measure_latency
from pcbad.report import results_to_markdown, save_results
from pcbad.viz import denormalize, save_panel


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--ae-ckpt", default="outputs/autoencoder.pt")
    p.add_argument("--pc-ckpt", default="outputs/patchcore.pt")
    p.add_argument("--image-size", type=int, default=256, help="used for PatchCore")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--device", default=None)
    p.add_argument("--out", default="outputs")
    p.add_argument("--n-figures", type=int, default=6, help="defective samples to render")
    a = p.parse_args()

    candidates = {"autoencoder": (a.ae_ckpt, make_autoencoder),
                  "patchcore": (a.pc_ckpt, make_patchcore)}
    sync = torch.cuda.synchronize if torch.cuda.is_available() else None
    results = {}
    for name, (ckpt, factory) in candidates.items():
        if not Path(ckpt).exists():
            print(f"[skip] {name}: checkpoint not found ({ckpt})")
            continue
        print(f"[eval] {name}")
        predict, size = factory(ckpt, a.device, a.image_size)
        ds = AnomalyDataset(a.root, "test", size)
        metrics, raw = evaluate_model(predict, DataLoader(ds, batch_size=a.batch_size))

        one = ds[0]["image"].unsqueeze(0)
        metrics["latency"] = measure_latency(lambda: predict(one), sync=sync)
        results[name] = metrics

        lo, hi = np.percentile(raw["maps"], [1, 99.9])
        order = np.concatenate([np.where(raw["labels"] == 1)[0][: a.n_figures],
                                np.where(raw["labels"] == 0)[0][:2]])
        for i in order:
            tag = raw["defect_types"][i]
            save_panel(Path(a.out) / "figures" / name / f"{int(i):03d}_{tag}.png",
                       denormalize(ds[int(i)]["image"].numpy()), raw["maps"][i],
                       raw["masks"][i] if raw["has_mask"][i] and raw["labels"][i] else None,
                       vmin=float(lo), vmax=float(hi))

    if not results:
        raise SystemExit("No checkpoints found. Train a model first (see README).")
    save_results(results, a.out)
    print("\n" + results_to_markdown(results))


if __name__ == "__main__":
    main()
