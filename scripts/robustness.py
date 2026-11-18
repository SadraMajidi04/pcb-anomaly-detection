"""Measure how image AUROC degrades under noise, blur, lighting and JPEG changes.

Example:
    python scripts/robustness.py --root data/VisA_pytorch/1cls/pcb1 --model patchcore
    python scripts/robustness.py --root data/VisA_pytorch/1cls/pcb1 --model autoencoder
"""
import argparse
import json
from pathlib import Path

import torch

from pcbad.data import index_split
from pcbad.factory import make_autoencoder, make_patchcore
from pcbad.robustness import robustness_table, robustness_to_markdown


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--model", choices=["autoencoder", "patchcore"], default="patchcore")
    p.add_argument("--ckpt", default=None, help="default: outputs/<model>.pt")
    p.add_argument("--image-size", type=int, default=256, help="used for PatchCore")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--device", default=None)
    p.add_argument("--out", default="outputs")
    a = p.parse_args()

    ckpt = a.ckpt or f"outputs/{a.model}.pt"
    factory = make_autoencoder if a.model == "autoencoder" else make_patchcore
    predict, size = factory(ckpt, a.device, a.image_size)

    rows = robustness_table(lambda x: predict(torch.from_numpy(x)),
                            index_split(a.root, "test"), size, a.batch_size)
    md = robustness_to_markdown(rows)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"robustness_{a.model}.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    (out / f"robustness_{a.model}.md").write_text(md, encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
