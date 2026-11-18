"""Fit PatchCore on good images, report image-level AUROC, save the memory bank.

Example:
    python scripts/train_patchcore.py --root data/VisA_pytorch/1cls/pcb1
    python scripts/train_patchcore.py --root data/VisA_pytorch/1cls/pcb1 --backbone resnet18 --coreset-ratio 0.01
"""
import argparse

from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from pcbad.data import AnomalyDataset
from pcbad.patchcore import PatchCore, score_loader


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True, help="category folder, e.g. data/VisA_pytorch/1cls/pcb1")
    p.add_argument("--image-size", type=int, default=256)
    p.add_argument("--backbone", default="wide_resnet50_2",
                   choices=["wide_resnet50_2", "resnet18"])
    p.add_argument("--coreset-ratio", type=float, default=0.1)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--device", default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="outputs/patchcore.pt")
    a = p.parse_args()

    model = PatchCore(a.backbone, coreset_ratio=a.coreset_ratio,
                      device=a.device, seed=a.seed)
    train = DataLoader(AnomalyDataset(a.root, "train", a.image_size), batch_size=a.batch_size)
    size = model.fit(train)
    print(f"memory bank size: {size}")
    model.save(a.out)

    test = DataLoader(AnomalyDataset(a.root, "test", a.image_size), batch_size=a.batch_size)
    labels, scores = score_loader(model, test)
    print(f"Image AUROC: {roc_auc_score(labels, scores):.4f}  "
          f"({int(labels.sum())} defective / {len(labels)} test images)")


if __name__ == "__main__":
    main()
