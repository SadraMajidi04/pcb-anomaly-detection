"""Train the autoencoder baseline and report image-level AUROC on the test split.

Example:
    python scripts/train_autoencoder.py --root data/VisA_pytorch/1cls/pcb1 --epochs 50
"""
import argparse

from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from pcbad.autoencoder import load_autoencoder, score_loader, train_autoencoder
from pcbad.data import AnomalyDataset
from pcbad.utils import get_device


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True, help="category folder, e.g. data/VisA_pytorch/1cls/pcb1")
    p.add_argument("--image-size", type=int, default=256)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--device", default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="outputs/autoencoder.pt")
    a = p.parse_args()

    train_autoencoder(a.root, a.image_size, a.epochs, a.batch_size, a.lr,
                      a.device, a.seed, a.out)

    model, size = load_autoencoder(a.out, a.device)
    loader = DataLoader(AnomalyDataset(a.root, "test", size), batch_size=a.batch_size)
    labels, scores = score_loader(model, loader, get_device(a.device))
    print(f"Image AUROC: {roc_auc_score(labels, scores):.4f}  "
          f"({int(labels.sum())} defective / {len(labels)} test images)")


if __name__ == "__main__":
    main()
