"""Convolutional autoencoder baseline for unsupervised anomaly detection.

Idea: train to reconstruct defect-free PCB images. At test time, regions the
model reconstructs poorly (large error) are flagged as anomalous.

Known weakness (documented honestly in the README): a sufficiently powerful
autoencoder also reconstructs defects well ("identity shortcut"), which is why
this is the *baseline* and PatchCore is the main model.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .data import AnomalyDataset
from .utils import get_device, set_seed

DOWNSAMPLE = 32  # 5 stride-2 convolutions -> input size must be a multiple of 32


def _down(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, 4, stride=2, padding=1),
        nn.BatchNorm2d(cout),
        nn.LeakyReLU(0.2, inplace=True),
    )


def _up(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(
        nn.ConvTranspose2d(cin, cout, 4, stride=2, padding=1),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
    )


class ConvAutoencoder(nn.Module):
    def __init__(self, base: int = 32, latent_channels: int = 64):
        super().__init__()
        self.encoder = nn.Sequential(
            _down(3, base),               # /2
            _down(base, base * 2),        # /4
            _down(base * 2, base * 4),    # /8
            _down(base * 4, base * 4),    # /16
            _down(base * 4, base * 4),    # /32
            nn.Conv2d(base * 4, latent_channels, 1),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(latent_channels, base * 4, 1),
            _up(base * 4, base * 4),
            _up(base * 4, base * 4),
            _up(base * 4, base * 2),
            _up(base * 2, base),
            nn.ConvTranspose2d(base, 3, 4, stride=2, padding=1),  # linear output
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or x.shape[1] != 3:
            raise ValueError(f"expected input of shape (B, 3, H, W), got {tuple(x.shape)}")
        h, w = x.shape[-2:]
        if h % DOWNSAMPLE or w % DOWNSAMPLE:
            raise ValueError(
                f"Input size must be a multiple of {DOWNSAMPLE}, got {h}x{w}"
            )
        return self.decoder(self.encoder(x))


@torch.no_grad()
def anomaly_map(model: nn.Module, x: torch.Tensor, blur_kernel: int = 7) -> torch.Tensor:
    """Per-pixel anomaly map (B, H, W): channel-mean squared error, lightly blurred."""
    model.eval()
    err = (model(x) - x).pow(2).mean(dim=1, keepdim=True)
    pad = blur_kernel // 2
    err = F.avg_pool2d(err, blur_kernel, stride=1, padding=pad)
    return err.squeeze(1)


def image_score(amap: torch.Tensor) -> torch.Tensor:
    """Image-level score = maximum of the anomaly map."""
    return amap.flatten(1).max(dim=1).values


@torch.no_grad()
def score_loader(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (labels, image_scores) for every sample in the loader."""
    labels: List[int] = []
    scores: List[float] = []
    for batch in loader:
        s = image_score(anomaly_map(model, batch["image"].to(device)))
        scores.extend(s.cpu().tolist())
        labels.extend(int(v) for v in batch["label"])
    return np.array(labels), np.array(scores)


def train_autoencoder(
    root: str | Path,
    image_size: int = 256,
    epochs: int = 50,
    batch_size: int = 16,
    lr: float = 1e-3,
    device: Optional[str] = None,
    seed: int = 0,
    out: Optional[str | Path] = "outputs/autoencoder.pt",
    num_workers: int = 0,
    verbose: bool = True,
) -> Dict[str, List[float]]:
    """Train on `root/train/good` and return the per-epoch loss history."""
    set_seed(seed)
    dev = get_device(device)
    ds = AnomalyDataset(root, "train", image_size)
    loader = DataLoader(
        ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=len(ds) > batch_size
    )
    model = ConvAutoencoder().to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    history: Dict[str, List[float]] = {"loss": []}
    for epoch in range(1, epochs + 1):
        model.train()
        total, n = 0.0, 0
        for batch in loader:
            x = batch["image"].to(dev)
            loss = F.mse_loss(model(x), x)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * x.size(0)
            n += x.size(0)
        history["loss"].append(total / n)
        if verbose and (epoch == 1 or epoch % 10 == 0 or epoch == epochs):
            print(f"epoch {epoch:3d}/{epochs}  loss {history['loss'][-1]:.5f}")

    if out is not None:
        out = Path(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": model.state_dict(), "image_size": image_size}, out)
    return history


def load_autoencoder(path: str | Path, device: Optional[str] = None):
    """Load a checkpoint written by train_autoencoder. Returns (model, image_size)."""
    dev = get_device(device)
    ckpt = torch.load(path, map_location=dev)
    model = ConvAutoencoder().to(dev)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt["image_size"]
