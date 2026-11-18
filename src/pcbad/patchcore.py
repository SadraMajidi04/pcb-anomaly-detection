"""PatchCore (Roth et al., CVPR 2022) re-implementation for PCB inspection.

Pipeline
  1. A frozen ImageNet CNN (layer2 + layer3) turns each image into a grid of
     local patch features (3x3 neighbourhood-aggregated, then channel-reduced).
  2. Patch features of all *good* training images form a memory bank, which is
     shrunk with greedy k-center coreset subsampling.
  3. At test time, a patch's anomaly score = distance to its nearest neighbour
     in the bank. Scores are upsampled + Gaussian-smoothed into a heatmap;
     the image score is the heatmap maximum.

No training in the gradient sense: "fit" only extracts features and selects
the coreset, which is why it needs very few good images.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import get_device, set_seed


MIN_INPUT_SIZE = 32  # smaller inputs leave <2x2 feature maps

# --------------------------------------------------------------------------- #
# Feature extractor
# --------------------------------------------------------------------------- #
class FeatureExtractor(nn.Module):
    """Frozen ResNet; returns concatenated, locally-aggregated layer2+layer3 maps."""

    def __init__(self, backbone: str = "wide_resnet50_2", pretrained: bool = True):
        super().__init__()
        import torchvision.models as tvm

        if backbone == "wide_resnet50_2":
            w = tvm.Wide_ResNet50_2_Weights.IMAGENET1K_V1 if pretrained else None
            net = tvm.wide_resnet50_2(weights=w)
        elif backbone == "resnet18":
            w = tvm.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            net = tvm.resnet18(weights=w)
        else:
            raise ValueError(f"Unsupported backbone {backbone!r}")

        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool, net.layer1)
        self.layer2, self.layer3 = net.layer2, net.layer3
        for p in self.parameters():
            p.requires_grad_(False)
        self.eval()

    def train(self, mode: bool = True):  # always stay in eval mode (frozen BN)
        return super().train(False)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f2 = self.layer2(self.stem(x))
        f3 = self.layer3(f2)
        f2 = F.avg_pool2d(f2, 3, stride=1, padding=1)
        f3 = F.avg_pool2d(f3, 3, stride=1, padding=1)
        f3 = F.interpolate(f3, size=f2.shape[-2:], mode="bilinear", align_corners=False)
        return torch.cat([f2, f3], dim=1)  # (B, C2+C3, H/8, W/8)


# --------------------------------------------------------------------------- #
# Numerical helpers (pure torch, unit-tested)
# --------------------------------------------------------------------------- #
@torch.no_grad()
def greedy_coreset(feats: torch.Tensor, k: int, proj_dim: int = 128, seed: int = 0) -> torch.Tensor:
    """Greedy k-center selection; returns k unique row indices (LongTensor).

    Distances are computed in a random low-dimensional projection for speed.
    """
    n = feats.shape[0]
    if k >= n:
        return torch.arange(n)
    g = torch.Generator().manual_seed(seed)
    z = feats
    if feats.shape[1] > proj_dim:
        proj = torch.randn(feats.shape[1], proj_dim, generator=g) / math.sqrt(proj_dim)
        z = feats @ proj.to(feats.device)
    start = int(torch.randint(n, (1,), generator=g))
    selected = [start]
    min_d = (z - z[start]).pow(2).sum(1)
    min_d[start] = -1.0  # selected points can never be chosen again
    for _ in range(k - 1):
        idx = int(torch.argmax(min_d))
        selected.append(idx)
        min_d = torch.minimum(min_d, (z - z[idx]).pow(2).sum(1))
        min_d[idx] = -1.0
    return torch.tensor(selected, dtype=torch.long)


@torch.no_grad()
def nn_distance(queries: torch.Tensor, bank: torch.Tensor, chunk: int = 4096) -> torch.Tensor:
    """Euclidean distance from each query row to its nearest row in `bank`."""
    out = []
    for i in range(0, queries.shape[0], chunk):
        out.append(torch.cdist(queries[i:i + chunk], bank).min(dim=1).values)
    return torch.cat(out)


def gaussian_blur(x: torch.Tensor, sigma: float = 4.0) -> torch.Tensor:
    """Separable Gaussian blur on (B, 1, H, W) with reflect padding."""
    if sigma <= 0:
        return x
    # reflect padding needs pad < size, so truncate the kernel on very small maps
    r = min(int(4 * sigma + 0.5), min(x.shape[-2:]) - 1)
    if r < 1:
        return x
    t = torch.arange(-r, r + 1, dtype=x.dtype, device=x.device)
    k = torch.exp(-(t ** 2) / (2 * sigma ** 2))
    k = k / k.sum()
    x = F.pad(x, (r, r, 0, 0), mode="reflect")
    x = F.conv2d(x, k.view(1, 1, 1, -1))
    x = F.pad(x, (0, 0, r, r), mode="reflect")
    return F.conv2d(x, k.view(1, 1, -1, 1))


# --------------------------------------------------------------------------- #
# PatchCore model
# --------------------------------------------------------------------------- #
class PatchCore:
    def __init__(
        self,
        backbone: str = "wide_resnet50_2",
        pretrained: bool = True,
        target_dim: int = 384,
        coreset_ratio: float = 0.1,
        sigma: float = 4.0,
        device: Optional[str] = None,
        seed: int = 0,
    ):
        if not 0 < coreset_ratio <= 1:
            raise ValueError("coreset_ratio must be in (0, 1]")
        self.cfg: Dict = dict(backbone=backbone, target_dim=target_dim,
                              coreset_ratio=coreset_ratio, sigma=sigma, seed=seed)
        self.device = get_device(device)
        self.extractor = FeatureExtractor(backbone, pretrained).to(self.device)
        self.bank: Optional[torch.Tensor] = None

    # -- features ----------------------------------------------------------- #
    @torch.no_grad()
    def _embed(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int, int]]:
        if x.ndim != 4 or x.shape[1] != 3:
            raise ValueError(f"expected input of shape (B, 3, H, W), got {tuple(x.shape)}")
        if min(x.shape[-2:]) < MIN_INPUT_SIZE:
            raise ValueError(f"input must be at least {MIN_INPUT_SIZE}x{MIN_INPUT_SIZE}, "
                             f"got {x.shape[-2]}x{x.shape[-1]}")
        f = self.extractor(x.to(self.device))
        b, c, h, w = f.shape
        f = f.permute(0, 2, 3, 1).reshape(-1, c)
        td = self.cfg["target_dim"]
        if c > td:  # channel reduction by adaptive average pooling
            f = F.adaptive_avg_pool1d(f.unsqueeze(1), td).squeeze(1)
        return f, (b, h, w)

    # -- fit ---------------------------------------------------------------- #
    @torch.no_grad()
    def fit(self, loader: Iterable[dict], verbose: bool = True) -> int:
        """Build the memory bank from good images. Returns bank size."""
        set_seed(self.cfg["seed"])
        chunks = [self._embed(batch["image"])[0] for batch in loader]
        if not chunks:
            raise ValueError("Empty loader: no training images")
        patches = torch.cat(chunks)
        k = max(1, int(patches.shape[0] * self.cfg["coreset_ratio"]))
        if verbose:
            print(f"patches: {patches.shape[0]}  dim: {patches.shape[1]}  coreset: {k}")
        idx = greedy_coreset(patches, k, seed=self.cfg["seed"])
        self.bank = patches[idx.to(patches.device)].contiguous()
        return self.bank.shape[0]

    # -- predict ------------------------------------------------------------ #
    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """x: (B,3,H,W) normalized. Returns (image_scores (B,), maps (B,H,W))."""
        if self.bank is None:
            raise RuntimeError("Call fit() or load() before predict()")
        feats, (b, h, w) = self._embed(x)
        d = nn_distance(feats, self.bank).reshape(b, 1, h, w)
        d = F.interpolate(d, size=x.shape[-2:], mode="bilinear", align_corners=False)
        amap = gaussian_blur(d, self.cfg["sigma"]).squeeze(1)
        return amap.flatten(1).max(dim=1).values, amap

    # -- persistence -------------------------------------------------------- #
    def save(self, path: str | Path) -> None:
        if self.bank is None:
            raise RuntimeError("Nothing to save: model is not fitted")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"bank": self.bank.cpu(), "cfg": self.cfg}, path)

    @classmethod
    def load(cls, path: str | Path, device: Optional[str] = None, pretrained: bool = True) -> "PatchCore":
        ckpt = torch.load(path, map_location="cpu")
        cfg = ckpt["cfg"]
        model = cls(backbone=cfg["backbone"], pretrained=pretrained,
                    target_dim=cfg["target_dim"], coreset_ratio=cfg["coreset_ratio"],
                    sigma=cfg["sigma"], device=device, seed=cfg["seed"])
        model.bank = ckpt["bank"].to(model.device)
        return model


@torch.no_grad()
def score_loader(model: PatchCore, loader: Iterable[dict]) -> Tuple[np.ndarray, np.ndarray]:
    """Return (labels, image_scores) for every sample in the loader."""
    labels, scores = [], []
    for batch in loader:
        s, _ = model.predict(batch["image"])
        scores.extend(s.cpu().tolist())
        labels.extend(int(v) for v in batch["label"])
    return np.array(labels), np.array(scores)
