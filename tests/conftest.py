import numpy as np
import pytest
from PIL import Image


def _save(path, arr):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(path)


@pytest.fixture
def fake_category(tmp_path):
    """Tiny synthetic MVTec-style category: 4 train, 2 good test, 2 defective."""
    root = tmp_path / "pcb_fake"
    rng = np.random.default_rng(0)
    for i in range(4):
        _save(root / "train" / "good" / f"{i:03d}.png",
              rng.integers(0, 255, (40, 60, 3), dtype=np.uint8))
    for i in range(2):
        _save(root / "test" / "good" / f"{i:03d}.png",
              rng.integers(0, 255, (40, 60, 3), dtype=np.uint8))
    for i in range(2):
        _save(root / "test" / "scratch" / f"{i:03d}.png",
              rng.integers(0, 255, (40, 60, 3), dtype=np.uint8))
        mask = np.zeros((40, 60), dtype=np.uint8)
        mask[10:20, 10:30] = 255
        _save(root / "ground_truth" / "scratch" / f"{i:03d}_mask.png", mask)
    return root
