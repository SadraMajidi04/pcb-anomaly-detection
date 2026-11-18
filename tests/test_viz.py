import numpy as np
from PIL import Image

from pcbad.data import normalize
from pcbad.viz import colorize, denormalize, overlay_heatmap, save_panel


def test_denormalize_roundtrip():
    rng = np.random.default_rng(0)
    x = rng.random((3, 16, 16)).astype(np.float32)
    back = denormalize(normalize(x))
    assert back.shape == (16, 16, 3) and back.dtype == np.uint8
    assert np.abs(back.astype(int) - (x.transpose(1, 2, 0) * 255).round().astype(int)).max() <= 1


def test_colorize_extremes():
    lo, hi = colorize(np.array([0.0]))[0], colorize(np.array([1.0]))[0]
    assert lo[2] > lo[0]      # low  -> blue
    assert hi[0] > hi[2]      # high -> red


def test_overlay_shape_dtype_and_constant_map():
    img = np.full((10, 12, 3), 100, dtype=np.uint8)
    out = overlay_heatmap(img, np.random.rand(10, 12))
    assert out.shape == img.shape and out.dtype == np.uint8
    flat = overlay_heatmap(img, np.ones((10, 12)))     # vmax == vmin must not crash
    assert flat.shape == img.shape


def test_overlay_size_mismatch_raises():
    import pytest
    with pytest.raises(ValueError):
        overlay_heatmap(np.zeros((10, 12, 3), np.uint8), np.zeros((5, 5)))


def test_save_panel_width(tmp_path):
    img = np.zeros((20, 20, 3), np.uint8)
    p = tmp_path / "sub" / "panel.png"
    save_panel(p, img, np.random.rand(20, 20), mask=np.zeros((20, 20)))
    assert Image.open(p).size == (60, 20)
    save_panel(tmp_path / "p2.png", img, np.random.rand(20, 20))
    assert Image.open(tmp_path / "p2.png").size == (40, 20)
