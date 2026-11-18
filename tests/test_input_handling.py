import numpy as np
import pytest
from PIL import Image

from pcbad.data import load_image, prepare_image


def test_grayscale_rgba_palette_all_become_rgb():
    arr = (np.random.rand(20, 30) * 255).astype(np.uint8)
    for pil in (Image.fromarray(arr, "L"),
                Image.fromarray(np.dstack([arr] * 4), "RGBA"),
                Image.fromarray(arr, "L").convert("P")):
        assert prepare_image(pil, 32).shape == (3, 32, 32)


def test_numpy_input_variants():
    a2 = np.zeros((10, 12), np.uint8)
    assert prepare_image(a2, 16).shape == (3, 16, 16)
    assert prepare_image(a2[:, :, None], 16).shape == (3, 16, 16)
    assert prepare_image(np.zeros((10, 12, 3), np.uint8), 16).shape == (3, 16, 16)
    assert prepare_image(np.zeros((10, 12, 4), np.uint8), 16).shape == (3, 16, 16)


def test_extreme_aspect_ratio_and_tiny_images():
    assert prepare_image(np.zeros((1, 1, 3), np.uint8), 32).shape == (3, 32, 32)
    assert prepare_image(np.zeros((5, 4000, 3), np.uint8), 32).shape == (3, 32, 32)


def test_invalid_arrays_rejected():
    with pytest.raises(ValueError):
        prepare_image(np.zeros((8, 8, 3), np.float32), 16)       # not uint8
    with pytest.raises(ValueError):
        prepare_image(np.zeros((8, 8, 5), np.uint8), 16)         # bad channel count
    with pytest.raises(ValueError):
        prepare_image(np.zeros((2, 8, 8, 3), np.uint8), 16)      # 4-D
    with pytest.raises(ValueError):
        prepare_image(np.zeros((0, 8, 3), np.uint8), 16)         # empty
    with pytest.raises(TypeError):
        prepare_image("not an image", 16)


def test_corrupt_file_raises_value_error(tmp_path):
    bad = tmp_path / "bad.png"
    bad.write_bytes(b"this is not a png")
    with pytest.raises(ValueError):
        load_image(bad, 32)


def test_truncated_file_raises_value_error(tmp_path):
    good = tmp_path / "good.png"
    Image.fromarray((np.random.rand(64, 64, 3) * 255).astype(np.uint8)).save(good)
    data = good.read_bytes()
    cut = tmp_path / "cut.png"
    cut.write_bytes(data[: len(data) // 2])
    with pytest.raises(ValueError):
        load_image(cut, 32)


def test_missing_file_raises_value_error(tmp_path):
    with pytest.raises(ValueError):
        load_image(tmp_path / "nope.png", 32)


def test_valid_file_still_loads(tmp_path):
    p = tmp_path / "ok.png"
    Image.fromarray(np.full((20, 20, 3), 128, np.uint8)).save(p)
    x = load_image(p, 16)
    assert x.shape == (3, 16, 16) and abs(float(x.mean()) - 128 / 255) < 0.01
