import json

import numpy as np
import pytest
from PIL import Image

from pcbad.inference import Calibration, describe, detect, load_calibration

SIZE = 32


def _predict(batch):
    maps = batch.mean(axis=1)                          # (B, S, S)
    return maps.reshape(len(maps), -1).max(axis=1), maps


def _img(bright_patch=True):
    a = np.full((40, 50, 3), 20, np.uint8)
    if bright_patch:
        a[10:20, 10:25] = 250
    return a


def test_threshold_controls_verdict():
    lo = detect(_img(), _predict, SIZE, threshold=0.0)
    hi = detect(_img(), _predict, SIZE, threshold=1e6)
    assert lo.is_anomalous and not hi.is_anomalous
    assert lo.score == hi.score


def test_defect_image_scores_higher_than_plain_image():
    assert detect(_img(True), _predict, SIZE, 1).score > detect(_img(False), _predict, SIZE, 1).score


def test_outputs_shapes_and_types():
    d = detect(_img(), _predict, SIZE, 1.0)
    assert d.overlay.shape == (SIZE, SIZE, 3) and d.overlay.dtype == np.uint8
    assert d.heatmap.shape == (SIZE, SIZE)


def test_accepts_pil_rgba_and_grayscale():
    arr = _img()
    for img in (Image.fromarray(arr), np.dstack([arr, np.full(arr.shape[:2], 255, np.uint8)]),
                arr[:, :, 0]):
        assert detect(img, _predict, SIZE, 1.0).overlay.shape == (SIZE, SIZE, 3)


def test_none_and_bad_inputs_raise_value_error():
    with pytest.raises(ValueError):
        detect(None, _predict, SIZE, 1.0)
    with pytest.raises(ValueError):
        detect(_img(), _predict, SIZE, float("nan"))
    with pytest.raises(ValueError):
        detect(_img(), _predict, SIZE, -1.0)
    with pytest.raises(ValueError):
        detect(np.zeros((8, 8, 3), np.float32), _predict, SIZE, 1.0)


def test_wrong_model_output_shape_raises():
    def bad(batch):
        return np.zeros(1), np.zeros((1, 4, 4))
    with pytest.raises(ValueError):
        detect(_img(), bad, SIZE, 1.0)


def test_describe_text():
    assert "DEFECT" in describe(detect(_img(), _predict, SIZE, 0.0))
    assert "normal" in describe(detect(_img(), _predict, SIZE, 1e6))


def test_load_calibration(tmp_path):
    p = tmp_path / "results.json"
    p.write_text(json.dumps({"patchcore": {"threshold": 2.5}, "autoencoder": {"threshold": 0.0}}))
    assert load_calibration(p, "patchcore") == Calibration(2.5, 0.0, 5.0)
    with pytest.raises(ValueError):
        load_calibration(p, "yolo")
    with pytest.raises(ValueError):
        load_calibration(p, "autoencoder")      # non-positive threshold
