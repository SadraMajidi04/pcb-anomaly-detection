import numpy as np
import pytest

from pcbad.metrics import (
    best_f1_threshold, image_auroc, measure_latency, per_defect_auroc, pixel_auroc,
)


def test_image_auroc_perfect_and_inverted():
    y = [0, 0, 1, 1]
    assert image_auroc(y, [0.1, 0.2, 0.8, 0.9]) == 1.0
    assert image_auroc(y, [0.9, 0.8, 0.2, 0.1]) == 0.0


def test_image_auroc_ties_give_half():
    assert abs(image_auroc([0, 1, 0, 1], [0.5, 0.5, 0.5, 0.5]) - 0.5) < 1e-9


def test_image_auroc_single_class_raises():
    with pytest.raises(ValueError):
        image_auroc([0, 0, 0], [0.1, 0.2, 0.3])


def test_image_auroc_shape_mismatch_raises():
    with pytest.raises(ValueError):
        image_auroc([0, 1], [0.1, 0.2, 0.3])


def test_pixel_auroc_perfect():
    masks = np.zeros((2, 8, 8), dtype=np.uint8)
    masks[:, 2:5, 2:5] = 1
    maps = masks.astype(np.float32) * 0.9 + 0.05
    assert pixel_auroc(masks, maps) == 1.0


def test_pixel_auroc_no_positive_pixels_raises():
    with pytest.raises(ValueError):
        pixel_auroc(np.zeros((1, 4, 4)), np.random.rand(1, 4, 4))


def test_pixel_auroc_shape_mismatch_raises():
    with pytest.raises(ValueError):
        pixel_auroc(np.zeros((1, 4, 4)), np.zeros((1, 5, 5)))


def test_best_f1_separable_data():
    r = best_f1_threshold([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])
    assert r["f1"] == 1.0 and r["precision"] == 1.0 and r["recall"] == 1.0
    assert 0.2 < r["threshold"] <= 0.8


def test_best_f1_overlapping_is_between_0_and_1():
    r = best_f1_threshold([0, 1, 0, 1, 1, 0], [0.4, 0.5, 0.6, 0.7, 0.3, 0.2])
    assert 0.0 < r["f1"] < 1.0


def test_per_defect_auroc():
    labels = [0, 0, 1, 1, 1, 1]
    scores = [0.1, 0.2, 0.9, 0.8, 0.15, 0.05]
    types = ["good", "good", "scratch", "scratch", "short", "short"]
    r = per_defect_auroc(labels, scores, types)
    assert set(r) == {"scratch", "short"}
    assert r["scratch"] == 1.0
    assert abs(r["short"] - 0.25) < 1e-9


def test_per_defect_needs_good_images():
    with pytest.raises(ValueError):
        per_defect_auroc([1, 1], [0.1, 0.2], ["a", "a"])


def test_latency_keys_and_call_count():
    calls = []
    syncs = []
    r = measure_latency(lambda: calls.append(1), warmup=2, runs=5,
                        sync=lambda: syncs.append(1))
    assert len(calls) == 7
    assert len(syncs) == 1 + 5
    assert set(r) == {"mean_ms", "p50_ms", "p95_ms"}
    assert 0 <= r["p50_ms"] <= r["p95_ms"]


def test_latency_invalid_args():
    with pytest.raises(ValueError):
        measure_latency(lambda: None, runs=0)
