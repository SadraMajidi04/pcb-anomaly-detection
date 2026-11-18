"""Randomised (seeded) property tests for the metric/perturbation code."""
import numpy as np

from pcbad.metrics import best_f1_threshold, image_auroc, pixel_auroc
from pcbad.perturb import PERTURBATIONS, SEVERITIES, apply_perturbation
from pcbad.viz import overlay_heatmap


def _labels_scores(rng, n=60):
    y = rng.integers(0, 2, n)
    y[0], y[1] = 0, 1                      # guarantee both classes
    return y, rng.random(n)


def test_auroc_in_unit_interval_and_complementary():
    rng = np.random.default_rng(0)
    for _ in range(25):
        y, s = _labels_scores(rng)
        a = image_auroc(y, s)
        assert 0.0 <= a <= 1.0
        assert abs(a + image_auroc(y, -s) - 1.0) < 1e-9


def test_auroc_invariant_to_monotone_transform_and_permutation():
    rng = np.random.default_rng(1)
    for _ in range(25):
        y, s = _labels_scores(rng)
        a = image_auroc(y, s)
        assert abs(a - image_auroc(y, 5 * s + 3)) < 1e-9
        assert abs(a - image_auroc(y, np.exp(s))) < 1e-9
        p = rng.permutation(len(y))
        assert abs(a - image_auroc(y[p], s[p])) < 1e-9


def test_best_f1_matches_bruteforce():
    rng = np.random.default_rng(2)
    for _ in range(20):
        y, s = _labels_scores(rng, 40)
        best = 0.0
        for t in np.unique(s):
            pred = s >= t
            tp = int((pred & (y == 1)).sum())
            fp = int((pred & (y == 0)).sum())
            fn = int((~pred & (y == 1)).sum())
            best = max(best, 2 * tp / max(2 * tp + fp + fn, 1))
        assert abs(best_f1_threshold(y, s)["f1"] - best) < 1e-9


def test_pixel_auroc_random_maps_stay_in_range():
    rng = np.random.default_rng(3)
    masks = (rng.random((3, 16, 16)) > 0.8).astype(np.uint8)
    masks[0, 0, 0], masks[0, 0, 1] = 0, 1
    assert 0.0 <= pixel_auroc(masks, rng.random((3, 16, 16))) <= 1.0


def test_perturbations_on_odd_image_sizes():
    rng = np.random.default_rng(4)
    for shape in [(1, 1, 3), (1, 7, 3), (5, 3, 3), (33, 17, 3), (64, 200, 3)]:
        img = rng.integers(0, 256, shape, dtype=np.uint8)
        for name in PERTURBATIONS:
            for sev in SEVERITIES:
                out = apply_perturbation(img, name, sev, np.random.default_rng(0))
                assert out.shape == img.shape and out.dtype == np.uint8


def test_overlay_survives_nan_inf_and_constant_maps():
    img = np.full((10, 10, 3), 128, np.uint8)
    for amap in (np.full((10, 10), np.nan), np.full((10, 10), np.inf),
                 np.zeros((10, 10)), np.random.rand(10, 10) * 1e9):
        out = overlay_heatmap(img, amap)
        assert out.shape == img.shape and out.dtype == np.uint8
    amap = np.random.rand(10, 10)
    amap[3, 3] = np.nan
    assert overlay_heatmap(img, amap).shape == img.shape
