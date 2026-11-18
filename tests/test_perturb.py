import numpy as np
import pytest

from pcbad.perturb import PERTURBATIONS, SEVERITIES, apply_perturbation


def _img(seed=0):
    return (np.random.default_rng(seed).random((48, 48, 3)) * 200 + 20).astype(np.uint8)


def _dist(a, b):
    return float(np.abs(a.astype(np.float32) - b.astype(np.float32)).mean())


def test_all_perturbations_keep_shape_dtype_and_input_untouched():
    img = _img()
    before = img.copy()
    for name in PERTURBATIONS:
        for sev in SEVERITIES:
            out = apply_perturbation(img, name, sev)
            assert out.shape == img.shape and out.dtype == np.uint8
    assert (img == before).all()


def test_severity_increases_distortion():
    img = _img()
    for name in PERTURBATIONS:
        d = [_dist(img, apply_perturbation(img, name, s)) for s in SEVERITIES]
        assert d[0] < d[1] < d[2], (name, d)


def test_noise_reproducible_with_same_seed_and_differs_across_seeds():
    img = _img()
    a = apply_perturbation(img, "noise", 2, np.random.default_rng(1))
    b = apply_perturbation(img, "noise", 2, np.random.default_rng(1))
    c = apply_perturbation(img, "noise", 2, np.random.default_rng(2))
    assert (a == b).all() and not (a == c).all()


def test_darken_darkens_and_brighten_brightens():
    img = _img()
    assert apply_perturbation(img, "darken", 2).mean() < img.mean()
    assert apply_perturbation(img, "brighten", 2).mean() > img.mean()


def test_contrast_reduces_spread():
    img = _img()
    assert apply_perturbation(img, "contrast", 3).std() < img.std()


def test_extreme_pixels_do_not_overflow():
    white = np.full((8, 8, 3), 255, np.uint8)
    black = np.zeros((8, 8, 3), np.uint8)
    for name in PERTURBATIONS:
        for img in (white, black):
            out = apply_perturbation(img, name, 3)
            assert out.dtype == np.uint8 and out.shape == img.shape


def test_unknown_name_and_bad_severity():
    with pytest.raises(ValueError):
        apply_perturbation(_img(), "rain", 1)
    with pytest.raises(ValueError):
        apply_perturbation(_img(), "noise", 7)


def test_bad_image_format_rejected():
    with pytest.raises(ValueError):
        apply_perturbation(np.zeros((8, 8), np.uint8), "noise", 1)
    with pytest.raises(ValueError):
        apply_perturbation(np.zeros((8, 8, 3), np.float32), "blur", 1)
