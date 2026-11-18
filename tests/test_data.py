import numpy as np
import pytest

from pcbad.data import index_split, load_image, load_mask, normalize


def test_index_counts(fake_category):
    assert len(index_split(fake_category, "train")) == 4
    test = index_split(fake_category, "test")
    assert len(test) == 4
    assert sum(s.label for s in test) == 2


def test_train_has_only_good(fake_category):
    assert all(s.label == 0 for s in index_split(fake_category, "train"))


def test_defective_samples_have_masks(fake_category):
    for s in index_split(fake_category, "test"):
        assert (s.mask_path is not None) == (s.label == 1)


def test_non_good_in_train_is_rejected(fake_category):
    (fake_category / "train" / "scratch").mkdir()
    with pytest.raises(ValueError):
        index_split(fake_category, "train")


def test_missing_split_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        index_split(tmp_path, "train")


def test_bad_split_name(fake_category):
    with pytest.raises(ValueError):
        index_split(fake_category, "val")


def test_load_image_any_size_gives_fixed_shape(fake_category):
    s = index_split(fake_category, "train")[0]
    for size in (32, 64, 224):
        x = load_image(s.image_path, size)
        assert x.shape == (3, size, size)
        assert 0.0 <= x.min() and x.max() <= 1.0


def test_mask_is_binary_and_resized(fake_category):
    s = [s for s in index_split(fake_category, "test") if s.label == 1][0]
    m = load_mask(s, 64)
    assert m.shape == (64, 64) and set(np.unique(m)) <= {0, 1} and m.sum() > 0


def test_good_mask_is_empty(fake_category):
    s = index_split(fake_category, "train")[0]
    assert load_mask(s, 64).sum() == 0


def test_normalize_shape_preserved():
    x = np.random.rand(3, 8, 8).astype(np.float32)
    assert normalize(x).shape == x.shape
