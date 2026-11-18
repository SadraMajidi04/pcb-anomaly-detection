import numpy as np

from pcbad.data import index_split
from pcbad.robustness import iter_batches, robustness_table, robustness_to_markdown


def _predict(images):
    maps = images.mean(axis=1)
    return maps.reshape(len(maps), -1).max(axis=1), maps


def test_iter_batches_shapes_and_batching(fake_category):
    samples = index_split(fake_category, "test")
    batches = list(iter_batches(samples, 32, batch_size=3))
    assert [len(b["label"]) for b in batches] == [3, 1]
    b = batches[0]
    assert b["image"].shape == (3, 3, 32, 32) and b["image"].dtype == np.float32
    assert b["mask"].shape == (3, 32, 32)


def test_perturb_callback_changes_images(fake_category):
    samples = index_split(fake_category, "test")
    clean = next(iter_batches(samples, 32))["image"]
    dark = next(iter_batches(samples, 32, perturb=lambda u8: (u8 * 0.2).astype(np.uint8)))["image"]
    assert not np.allclose(clean, dark)


def test_robustness_table_structure_and_determinism(fake_category):
    samples = index_split(fake_category, "test")
    kw = dict(size=32, batch_size=2, names=["noise", "blur"], severities=(1, 3))
    rows = robustness_table(_predict, samples, **kw)
    assert len(rows) == 1 + 2 * 2
    assert rows[0]["perturbation"] == "clean" and rows[0]["delta"] == 0.0
    assert all(0.0 <= r["image_auroc"] <= 1.0 for r in rows)
    assert rows == robustness_table(_predict, samples, **kw)


def test_robustness_table_feeds_perturbed_data_to_model(fake_category):
    samples = index_split(fake_category, "test")
    seen = []

    def spy(images):
        seen.append(float(images.mean()))
        return _predict(images)

    robustness_table(spy, samples, 32, batch_size=4, names=["darken"], severities=(3,))
    assert len(seen) == 2 and seen[0] != seen[1]    # clean pass vs. darkened pass


def test_markdown_table():
    md = robustness_to_markdown([
        {"perturbation": "clean", "severity": 0, "image_auroc": 0.95, "delta": 0.0},
        {"perturbation": "noise", "severity": 2, "image_auroc": 0.9, "delta": -0.05},
    ])
    assert "| noise | 2 | 0.900 | -0.050 |" in md
