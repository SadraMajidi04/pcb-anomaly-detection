import numpy as np
import pytest

from pcbad.evaluate import evaluate_model
from pcbad.report import results_to_markdown, save_results


def _batches(with_has_mask=True, bad_mask_everywhere=False):
    """Two batches of 2 images (8x8). Defective images contain a bright 3x3 patch."""
    out = []
    for b in range(2):
        imgs = np.zeros((2, 3, 8, 8), dtype=np.float32)
        masks = np.zeros((2, 8, 8), dtype=np.uint8)
        imgs[1, :, 2:5, 2:5] = 1.0
        masks[1, 2:5, 2:5] = 1
        batch = {
            "image": imgs, "mask": masks, "label": np.array([0, 1]),
            "defect_type": ["good", "scratch"],
            "path": [f"g{b}.png", f"d{b}.png"],
        }
        if with_has_mask:
            batch["has_mask"] = np.array([True, True])
        out.append(batch)
    return out


def _predict(images):
    maps = images.mean(axis=1)               # (B, 8, 8)
    return maps.reshape(len(maps), -1).max(axis=1), maps


def test_perfect_model_gets_perfect_scores():
    m, raw = evaluate_model(_predict, _batches())
    assert m["image_auroc"] == 1.0
    assert m["pixel_auroc"] == 1.0
    assert m["f1"] == 1.0
    assert m["n_images"] == 4 and m["n_defective"] == 2
    assert m["per_defect_auroc"] == {"scratch": 1.0}
    assert raw["maps"].shape == (4, 8, 8)
    assert raw["paths"] == ["g0.png", "d0.png", "g1.png", "d1.png"]


def test_works_without_has_mask_key():
    m, _ = evaluate_model(_predict, _batches(with_has_mask=False))
    assert m["pixel_auroc"] == 1.0


def test_pixel_auroc_none_when_defects_have_no_masks():
    batches = _batches()
    for b in batches:
        b["has_mask"] = np.array([True, False])   # defective image lacks a mask
    m, _ = evaluate_model(_predict, batches)
    assert m["pixel_auroc"] is None
    assert m["image_auroc"] == 1.0


def test_map_mask_shape_mismatch_raises():
    def bad_predict(images):
        return np.zeros(len(images)), np.zeros((len(images), 4, 4))
    with pytest.raises(ValueError):
        evaluate_model(bad_predict, _batches())


def test_empty_loader_raises():
    with pytest.raises(ValueError):
        evaluate_model(_predict, [])


def test_report_markdown_and_files(tmp_path):
    m, _ = evaluate_model(_predict, _batches())
    m["latency"] = {"mean_ms": 12.34, "p50_ms": 12.0, "p95_ms": 15.0}
    m2 = dict(m, pixel_auroc=None)
    md = results_to_markdown({"patchcore": m, "autoencoder": m2})
    assert "| patchcore |" in md and "| autoencoder |" in md
    assert "n/a" in md and "12.3" in md
    save_results({"patchcore": m}, tmp_path / "out")
    assert (tmp_path / "out" / "results.json").exists()
    assert (tmp_path / "out" / "results.md").read_text().startswith("| Model")
