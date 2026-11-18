"""Edge cases seen in real datasets (VisA layout, upper-case extensions, junk files)."""
import numpy as np
import pytest
from PIL import Image

from pcbad.data import index_split, load_mask


def _rgb(path, w=30, h=20):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.full((h, w, 3), 90, np.uint8)).save(path)


def _mask(path, w=30, h=20):
    path.parent.mkdir(parents=True, exist_ok=True)
    m = np.zeros((h, w), np.uint8)
    m[5:10, 5:10] = 255
    Image.fromarray(m).save(path)


def test_visa_layout_uppercase_jpg_and_stem_named_png_masks(tmp_path):
    root = tmp_path / "pcb1"
    _rgb(root / "train" / "good" / "0000.JPG")
    _rgb(root / "test" / "good" / "0001.JPG")
    _rgb(root / "test" / "bad" / "0002.JPG")
    _mask(root / "ground_truth" / "bad" / "0002.png")
    train, test = index_split(root, "train"), index_split(root, "test")
    assert len(train) == 1 and len(test) == 2
    bad = [s for s in test if s.label == 1][0]
    assert bad.defect_type == "bad" and bad.mask_path is not None
    assert load_mask(bad, 16).sum() > 0


def test_uppercase_mask_extension_is_found(tmp_path):
    root = tmp_path / "c"
    _rgb(root / "test" / "good" / "a.png")
    _rgb(root / "test" / "x" / "b.png")
    _mask(root / "ground_truth" / "x" / "b_mask.PNG")
    s = [s for s in index_split(root, "test") if s.label == 1][0]
    assert s.mask_path is not None


def test_junk_files_and_nested_dirs_are_ignored(tmp_path):
    root = tmp_path / "c"
    _rgb(root / "train" / "good" / "a.png")
    (root / "train" / "good" / ".DS_Store").write_bytes(b"junk")
    (root / "train" / "good" / "notes.txt").write_text("hi")
    (root / "train" / "good" / "subdir").mkdir()
    _rgb(root / "train" / "good" / "subdir" / "hidden.png")
    assert len(index_split(root, "train")) == 1


def test_defective_image_without_mask_has_none(tmp_path):
    root = tmp_path / "c"
    _rgb(root / "test" / "good" / "a.png")
    _rgb(root / "test" / "x" / "b.png")
    s = [s for s in index_split(root, "test") if s.label == 1][0]
    assert s.mask_path is None
    assert load_mask(s, 8).sum() == 0


def test_empty_split_folder_raises(tmp_path):
    (tmp_path / "train" / "good").mkdir(parents=True)
    with pytest.raises(ValueError):
        index_split(tmp_path, "train")


def test_deterministic_sorted_order(tmp_path):
    root = tmp_path / "c"
    for n in ("c", "a", "b"):
        _rgb(root / "train" / "good" / f"{n}.png")
    names = [s.image_path.name for s in index_split(root, "train")]
    assert names == sorted(names)
