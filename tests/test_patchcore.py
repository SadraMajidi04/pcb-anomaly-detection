import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

from pcbad.patchcore import (  # noqa: E402
    PatchCore, gaussian_blur, greedy_coreset, nn_distance,
)


def test_greedy_coreset_unique_and_size():
    feats = torch.randn(500, 64)
    idx = greedy_coreset(feats, 50)
    assert idx.shape == (50,)
    assert len(set(idx.tolist())) == 50


def test_greedy_coreset_with_duplicate_points_stays_unique():
    feats = torch.zeros(100, 8)
    idx = greedy_coreset(feats, 20)
    assert len(set(idx.tolist())) == 20


def test_greedy_coreset_k_ge_n_returns_all():
    assert greedy_coreset(torch.randn(10, 4), 10).tolist() == list(range(10))
    assert greedy_coreset(torch.randn(10, 4), 99).shape == (10,)


def test_nn_distance_matches_bruteforce():
    q, b = torch.randn(37, 16), torch.randn(23, 16)
    expected = torch.stack([(b - qi).norm(dim=1).min() for qi in q])
    assert torch.allclose(nn_distance(q, b, chunk=10), expected, atol=1e-4)


def test_gaussian_blur_shape_and_constant_preserved():
    x = torch.full((2, 1, 64, 64), 3.0)
    y = gaussian_blur(x, sigma=4.0)
    assert y.shape == x.shape
    assert torch.allclose(y, x, atol=1e-4)


def _smooth_images(n=4, size=64):
    yy, xx = torch.meshgrid(torch.arange(size), torch.arange(size), indexing="ij")
    imgs = []
    for i in range(n):
        base = torch.sin(xx / 9.0 + i * 0.3) + torch.cos(yy / 11.0)
        imgs.append(torch.stack([base, base * 0.5, -base]).float())
    return torch.stack(imgs)


def _model(ratio=1.0):
    return PatchCore(backbone="resnet18", pretrained=False, coreset_ratio=ratio,
                     device="cpu", seed=0)


def test_defect_scores_higher_than_good_training_image():
    imgs = _smooth_images()
    m = _model(1.0)
    m.fit([{"image": imgs}], verbose=False)
    good = imgs[:1].clone()
    bad = imgs[:1].clone()
    bad[:, :, 24:40, 24:40] += 3.0
    s_good, _ = m.predict(good)
    s_bad, amap = m.predict(bad)
    assert amap.shape == (1, 64, 64)
    assert s_bad.item() > s_good.item()
    assert s_good.item() < 0.2 * s_bad.item()


def test_heatmap_peaks_near_defect():
    imgs = _smooth_images()
    m = _model(1.0)
    m.fit([{"image": imgs}], verbose=False)
    bad = imgs[:1].clone()
    bad[:, :, 24:40, 24:40] += 3.0
    _, amap = m.predict(bad)
    inside = amap[0, 20:44, 20:44].mean()
    outside = torch.cat([amap[0, :12].flatten(), amap[0, -12:].flatten()]).mean()
    assert inside > outside


def test_coreset_shrinks_bank():
    imgs = _smooth_images()
    full = _model(1.0)
    small = _model(0.1)
    n_full = full.fit([{"image": imgs}], verbose=False)
    n_small = small.fit([{"image": imgs}], verbose=False)
    assert n_small < n_full


def test_predict_before_fit_raises():
    with pytest.raises(RuntimeError):
        _model().predict(torch.randn(1, 3, 64, 64))


def test_invalid_ratio_and_backbone():
    with pytest.raises(ValueError):
        PatchCore(backbone="resnet18", pretrained=False, coreset_ratio=0.0)
    with pytest.raises(ValueError):
        PatchCore(backbone="vgg", pretrained=False)


def test_save_load_roundtrip(tmp_path):
    imgs = _smooth_images()
    m = _model(0.5)
    m.fit([{"image": imgs}], verbose=False)
    path = tmp_path / "pc.pt"
    m.save(path)
    m2 = PatchCore.load(path, device="cpu", pretrained=False)
    x = imgs[:2]
    s1, _ = m.predict(x)
    s2, _ = m2.predict(x)
    assert torch.allclose(s1, s2, atol=1e-5)
