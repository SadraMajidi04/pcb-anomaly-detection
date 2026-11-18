import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

from pcbad.autoencoder import ConvAutoencoder, anomaly_map  # noqa: E402
from pcbad.patchcore import PatchCore, gaussian_blur  # noqa: E402


def _fitted_patchcore():
    g = torch.Generator().manual_seed(0)
    imgs = torch.randn(3, 3, 64, 64, generator=g).clamp(-2, 2)
    m = PatchCore(backbone="resnet18", pretrained=False, coreset_ratio=0.2, device="cpu")
    m.fit([{"image": imgs}], verbose=False)
    return m


def test_autoencoder_finite_on_constant_and_extreme_inputs():
    model = ConvAutoencoder(base=8, latent_channels=8)
    for x in (torch.zeros(1, 3, 64, 64), torch.full((1, 3, 64, 64), 5.0),
              torch.full((1, 3, 64, 64), -5.0)):
        amap = anomaly_map(model, x)
        assert amap.shape == (1, 64, 64) and torch.isfinite(amap).all()
        assert not amap.requires_grad


def test_autoencoder_deterministic_in_eval():
    model = ConvAutoencoder(base=8, latent_channels=8)
    x = torch.randn(2, 3, 64, 64)
    assert torch.equal(anomaly_map(model, x), anomaly_map(model, x))


def test_autoencoder_rejects_bad_rank_and_channels():
    model = ConvAutoencoder(base=8, latent_channels=8)
    with pytest.raises(ValueError):
        model(torch.randn(3, 64, 64))
    with pytest.raises(ValueError):
        model(torch.randn(1, 1, 64, 64))


def test_patchcore_handles_other_resolutions_and_batch_of_one():
    m = _fitted_patchcore()
    for size in (32, 64, 96):
        s, amap = m.predict(torch.randn(1, 3, size, size).clamp(-2, 2))
        assert s.shape == (1,) and amap.shape == (1, size, size)
        assert torch.isfinite(amap).all()


def test_patchcore_constant_image_is_finite():
    m = _fitted_patchcore()
    s, amap = m.predict(torch.zeros(2, 3, 64, 64))
    assert torch.isfinite(s).all() and torch.isfinite(amap).all()


def test_patchcore_rejects_tiny_and_malformed_input():
    m = _fitted_patchcore()
    with pytest.raises(ValueError):
        m.predict(torch.randn(1, 3, 16, 16))
    with pytest.raises(ValueError):
        m.predict(torch.randn(3, 64, 64))
    with pytest.raises(ValueError):
        m.predict(torch.randn(1, 1, 64, 64))


def test_patchcore_deterministic():
    m = _fitted_patchcore()
    x = torch.randn(2, 3, 64, 64).clamp(-2, 2)
    assert torch.equal(m.predict(x)[0], m.predict(x)[0])


def test_gaussian_blur_on_tiny_map_does_not_crash():
    x = torch.rand(1, 1, 8, 8)
    assert gaussian_blur(x, sigma=4.0).shape == x.shape
    assert gaussian_blur(torch.rand(1, 1, 1, 1), sigma=4.0).shape == (1, 1, 1, 1)
