import numpy as np
import pytest

torch = pytest.importorskip("torch")

from PIL import Image  # noqa: E402

from pcbad.autoencoder import (  # noqa: E402
    ConvAutoencoder, anomaly_map, image_score, load_autoencoder, train_autoencoder,
)


def test_output_shape_matches_input():
    model = ConvAutoencoder(base=8, latent_channels=8)
    x = torch.randn(2, 3, 64, 96)
    assert model(x).shape == x.shape


def test_rejects_size_not_multiple_of_32():
    model = ConvAutoencoder(base=8, latent_channels=8)
    with pytest.raises(ValueError):
        model(torch.randn(1, 3, 50, 64))


def test_anomaly_map_properties():
    model = ConvAutoencoder(base=8, latent_channels=8)
    x = torch.randn(3, 3, 64, 64)
    amap = anomaly_map(model, x)
    assert amap.shape == (3, 64, 64)
    assert (amap >= 0).all()
    assert image_score(amap).shape == (3,)


def _make_smooth_category(tmp_path, n=6, size=64):
    root = tmp_path / "smooth"
    (root / "train" / "good").mkdir(parents=True)
    yy, xx = np.mgrid[0:size, 0:size]
    for i in range(n):
        base = (np.sin(xx / 9.0 + i * 0.1) + np.cos(yy / 11.0)) * 40 + 128
        img = np.stack([base, base * 0.9, base * 0.8], -1).clip(0, 255).astype(np.uint8)
        Image.fromarray(img).save(root / "train" / "good" / f"{i}.png")
    return root


def test_training_reduces_loss_and_checkpoint_roundtrip(tmp_path):
    root = _make_smooth_category(tmp_path)
    ckpt = tmp_path / "ae.pt"
    hist = train_autoencoder(root, image_size=64, epochs=15, batch_size=3,
                             device="cpu", out=ckpt, verbose=False)
    assert all(np.isfinite(hist["loss"]))
    assert hist["loss"][-1] < hist["loss"][0]
    model, size = load_autoencoder(ckpt, device="cpu")
    assert size == 64
    assert model(torch.randn(1, 3, 64, 64)).shape == (1, 3, 64, 64)
