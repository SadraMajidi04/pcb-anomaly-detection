import numpy as np
import pytest

torch = pytest.importorskip("torch")

from pcbad.factory import to_numpy_predictor  # noqa: E402


def test_to_numpy_predictor_roundtrip():
    def fake(x):
        assert isinstance(x, torch.Tensor)
        return x.flatten(1).max(1).values, x.mean(1)
    s, m = to_numpy_predictor(fake)(np.random.rand(2, 3, 8, 8).astype(np.float32))
    assert isinstance(s, np.ndarray) and s.shape == (2,)
    assert isinstance(m, np.ndarray) and m.shape == (2, 8, 8)
