import numpy as np
import pytest

from robot_ad.features import STATS, batch_features, cycle_features, feature_names


def test_hand_calculated_example():
    # one channel: [3, -4, 0, 1]
    x = np.array([[3.0], [-4.0], [0.0], [1.0]])
    f = dict(zip(STATS, cycle_features(x)))
    assert f["mean"] == pytest.approx(0.0)
    assert f["std"] == pytest.approx(np.sqrt((9 + 16 + 0 + 1) / 4))  # ddof=0
    assert f["min"] == -4.0
    assert f["max"] == 3.0
    assert f["rms"] == pytest.approx(np.sqrt(26 / 4))
    # |x| sorted = [0, 1, 3, 4]; linear q95 at position 0.95*3 = 2.85 -> 3 + 0.85*(4-3)
    assert f["q95_abs"] == pytest.approx(3.85)


def test_shape_and_order():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((100, 3))
    f = cycle_features(x)
    assert f.shape == (18,)
    assert f[6] == pytest.approx(x[:, 1].mean())  # channel-major order
    assert feature_names(["a", "b"])[:2] == ["a__mean", "a__std"]
    assert len(feature_names([f"c{i}" for i in range(18)])) == 108


def test_batch_matches_single():
    rng = np.random.default_rng(1)
    a, b = rng.standard_normal((50, 4)), rng.standard_normal((70, 4))
    out = batch_features(np.vstack([a, b]), np.array([0, 50, 120]))
    np.testing.assert_allclose(out[0], cycle_features(a))
    np.testing.assert_allclose(out[1], cycle_features(b))


def test_rejects_non_finite():
    with pytest.raises(ValueError):
        cycle_features(np.array([[1.0], [np.nan]]))
