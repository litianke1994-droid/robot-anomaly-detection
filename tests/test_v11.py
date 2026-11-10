import numpy as np
import pytest

from robot_ad.bundle import save_bundle
from robot_ad.features import cycle_features, feature_names
from robot_ad.models import KNNDistance, ZScoreMax, make_detector
from robot_ad.predict import Predictor
from robot_ad.schema import ValidationRules
from robot_ad.scorer import CycleScorer

torch = pytest.importorskip("torch")

from conftest import CFG, CHANNELS, synthetic_cycle  # noqa: E402

from robot_ad.sequence import SEQ_LEN, SeqAutoencoder, resample_cycle  # noqa: E402


def test_resample_shape_and_endpoints():
    t = np.linspace(0, 10, 1000)
    x = np.column_stack([t, -t])
    r = resample_cycle(t, x)
    assert r.shape == (2, SEQ_LEN)
    assert r[0, 0] == pytest.approx(0) and r[0, -1] == pytest.approx(10)


def test_knn_leave_one_out_train_scores():
    X = np.random.default_rng(0).standard_normal((50, 4))
    det = KNNDistance(k=1).fit(X)
    assert np.all(det.train_scores_ > 0)  # not the zero self-distance


def test_scorer_single_member_keeps_ranking():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((200, 6))
    Z = rng.standard_normal((30, 6)) * 2
    det = make_detector("mahalanobis")
    sc = CycleScorer([("m", make_detector("mahalanobis"))]).fit(X)
    det.fit(X)
    assert np.array_equal(np.argsort(sc.score(Z)), np.argsort(det.score(Z)))


def _sequence_bundle(tmp_path):
    rng = np.random.default_rng(0)
    cycles = [synthetic_cycle(rng, T=int(rng.integers(1040, 1120))) for _ in range(24)]
    F = np.array([cycle_features(x) for _, x in cycles])
    S = np.stack([resample_cycle(t, x) for t, x in cycles])
    names = feature_names(CHANNELS)
    scorer = CycleScorer(
        [("maha", make_detector("mahalanobis")), ("ae", SeqAutoencoder(epochs=2, width=8, score_mode="mean"))]
    ).fit(F[:16], S[:16], names)
    cal = scorer.score(F[16:], S[16:])
    levels = {"warning": float(np.quantile(cal, 0.95)), "alarm": float(np.quantile(cal, 0.99))}
    bundle = {
        "model_version": "test-v11",
        "schema_version": CFG["schema_version"],
        "channels": CHANNELS,
        "feature_names": names,
        "feature_spec": {"stats": CFG["features"]["stats"], "std_ddof": 0, "quantile_method": "linear"},
        "scorer": scorer,
        "detector": scorer,
        "deviation": ZScoreMax().fit(F[:16], names),
        "threshold": {"threshold": levels["warning"]},
        "threshold_levels": levels,
        "input_validation": ValidationRules.from_config(CFG, CHANNELS).to_dict(),
    }
    path = tmp_path / "v11.joblib"
    save_bundle(bundle, path)
    return path, cycles, scorer


def test_sequence_bundle_roundtrip(tmp_path, valid_payload):
    path, cycles, scorer = _sequence_bundle(tmp_path)
    p1 = Predictor.from_path(path).predict_payload(valid_payload)
    p2 = Predictor.from_path(path).predict_payload(valid_payload)
    assert p1["anomaly_score"] == pytest.approx(p2["anomaly_score"], abs=1e-6)
    assert p1["alert_level"] in {"normal", "warning", "alarm"}
    # Predictor path equals direct scorer call on the same cycle
    t = np.asarray(valid_payload["time_s"])
    x = np.column_stack([valid_payload["signals"][c] for c in CHANNELS])
    direct = scorer.score(cycle_features(x)[None], resample_cycle(t, x)[None])[0]
    assert p1["anomaly_score"] == pytest.approx(float(direct), abs=1e-5)
