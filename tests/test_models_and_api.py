import json
import subprocess
import sys

import numpy as np
import pytest
from fastapi.testclient import TestClient

from robot_ad.models import PCARecon, ZScoreMax, calibrate_threshold, make_detector
from robot_ad.predict import Predictor


def test_threshold_rule():
    thr = calibrate_threshold(np.arange(1, 101, dtype=float), 0.95)
    assert thr["threshold"] == pytest.approx(95.05)
    assert thr["n_cal_above"] == 5
    assert thr["rule"] == "score > threshold"


@pytest.mark.parametrize(
    "name",
    ["zscore_max", "pca_recon", "iforest", "mahalanobis", "knn", "lof", "ocsvm", "ecod", "combo:zscore_max+pca_recon"],
)
def test_score_direction(name):
    rng = np.random.default_rng(0)
    X = rng.standard_normal((300, 10))
    det = make_detector(name, {"pca_recon": {"variance": 0.9}, "iforest": {"n_estimators": 100, "random_state": 0}})
    det.fit(X)
    s_normal = det.score(rng.standard_normal((50, 10))).mean()
    s_far = det.score(rng.standard_normal((50, 10)) + 6).mean()
    assert s_far > s_normal


def test_zscore_constant_column_excluded():
    X = np.column_stack([np.random.default_rng(0).standard_normal(50), np.ones(50)])
    det = ZScoreMax().fit(X, ["a", "b"])
    assert det.constant_features_ == ["b"]
    assert np.isfinite(det.score(np.array([[0.0, 5.0]]))).all()


def test_pca_keeps_variance():
    X = np.random.default_rng(0).standard_normal((200, 8))
    assert PCARecon(0.95).fit(X).pca_.explained_variance_ratio_.sum() >= 0.95


def test_reload_is_deterministic(bundle_path, valid_payload):
    a = Predictor.from_path(bundle_path).predict_payload(valid_payload)
    b = Predictor.from_path(bundle_path).predict_payload(valid_payload)
    assert a["anomaly_score"] == b["anomaly_score"]
    assert set(a) >= {"is_anomaly", "anomaly_score", "threshold", "model_version", "schema_version"}


def test_cli(bundle_path, valid_payload, tmp_path):
    f = tmp_path / "cycle.json"
    f.write_text(json.dumps(valid_payload), encoding="utf-8")
    out = subprocess.run(
        [sys.executable, "-m", "robot_ad.cli", "--bundle", str(bundle_path), "predict", "--input", str(f)],
        capture_output=True,
        text=True,
        check=True,
    )
    res = json.loads(out.stdout)
    ref = Predictor.from_path(bundle_path).predict_payload(valid_payload)
    assert abs(res["anomaly_score"] - ref["anomaly_score"]) <= 1e-6


def test_api(bundle_path, valid_payload, monkeypatch):
    import api.main as m

    monkeypatch.setattr(m, "BUNDLE_PATH", str(bundle_path))
    monkeypatch.setattr(m, "_state", {"predictor": None, "load_error": None})
    with TestClient(m.app) as client:
        h = client.get("/health")
        assert h.status_code == 200 and h.json()["model_version"] == "test-iforest"
        r = client.post("/predict", json=valid_payload)
        assert r.status_code == 200
        ref = Predictor.from_path(bundle_path).predict_payload(valid_payload)
        assert abs(r.json()["anomaly_score"] - ref["anomaly_score"]) <= 1e-6
        bad = dict(valid_payload, schema_version="x")
        r = client.post("/predict", json=bad)
        assert r.status_code == 422 and r.json()["error"]["code"] == "unknown_schema_version"
        r = client.post("/predict", content=b"{not json", headers={"content-type": "application/json"})
        assert r.status_code == 422 and r.json()["error"]["code"] == "invalid_json"


def test_alert_levels(bundle_path):
    p = Predictor.from_path(bundle_path)
    p.levels = {"warning": 1.0, "alarm": 2.0}
    assert [p.alert_level(s) for s in (0.5, 1.5, 2.5)] == ["normal", "warning", "alarm"]


def test_api_not_ready(monkeypatch, tmp_path):
    import api.main as m

    monkeypatch.setattr(m, "BUNDLE_PATH", str(tmp_path / "missing.joblib"))
    monkeypatch.setattr(m, "_state", {"predictor": None, "load_error": None})
    with TestClient(m.app) as client:
        assert client.get("/health").status_code == 503
