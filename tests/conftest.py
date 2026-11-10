"""Synthetic fixtures: tests never need the real dataset."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # top-level api package

from robot_ad.bundle import save_bundle  # noqa: E402
from robot_ad.config import load_config  # noqa: E402
from robot_ad.features import cycle_features, feature_names  # noqa: E402
from robot_ad.models import IForest, ZScoreMax, calibrate_threshold  # noqa: E402
from robot_ad.schema import ValidationRules, cycle_to_payload  # noqa: E402

CFG = load_config()
CHANNELS = CFG["channels"]["full_18"]


def synthetic_cycle(rng: np.random.Generator, T: int = 1080, amp: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    t = np.arange(T) * 0.01
    base = np.sin(np.outer(t, np.linspace(0.5, 2.0, len(CHANNELS))))
    x = amp * base + 0.05 * rng.standard_normal((T, len(CHANNELS)))
    return t, x


@pytest.fixture(scope="session")
def rules() -> ValidationRules:
    return ValidationRules.from_config(CFG, CHANNELS)


@pytest.fixture(scope="session")
def valid_payload() -> dict:
    t, x = synthetic_cycle(np.random.default_rng(1))
    return cycle_to_payload(t, x, CHANNELS, CFG["schema_version"])


@pytest.fixture(scope="session")
def bundle_path(tmp_path_factory) -> Path:
    rng = np.random.default_rng(0)
    X = np.array([cycle_features(synthetic_cycle(rng, T=int(rng.integers(1040, 1120)))[1]) for _ in range(80)])
    fitX, calX = X[:60], X[60:]
    names = feature_names(CHANNELS)
    det = IForest(n_estimators=50, random_state=0).fit(fitX)
    bundle = {
        "model_version": "test-iforest",
        "schema_version": CFG["schema_version"],
        "model_name": "iforest",
        "channel_set": "full_18",
        "is_primary": True,
        "channels": CHANNELS,
        "feature_names": names,
        "feature_spec": {"stats": CFG["features"]["stats"], "std_ddof": 0, "quantile_method": "linear"},
        "detector": det,
        "deviation": ZScoreMax().fit(fitX, names),
        "deviation_top_k": 3,
        "threshold": calibrate_threshold(det.score(calX)),
        "input_validation": ValidationRules.from_config(CFG, CHANNELS).to_dict(),
        "git_commit": "test",
    }
    path = tmp_path_factory.mktemp("models") / "bundle.joblib"
    save_bundle(bundle, path)
    return path
