"""Shared prediction logic used by scripts, CLI, API and the replay page."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np

from .bundle import load_bundle
from .features import cycle_features
from .schema import ValidationRules, validate_cycle


class Predictor:
    def __init__(self, bundle: dict):
        self.bundle = bundle
        self.channels: list[str] = list(bundle["channels"])
        self.threshold: float = float(bundle["threshold"]["threshold"])
        # Two-tier alerting (v1.1+): {"warning": tau_95, "alarm": tau_99}. v1 bundles only have one level.
        levels = bundle.get("threshold_levels") or {"warning": self.threshold}
        self.levels: dict[str, float] = {k: float(v) for k, v in sorted(levels.items(), key=lambda kv: kv[1])}
        self.rules = ValidationRules(**bundle["input_validation"])
        self.spec = bundle["feature_spec"]
        self.top_k = int(bundle.get("deviation_top_k", 3))

    @classmethod
    def from_path(cls, path: str | Path) -> Predictor:
        return cls(load_bundle(path))

    @property
    def model_version(self) -> str:
        return self.bundle["model_version"]

    @property
    def schema_version(self) -> str:
        return self.bundle["schema_version"]

    def features(self, signals: np.ndarray) -> np.ndarray:
        return cycle_features(signals, std_ddof=self.spec["std_ddof"], quantile_method=self.spec["quantile_method"])

    def score_features(self, X: np.ndarray) -> np.ndarray:
        return self.bundle["detector"].score(np.atleast_2d(X))

    def deviation_hints(self, x: np.ndarray) -> list[dict]:
        """Per-channel max |z| against fit_normal statistics. Auxiliary hint, not a causal explanation."""
        z = self.bundle["deviation"].zscores(np.atleast_2d(x))[0]
        names = self.bundle["feature_names"]
        n_stats = len(self.spec["stats"])
        hints = []
        for i, ch in enumerate(self.channels):
            block = np.abs(z[i * n_stats : (i + 1) * n_stats])
            j = int(np.argmax(block))
            hints.append({"channel": ch, "max_abs_z": round(float(block[j]), 3), "feature": names[i * n_stats + j]})
        hints.sort(key=lambda h: h["max_abs_z"], reverse=True)
        return hints[: self.top_k]

    def alert_level(self, score: float) -> str:
        level = "normal"
        for name, tau in self.levels.items():  # ascending thresholds
            if score > tau:
                level = name
        return level

    def predict_payload(self, payload: Any) -> dict:
        t0 = time.perf_counter()
        _, signals = validate_cycle(payload, self.rules)
        x = self.features(signals)
        t1 = time.perf_counter()
        score = float(self.score_features(x)[0])
        t2 = time.perf_counter()
        return {
            "is_anomaly": bool(score > self.threshold),
            "anomaly_score": score,
            "threshold": self.threshold,
            "alert_level": self.alert_level(score),
            "alert_levels": self.levels,
            "model_version": self.model_version,
            "schema_version": self.schema_version,
            "n_points": int(signals.shape[0]),
            "reference_deviations": self.deviation_hints(x),
            "reference_deviations_note": (
                "Standardized deviation of channel statistics from fit_normal; "
                "an inspection hint, not an explanation of the Isolation Forest score."
            ),
            "timing_ms": {"validate_and_features": round((t1 - t0) * 1e3, 3), "score": round((t2 - t1) * 1e3, 3)},
        }
