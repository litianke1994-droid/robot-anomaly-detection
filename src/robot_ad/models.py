"""Three normal-only anomaly detectors with a common interface.

Every detector is fitted on normal cycles only and returns a score where
larger means more anomalous.
"""

from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


class ZScoreMax:
    """Score = max |z| over features, z computed with the fit_normal scaler.

    Constant-column rule: a feature whose fit_normal std <= eps is excluded
    from the max (its z would be undefined); the excluded names are stored.
    """

    name = "zscore_max"

    def __init__(self, constant_std_eps: float = 1e-12):
        self.constant_std_eps = constant_std_eps

    def fit(self, X: np.ndarray, feature_names: list[str] | None = None) -> "ZScoreMax":
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0, ddof=0)
        self.active_ = self.std_ > self.constant_std_eps
        names = feature_names or [str(i) for i in range(X.shape[1])]
        self.constant_features_ = [n for n, a in zip(names, self.active_) if not a]
        return self

    def zscores(self, X: np.ndarray) -> np.ndarray:
        z = np.zeros_like(X, dtype=np.float64)
        a = self.active_
        z[:, a] = (X[:, a] - self.mean_[a]) / self.std_[a]
        return z

    def score(self, X: np.ndarray) -> np.ndarray:
        return np.abs(self.zscores(X)).max(axis=1)

    def params(self) -> dict:
        return {"constant_std_eps": self.constant_std_eps, "constant_features": self.constant_features_}


class PCARecon:
    """StandardScaler + PCA keeping a fixed fraction of fit_normal variance.

    Score = mean squared reconstruction error in standardized feature space.
    """

    name = "pca_recon"

    def __init__(self, variance: float = 0.95, svd_solver: str = "full"):
        self.variance = variance
        self.svd_solver = svd_solver

    def fit(self, X: np.ndarray, feature_names: list[str] | None = None) -> "PCARecon":
        self.scaler_ = StandardScaler().fit(X)
        self.pca_ = PCA(n_components=self.variance, svd_solver=self.svd_solver).fit(self.scaler_.transform(X))
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        Z = self.scaler_.transform(X)
        R = self.pca_.inverse_transform(self.pca_.transform(Z))
        return np.mean((Z - R) ** 2, axis=1)

    def params(self) -> dict:
        return {
            "variance": self.variance,
            "svd_solver": self.svd_solver,
            "n_components": int(self.pca_.n_components_),
            "explained_variance": float(self.pca_.explained_variance_ratio_.sum()),
        }


class IForest:
    """Isolation Forest on raw features. Score = -score_samples (larger = more anomalous).

    The sklearn predict()/contamination threshold is NOT used; the threshold
    comes from the independent cal_normal set.
    """

    name = "iforest"

    def __init__(self, n_estimators: int = 300, max_samples="auto", max_features: float = 1.0, random_state: int = 42):
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.max_features = max_features
        self.random_state = random_state

    def fit(self, X: np.ndarray, feature_names: list[str] | None = None) -> "IForest":
        self.model_ = IsolationForest(
            n_estimators=self.n_estimators,
            max_samples=self.max_samples,
            max_features=self.max_features,
            random_state=self.random_state,
            n_jobs=1,
        ).fit(X)
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        return -self.model_.score_samples(X)

    def params(self) -> dict:
        return {
            "n_estimators": self.n_estimators,
            "max_samples": self.max_samples,
            "max_features": self.max_features,
            "random_state": self.random_state,
        }


def make_detector(name: str, cfg_models: dict):
    kwargs = dict(cfg_models.get(name, {}))
    if name == "zscore_max":
        return ZScoreMax(**kwargs)
    if name == "pca_recon":
        return PCARecon(**kwargs)
    if name == "iforest":
        return IForest(**kwargs)
    raise ValueError(f"unknown detector {name!r}")


def calibrate_threshold(cal_scores: np.ndarray, quantile: float = 0.95, method: str = "linear") -> dict:
    """Threshold tau = quantile of cal_normal scores; alarm iff score > tau."""
    cal_scores = np.asarray(cal_scores, dtype=np.float64)
    tau = float(np.quantile(cal_scores, quantile, method=method))
    return {
        "threshold": tau,
        "quantile": quantile,
        "quantile_method": method,
        "rule": "score > threshold",
        "n_cal": int(cal_scores.size),
        "n_cal_above": int(np.sum(cal_scores > tau)),
        "n_cal_equal": int(np.sum(cal_scores == tau)),
        "cal_alarm_rate": float(np.mean(cal_scores > tau)),
    }
