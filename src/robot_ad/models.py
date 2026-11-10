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

    def fit(self, X: np.ndarray, feature_names: list[str] | None = None) -> ZScoreMax:
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

    def fit(self, X: np.ndarray, feature_names: list[str] | None = None) -> PCARecon:
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

    def fit(self, X: np.ndarray, feature_names: list[str] | None = None) -> IForest:
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


class _Scaled:
    """Base for detectors that work on StandardScaler-transformed features (scaler fitted on normal data)."""

    def _fit_scaler(self, X: np.ndarray) -> np.ndarray:
        self.scaler_ = StandardScaler().fit(X)
        return self.scaler_.transform(X)


class Mahalanobis(_Scaled):
    """Squared Mahalanobis distance with a Ledoit-Wolf shrunk covariance of normal features."""

    name = "mahalanobis"

    def fit(self, X, feature_names=None):
        from sklearn.covariance import LedoitWolf

        self.cov_ = LedoitWolf().fit(self._fit_scaler(X))
        return self

    def score(self, X):
        return self.cov_.mahalanobis(self.scaler_.transform(X))

    def params(self):
        return {"covariance": "LedoitWolf", "shrinkage": float(self.cov_.shrinkage_)}


class KNNDistance(_Scaled):
    """Mean Euclidean distance to the k nearest normal training cycles (standardized space)."""

    name = "knn"

    def __init__(self, k: int = 5):
        self.k = k

    def fit(self, X, feature_names=None):
        from sklearn.neighbors import NearestNeighbors

        Z = self._fit_scaler(X)
        self.nn_ = NearestNeighbors(n_neighbors=self.k).fit(Z)
        # Leave-one-out training scores: a training point's nearest neighbour is itself (distance 0).
        d, _ = self.nn_.kneighbors(Z, n_neighbors=self.k + 1)
        self.train_scores_ = d[:, 1:].mean(axis=1)
        return self

    def score(self, X):
        d, _ = self.nn_.kneighbors(self.scaler_.transform(X))
        return d.mean(axis=1)

    def params(self):
        return {"k": self.k}


class LOF(_Scaled):
    """Local Outlier Factor in novelty mode; score = -score_samples."""

    name = "lof"

    def __init__(self, n_neighbors: int = 20):
        self.n_neighbors = n_neighbors

    def fit(self, X, feature_names=None):
        from sklearn.neighbors import LocalOutlierFactor

        self.model_ = LocalOutlierFactor(n_neighbors=self.n_neighbors, novelty=True).fit(self._fit_scaler(X))
        self.train_scores_ = -self.model_.negative_outlier_factor_  # training LOF excludes the point itself
        return self

    def score(self, X):
        return -self.model_.score_samples(self.scaler_.transform(X))

    def params(self):
        return {"n_neighbors": self.n_neighbors}


class OCSVM(_Scaled):
    """One-class SVM with RBF kernel; score = -decision_function."""

    name = "ocsvm"

    def __init__(self, nu: float = 0.05, gamma: str | float = "scale"):
        self.nu = nu
        self.gamma = gamma

    def fit(self, X, feature_names=None):
        from sklearn.svm import OneClassSVM

        self.model_ = OneClassSVM(nu=self.nu, gamma=self.gamma).fit(self._fit_scaler(X))
        return self

    def score(self, X):
        return -self.model_.decision_function(self.scaler_.transform(X))

    def params(self):
        return {"nu": self.nu, "gamma": self.gamma}


class ECOD:
    """Empirical-CDF outlier score (ECOD idea, Li et al. 2022), implemented with numpy.

    For each feature, left and right tail probabilities of a value are estimated from
    the normal training distribution; the score sums -log tail probabilities, taking
    for each feature the tail indicated by the training skewness (plus the max of
    left/right/skew-corrected aggregates, as in the original method).
    """

    name = "ecod"

    def fit(self, X, feature_names=None):
        self.sorted_ = np.sort(X, axis=0)
        n = X.shape[0]
        self.n_ = n
        from scipy.stats import skew

        self.skew_sign_ = np.sign(skew(X, axis=0))
        return self

    def _tails(self, X):
        n = self.n_
        left = np.empty_like(X, dtype=np.float64)
        right = np.empty_like(X, dtype=np.float64)
        for j in range(X.shape[1]):
            col = self.sorted_[:, j]
            left[:, j] = np.searchsorted(col, X[:, j], side="right") / (n + 1)
            right[:, j] = (n - np.searchsorted(col, X[:, j], side="left")) / (n + 1)
        eps = 1.0 / (n + 1)
        return -np.log(np.clip(left, eps, 1)), -np.log(np.clip(right, eps, 1))

    def score(self, X):
        ul, ur = self._tails(np.asarray(X, dtype=np.float64))
        uskew = np.where(self.skew_sign_ < 0, ul, ur)
        return np.maximum.reduce([ul.sum(axis=1), ur.sum(axis=1), uskew.sum(axis=1)])

    def params(self):
        return {"n_train": int(self.n_)}


class Combo:
    """Mean of member scores, each normalized as (s - median) / (q95 - median) of its normal training scores."""

    name = "combo"

    def __init__(self, members: list, member_names: list[str]):
        self.members = members
        self.member_names = member_names

    def fit(self, X, feature_names=None):
        self.norm_ = []
        for m in self.members:
            m.fit(X, feature_names)
            s = m.score(X)
            med, q95 = np.median(s), np.quantile(s, 0.95)
            self.norm_.append((float(med), float(max(q95 - med, 1e-12))))
        return self

    def score(self, X):
        parts = [(m.score(X) - med) / scale for m, (med, scale) in zip(self.members, self.norm_)]
        return np.mean(parts, axis=0)

    def params(self):
        return {"members": self.member_names, "normalization": "(s - median_fit) / (q95_fit - median_fit)"}


DETECTORS = {
    "zscore_max": ZScoreMax,
    "pca_recon": PCARecon,
    "iforest": IForest,
    "mahalanobis": Mahalanobis,
    "knn": KNNDistance,
    "lof": LOF,
    "ocsvm": OCSVM,
    "ecod": ECOD,
}


def make_detector(name: str, cfg_models: dict | None = None, **overrides):
    """Build a detector. `name` may be 'combo:a+b' to average normalized member scores."""
    cfg_models = cfg_models or {}
    if name.startswith("combo:"):
        member_names = name.split(":", 1)[1].split("+")
        return Combo([make_detector(m, cfg_models) for m in member_names], member_names)
    if name not in DETECTORS:
        raise ValueError(f"unknown detector {name!r}")
    kwargs = {**cfg_models.get(name, {}), **overrides}
    return DETECTORS[name](**kwargs)


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
