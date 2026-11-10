"""CycleScorer: one or more detectors over features and/or sequences, combined into one score.

Each member's score is normalized with its own normal-fit statistics,
(s - median) / (q95 - median), and the members are averaged. With a single
member the normalization is a monotone rescaling, so ranking is unchanged.
"""

from __future__ import annotations

import numpy as np


class CycleScorer:
    def __init__(self, members: list[tuple[str, object]]):
        self.members = members  # [(label, detector)], detector.input_kind in {"features", "sequence"}

    @property
    def needs_sequence(self) -> bool:
        return any(getattr(d, "input_kind", "features") == "sequence" for _, d in self.members)

    @staticmethod
    def _input(det, F, S):
        return S if getattr(det, "input_kind", "features") == "sequence" else F

    def fit(self, F: np.ndarray, S: np.ndarray | None = None, feature_names=None) -> CycleScorer:
        self.norm_ = []
        for _, det in self.members:
            X = self._input(det, F, S)
            det.fit(X, feature_names)
            # Leave-one-out training scores where the detector provides them (kNN, LOF); in-sample otherwise.
            s = getattr(det, "train_scores_", None)
            s = det.score(X) if s is None else s
            med, q95 = float(np.median(s)), float(np.quantile(s, 0.95))
            self.norm_.append((med, max(q95 - med, 1e-12)))
        return self

    def member_scores(self, F, S=None) -> dict[str, np.ndarray]:
        return {label: det.score(self._input(det, F, S)) for label, det in self.members}

    def score(self, F: np.ndarray, S: np.ndarray | None = None) -> np.ndarray:
        parts = [
            (det.score(self._input(det, F, S)) - med) / scale
            for (_, det), (med, scale) in zip(self.members, self.norm_)
        ]
        return np.mean(parts, axis=0)

    def params(self) -> dict:
        return {label: det.params() for label, det in self.members}
