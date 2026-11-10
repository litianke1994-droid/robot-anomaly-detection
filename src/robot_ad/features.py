"""Cycle-level statistical features.

One cycle is an array of shape (T_i, C) holding C whitelisted channels in a
fixed order. Each channel yields six statistics, so 18 channels -> 108 features.

Conventions (frozen for schema robot-cycle-v1):
- std uses ddof=0
- rms = sqrt(mean(x**2))
- q95_abs = 95 % quantile of |x| using numpy's "linear" method
- feature name = f"{channel}__{stat}", ordered channel-major then stat order
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

STATS: tuple[str, ...] = ("mean", "std", "min", "max", "rms", "q95_abs")


def feature_names(channels: Sequence[str], stats: Sequence[str] = STATS) -> list[str]:
    return [f"{ch}__{st}" for ch in channels for st in stats]


def cycle_features(signals: np.ndarray, std_ddof: int = 0, quantile_method: str = "linear") -> np.ndarray:
    """Compute the feature vector of one cycle.

    Args:
        signals: float array, shape (T, C), T >= 1.

    Returns:
        1-D array of length C * 6, in feature_names() order.
    """
    x = np.asarray(signals, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] < 1:
        raise ValueError(f"signals must have shape (T, C) with T >= 1, got {x.shape}")
    if not np.all(np.isfinite(x)):
        raise ValueError("signals contain NaN or infinite values")

    stats = np.empty((x.shape[1], len(STATS)), dtype=np.float64)
    stats[:, 0] = x.mean(axis=0)
    stats[:, 1] = x.std(axis=0, ddof=std_ddof)
    stats[:, 2] = x.min(axis=0)
    stats[:, 3] = x.max(axis=0)
    stats[:, 4] = np.sqrt(np.mean(x * x, axis=0))
    stats[:, 5] = np.quantile(np.abs(x), 0.95, axis=0, method=quantile_method)
    return stats.reshape(-1)  # channel-major: ch0 stats, ch1 stats, ...


def batch_features(signals: np.ndarray, boundaries: np.ndarray, **kwargs) -> np.ndarray:
    """Features for many cycles stored back to back.

    Args:
        signals: shape (sum T_i, C), rows of cycle k are boundaries[k]:boundaries[k+1].
        boundaries: int array of length N + 1.
    """
    n = len(boundaries) - 1
    out = np.empty((n, signals.shape[1] * len(STATS)), dtype=np.float64)
    for k in range(n):
        out[k] = cycle_features(signals[boundaries[k] : boundaries[k + 1]], **kwargs)
    return out
