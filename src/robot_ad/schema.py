"""Input contract robot-cycle-v1 and its validation.

Payload (JSON):
    {
      "schema_version": "robot-cycle-v1",
      "time_s": [t0, t1, ...],                      # length T, seconds, strictly increasing
      "signals": {"motor_iq_1": [...], ...}         # exactly the whitelisted channels, each length T
    }

The same function is used by the CLI, the API and the replay page.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np


class InputError(ValueError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message

    def to_dict(self) -> dict:
        return {"code": self.code, "message": self.message}


@dataclass(frozen=True)
class ValidationRules:
    schema_version: str
    channels: tuple[str, ...]
    min_length: int
    max_length: int
    dt_min_s: float
    dt_max_s: float

    @classmethod
    def from_config(cls, cfg: dict, channels: Sequence[str]) -> "ValidationRules":
        v = cfg["input_validation"]
        return cls(
            schema_version=cfg["schema_version"],
            channels=tuple(channels),
            min_length=int(v["min_length"]),
            max_length=int(v["max_length"]),
            dt_min_s=float(v["dt_min_s"]),
            dt_max_s=float(v["dt_max_s"]),
        )

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "channels": list(self.channels),
            "min_length": self.min_length,
            "max_length": self.max_length,
            "dt_min_s": self.dt_min_s,
            "dt_max_s": self.dt_max_s,
        }


def _as_float_array(values: Any, name: str) -> np.ndarray:
    if not isinstance(values, (list, tuple, np.ndarray)):
        raise InputError("invalid_type", f"{name} must be a list of numbers")
    try:
        arr = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError):
        raise InputError("invalid_type", f"{name} must contain only numbers") from None
    if arr.ndim != 1:
        raise InputError("invalid_type", f"{name} must be a flat list")
    return arr


def validate_cycle(payload: Any, rules: ValidationRules) -> tuple[np.ndarray, np.ndarray]:
    """Validate a payload and return (time_s of shape (T,), signals of shape (T, C))."""
    if not isinstance(payload, dict):
        raise InputError("invalid_type", "payload must be a JSON object")

    version = payload.get("schema_version")
    if version != rules.schema_version:
        raise InputError("unknown_schema_version", f"schema_version must be {rules.schema_version!r}, got {version!r}")

    if "time_s" not in payload:
        raise InputError("missing_field", "time_s is required")
    signals = payload.get("signals")
    if not isinstance(signals, dict):
        raise InputError("missing_field", "signals must be an object mapping channel -> list")

    missing = [c for c in rules.channels if c not in signals]
    extra = sorted(set(signals) - set(rules.channels))
    if missing:
        raise InputError("missing_channel", f"missing channels: {missing}")
    if extra:
        raise InputError("unexpected_channel", f"unexpected channels: {extra}")

    t = _as_float_array(payload["time_s"], "time_s")
    T = t.size
    if T < rules.min_length or T > rules.max_length:
        raise InputError(
            "length_out_of_range", f"cycle length {T} outside supported range [{rules.min_length}, {rules.max_length}]"
        )

    cols = []
    for ch in rules.channels:
        a = _as_float_array(signals[ch], ch)
        if a.size != T:
            raise InputError("length_mismatch", f"channel {ch} has length {a.size}, time_s has length {T}")
        cols.append(a)
    X = np.column_stack(cols)

    if not np.all(np.isfinite(t)):
        raise InputError("non_finite", "time_s contains NaN or infinite values")
    bad = [ch for ch, a in zip(rules.channels, cols) if not np.all(np.isfinite(a))]
    if bad:
        raise InputError("non_finite", f"non-finite values in channels: {bad}")

    dt = np.diff(t)
    if np.any(dt <= 0):
        raise InputError("time_not_increasing", "time_s must be strictly increasing")
    if dt.min() < rules.dt_min_s or dt.max() > rules.dt_max_s:
        raise InputError(
            "sampling_interval_out_of_range",
            f"sampling interval range [{dt.min():.5f}, {dt.max():.5f}] s outside "
            f"[{rules.dt_min_s}, {rules.dt_max_s}] s",
        )
    return t, X


def cycle_to_payload(time_s: np.ndarray, signals: np.ndarray, channels: Sequence[str], schema_version: str) -> dict:
    """Build a payload from arrays (used to export examples)."""
    def clean(a):
        return [float(v) if math.isfinite(v) else v for v in np.asarray(a, dtype=np.float64)]

    return {
        "schema_version": schema_version,
        "time_s": clean(time_s),
        "signals": {ch: clean(signals[:, i]) for i, ch in enumerate(channels)},
    }
