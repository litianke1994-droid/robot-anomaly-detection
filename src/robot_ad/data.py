"""Reading the voraus-AD Parquet file column by column."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

META_COLUMNS = ["sample", "time", "anomaly", "category", "setting"]


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_columns(path: str | Path, columns: Sequence[str]) -> pd.DataFrame:
    return pd.read_parquet(path, columns=list(columns))


def cycle_boundaries(sample_col: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (sample_ids, boundaries) for a sample column that is grouped contiguously."""
    sample_col = np.asarray(sample_col)
    change = np.flatnonzero(np.diff(sample_col) != 0) + 1
    starts = np.concatenate([[0], change])
    ids = sample_col[starts]
    if len(np.unique(ids)) != len(ids):
        raise ValueError("sample ids are not stored contiguously; sort by sample first")
    boundaries = np.concatenate([starts, [len(sample_col)]])
    return ids, boundaries


def cycle_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """One row per sample with label/setting and length; asserts labels are constant within a cycle."""
    g = df.groupby("sample", sort=True)
    meta = g.agg(
        n_points=("time", "size"),
        t_start=("time", "first"),
        t_end=("time", "last"),
        anomaly=("anomaly", "first"),
        category=("category", "first"),
        setting=("setting", "first"),
        n_anomaly_values=("anomaly", "nunique"),
        n_category_values=("category", "nunique"),
        n_setting_values=("setting", "nunique"),
    )
    meta["duration_s"] = meta["t_end"] - meta["t_start"]
    return meta
