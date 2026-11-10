"""Model bundle: features spec + fitted detector + calibrated threshold, saved together."""

from __future__ import annotations

import json
import platform
import subprocess
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path

import joblib

from .config import PROJECT_ROOT

TRACKED_PACKAGES = ("numpy", "pandas", "scikit-learn", "scipy", "joblib", "pyarrow")


def git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        return out + ("-dirty" if dirty else "")
    except Exception:
        return "unknown"


def dependency_versions() -> dict:
    out = {"python": platform.python_version()}
    for p in TRACKED_PACKAGES:
        try:
            out[p] = metadata.version(p)
        except metadata.PackageNotFoundError:
            out[p] = "not-installed"
    return out


def now_utc() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def save_bundle(bundle: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)
    meta = {k: v for k, v in bundle.items() if k not in ("detector", "deviation")}
    with open(path.with_suffix(".json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2, ensure_ascii=False)


def load_bundle(path: str | Path) -> dict:
    bundle = joblib.load(path)
    required = {"model_version", "schema_version", "channels", "feature_names", "detector", "threshold", "deviation"}
    missing = required - set(bundle)
    if missing:
        raise ValueError(f"bundle missing keys: {sorted(missing)}")
    return bundle
