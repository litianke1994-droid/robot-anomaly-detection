"""Project configuration and paths."""

from __future__ import annotations

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "v1.json"

RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
EXAMPLES_DIR = PROJECT_ROOT / "data" / "examples"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# Columns that must never enter the feature matrix.
FORBIDDEN_FEATURE_COLUMNS = ("sample", "anomaly", "category", "setting", "variant", "action", "active", "time")

# Category / setting names, transcribed from the official loader voraus_ad.py
# (vorausrobotik/voraus-ad-dataset, commit a91a86a). Used for reporting only.
CATEGORY_NAMES = {
    0: "AXIS_FRICTION",
    1: "AXIS_WEIGHT",
    2: "COLLISION_FOAM",
    3: "COLLISION_CABLE",
    4: "COLLISION_CARTON",
    5: "MISS_CAN",
    6: "LOSE_CAN",
    7: "CAN_WEIGHT",
    8: "ENTANGLED",
    9: "INVALID_POSITION",
    10: "MOTOR_COMMUTATION",
    11: "WOBBLING_STATION",
    12: "NORMAL_OPERATION",
}
NORMAL_SETTING_NAMES = {72: "PRE_A", 73: "PRE_B", 74: "BETWEEN_A", 75: "BETWEEN_B", 76: "BETWEEN_C"}


def load_config(path: str | Path = DEFAULT_CONFIG) -> dict:
    with open(path, encoding="utf-8") as fh:
        cfg = json.load(fh)
    for ch in cfg["channels"].values():
        bad = set(ch) & set(FORBIDDEN_FEATURE_COLUMNS)
        if bad:
            raise ValueError(f"Forbidden columns in channel whitelist: {sorted(bad)}")
    return cfg


def raw_path(cfg: dict) -> Path:
    return PROJECT_ROOT / cfg["data"]["raw_path"]
