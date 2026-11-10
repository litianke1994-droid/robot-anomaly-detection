"""Week 3: extract 108 cycle features (18 channels x 6 stats) for every cycle.

Outputs:
    data/processed/features_v1.parquet   (not in Git)
    reports/feature_spec.json
"""

from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd

from robot_ad.config import PROCESSED_DIR, REPORTS_DIR, load_config, raw_path
from robot_ad.data import cycle_boundaries, read_columns
from robot_ad.features import STATS, batch_features, feature_names


def main() -> None:
    cfg = load_config()
    channels = cfg["channels"]["full_18"]
    fcfg = cfg["features"]
    assert tuple(fcfg["stats"]) == STATS
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    df = read_columns(raw_path(cfg), ["sample"] + channels)
    t_read = time.perf_counter() - t0

    ids, bounds = cycle_boundaries(df["sample"].to_numpy())
    signals = df[channels].to_numpy(dtype=np.float64)
    del df

    t0 = time.perf_counter()
    X = batch_features(signals, bounds, std_ddof=fcfg["std_ddof"], quantile_method=fcfg["quantile_method"])
    t_feat = time.perf_counter() - t0

    names = feature_names(channels)
    feats = pd.DataFrame(X, index=pd.Index(ids, name="sample"), columns=names)
    assert feats.shape == (2122, 108), feats.shape
    assert np.isfinite(feats.to_numpy()).all()
    feats.to_parquet(PROCESSED_DIR / "features_v1.parquet")

    spec = {
        "schema_version": cfg["schema_version"],
        "channels": channels,
        "stats": list(STATS),
        "std_ddof": fcfg["std_ddof"],
        "quantile_method": fcfg["quantile_method"],
        "rms": "sqrt(mean(x**2))",
        "q95_abs": "np.quantile(abs(x), 0.95, method='linear')",
        "order": "channel-major: for each channel in whitelist order, the stats in listed order",
        "feature_names": names,
        "n_features": len(names),
        "n_cycles": int(len(ids)),
        "timing": {
            "read_seconds": round(t_read, 2),
            "feature_seconds_total": round(t_feat, 2),
            "feature_ms_per_cycle": round(1e3 * t_feat / len(ids), 3),
        },
    }
    with open(REPORTS_DIR / "feature_spec.json", "w", encoding="utf-8") as fh:
        json.dump(spec, fh, indent=2)
    print(f"features {feats.shape}, {spec['timing']}")


if __name__ == "__main__":
    main()
