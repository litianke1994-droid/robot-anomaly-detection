"""Week 2: data audit. Reads only the meta columns and the 18 whitelisted channels.

Outputs:
    reports/audit_summary.json
    reports/cycle_metadata.csv
    reports/figures/single_cycle_normal.png
"""

from __future__ import annotations

import json
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from robot_ad.config import CATEGORY_NAMES, FIGURES_DIR, NORMAL_SETTING_NAMES, REPORTS_DIR, load_config, raw_path
from robot_ad.data import META_COLUMNS, cycle_boundaries, cycle_metadata, read_columns, sha256_file

PAPER_COUNTS = {"train_normal": 948, "test_normal": 419, "test_anomaly": 755}


def main() -> None:
    cfg = load_config()
    path = raw_path(cfg)
    channels = cfg["channels"]["full_18"]
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    sha = sha256_file(path)
    t_hash = time.perf_counter() - t0
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(path)
    schema = {f.name: str(f.type) for f in pf.schema_arrow}

    t0 = time.perf_counter()
    df = read_columns(path, META_COLUMNS + channels)
    t_read = time.perf_counter() - t0
    mem_mb = df.memory_usage(deep=True).sum() / 2**20

    # --- structural checks -------------------------------------------------
    ids, bounds = cycle_boundaries(df["sample"].to_numpy())
    meta = cycle_metadata(df)
    label_consistent = bool(
        (meta[["n_anomaly_values", "n_category_values", "n_setting_values"]] == 1).all().all()
    )
    normal_cat_consistent = bool(((meta["category"] == 12) == (~meta["anomaly"])).all())

    dt = df.groupby("sample")["time"].diff().dropna()
    sig = df[channels].to_numpy()
    n_nan = {c: int(df[c].isna().sum()) for c in channels}
    n_inf = {c: int(np.isinf(df[c].to_numpy()).sum()) for c in channels}

    # constant channels within a cycle, duplicate cycles
    const_cycles = {c: 0 for c in channels}
    digests = []
    for k in range(len(ids)):
        block = sig[bounds[k] : bounds[k + 1]]
        rng = block.max(axis=0) - block.min(axis=0)
        for c, r in zip(channels, rng):
            if r == 0:
                const_cycles[c] += 1
        digests.append(hash(np.ascontiguousarray(block).tobytes()))
    n_dup = int(len(digests) - len(set(digests)))
    dup_rows = int(df.duplicated(subset=["sample", "time"]).sum())

    # --- split counts vs. paper -------------------------------------------
    train_code = cfg["data"]["train_setting_code"]
    is_train = meta["setting"] == train_code
    counts = {
        "train_normal": int((is_train & ~meta["anomaly"]).sum()),
        "train_anomaly": int((is_train & meta["anomaly"]).sum()),
        "test_normal": int((~is_train & ~meta["anomaly"]).sum()),
        "test_anomaly": int((~is_train & meta["anomaly"]).sum()),
    }
    per_category = {
        CATEGORY_NAMES[int(k)]: int(v) for k, v in meta[~is_train].groupby("category").size().items()
    }
    normal_settings = {
        NORMAL_SETTING_NAMES.get(int(k), str(k)): int(v)
        for k, v in meta[~meta["anomaly"]].groupby("setting").size().items()
    }

    ranges = {
        c: {
            "min": float(df[c].min()),
            "p01": float(df[c].quantile(0.01)),
            "median": float(df[c].median()),
            "p99": float(df[c].quantile(0.99)),
            "max": float(df[c].max()),
        }
        for c in channels
    }

    length = meta["n_points"]
    summary = {
        "file": {
            "path": cfg["data"]["raw_path"],
            "bytes": path.stat().st_size,
            "sha256": sha,
            "sha256_matches_download_record": sha == cfg["data"]["expected_sha256"],
            "parquet_rows": pf.metadata.num_rows,
            "parquet_columns": pf.metadata.num_columns,
            "row_groups": pf.metadata.num_row_groups,
        },
        "read": {
            "columns_read": META_COLUMNS + channels,
            "seconds": round(t_read, 1),
            "hash_seconds": round(t_hash, 1),
            "dataframe_memory_mb": round(mem_mb, 1),
        },
        "schema_selected": {c: schema[c] for c in META_COLUMNS + channels},
        "cycles": {
            "n_cycles": int(len(ids)),
            "sample_id_min": int(ids.min()),
            "sample_id_max": int(ids.max()),
            "sample_ids_contiguous_in_file": bool(df["sample"].is_monotonic_increasing),
            "sample_ids_are_0_to_n_minus_1": bool(np.array_equal(np.sort(ids), np.arange(len(ids)))),
            "train_sample_id_range": [int(meta.index[is_train].min()), int(meta.index[is_train].max())],
            "labels_constant_within_cycle": label_consistent,
            "category_12_iff_normal": normal_cat_consistent,
        },
        "counts_measured": counts,
        "counts_paper": PAPER_COUNTS,
        "counts_match_paper": all(counts[k] == v for k, v in PAPER_COUNTS.items()),
        "test_anomaly_per_category": per_category,
        "normal_cycles_per_setting": normal_settings,
        "cycle_length_points": {
            "all": {k: float(v) for k, v in length.describe().items()},
            "train_normal": {k: float(v) for k, v in length[is_train].describe().items()},
            "test": {k: float(v) for k, v in length[~is_train].describe().items()},
        },
        "time": {
            "n_non_increasing_steps": int((dt <= 0).sum()),
            "dt_min_s": float(dt.min()),
            "dt_median_s": float(dt.median()),
            "dt_max_s": float(dt.max()),
            "t_start_max_s": float(meta["t_start"].max()),
        },
        "quality": {
            "nan_per_channel": n_nan,
            "inf_per_channel": n_inf,
            "duplicate_sample_time_rows": dup_rows,
            "duplicate_cycles_identical_signals": n_dup,
            "cycles_with_constant_channel": const_cycles,
        },
        "channel_value_ranges": ranges,
        "notes": [
            "Units (A, rad/s, N*m) follow the paper signal table and the field names; the Parquet file carries no unit metadata. 待核实 against the paper table.",
            "Cycle duration is analysed separately and is not a v1 feature.",
        ],
    }

    with open(REPORTS_DIR / "audit_summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)

    out = meta[["n_points", "duration_s", "anomaly", "category", "setting"]].copy()
    out["category_name"] = out["category"].map(CATEGORY_NAMES)
    out["is_official_train"] = is_train
    out.to_csv(REPORTS_DIR / "cycle_metadata.csv")

    # --- one normal training cycle -----------------------------------------
    s = int(meta.index[is_train][0])
    cyc = df[df["sample"] == s]
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    groups = [("motor_iq", "q-axis current [A]"), ("joint_velocity", "joint velocity [rad/s]"), ("torque_sensor_a", "torque sensor A [N·m]")]
    for ax, (prefix, label) in zip(axes, groups):
        for i in range(1, 7):
            ax.plot(cyc["time"], cyc[f"{prefix}_{i}"], lw=0.9, label=f"axis {i}")
        ax.set_ylabel(label)
        ax.grid(alpha=0.3)
    axes[0].legend(ncol=6, fontsize=8, loc="upper right")
    axes[-1].set_xlabel("time [s]")
    fig.suptitle(f"Normal training cycle (sample {s}, {len(cyc)} points, setting PRE_A)")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "single_cycle_normal.png", dpi=130)

    print(json.dumps({k: summary[k] for k in ("counts_measured", "counts_match_paper", "cycles", "time")}, indent=2))
    print("quality:", json.dumps(summary["quality"]["duplicate_cycles_identical_signals"]), "dup cycles")


if __name__ == "__main__":
    main()
