"""Week 7-8: export demo cycles from the test set.

- data/examples/<name>.json        one robot-cycle-v1 payload per file (CLI / API input)
- data/examples/invalid/*.json     deliberately broken payloads for input-validation checks
- data/processed/demo_cycles.parquet   signals + known labels for the replay page

Labels are written to a separate index file and are never part of a payload.
These files are derived from CC BY-NC-SA data and are not committed.
"""

from __future__ import annotations

import copy
import json

import numpy as np
import pandas as pd

from robot_ad.config import EXAMPLES_DIR, PROCESSED_DIR, REPORTS_DIR, load_config, raw_path
from robot_ad.schema import cycle_to_payload


def main() -> None:
    cfg = load_config()
    channels = cfg["channels"]["full_18"]
    man = pd.read_csv(REPORTS_DIR / "split_manifest.csv", index_col="sample")
    err = json.loads((REPORTS_DIR / "error_analysis.json").read_text(encoding="utf-8"))
    test = man[man["split"] == "test"]

    rng = np.random.default_rng(7)
    picks = {c["sample"]: f"case_{c['kind']}" for c in err["cases"]}
    for s in rng.choice(test.index[~test["anomaly"]], 4, replace=False):
        picks.setdefault(int(s), "test_normal")
    for cat, g in test[test["anomaly"]].groupby("category_name"):
        picks.setdefault(int(rng.choice(g.index)), f"test_{cat.lower()}")

    df = pd.read_parquet(raw_path(cfg), columns=["sample", "time"] + channels, filters=[("sample", "in", list(picks))])
    EXAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    (EXAMPLES_DIR / "invalid").mkdir(exist_ok=True)

    index = []
    for sid, g in df.groupby("sample"):
        name = f"{picks[sid]}_{sid:04d}"
        payload = cycle_to_payload(g["time"].to_numpy(), g[channels].to_numpy(), channels, cfg["schema_version"])
        (EXAMPLES_DIR / f"{name}.json").write_text(json.dumps(payload), encoding="utf-8")
        row = man.loc[sid]
        index.append({"file": f"{name}.json", "sample": int(sid), "anomaly": bool(row["anomaly"]),
                      "category": row["category_name"], "setting": int(row["setting"])})
    (EXAMPLES_DIR / "index.json").write_text(json.dumps(index, indent=2), encoding="utf-8")

    # invalid payloads built from the first normal example
    base = json.loads((EXAMPLES_DIR / index[0]["file"]).read_text(encoding="utf-8"))
    bad = {}
    p = copy.deepcopy(base); del p["signals"]["motor_iq_3"]; bad["missing_channel"] = p
    p = copy.deepcopy(base); p["signals"]["torque_sensor_a_2"] = p["signals"]["torque_sensor_a_2"][:-5]; bad["length_mismatch"] = p
    p = copy.deepcopy(base); p["signals"]["joint_velocity_1"][100] = float("nan"); bad["non_finite"] = p
    p = copy.deepcopy(base); p["time_s"][200], p["time_s"][201] = p["time_s"][201], p["time_s"][200]; bad["time_not_increasing"] = p
    p = copy.deepcopy(base); p["schema_version"] = "robot-cycle-v0"; bad["unknown_schema_version"] = p
    p = copy.deepcopy(base); p["signals"]["robot_voltage"] = p["signals"]["motor_iq_1"]; bad["unexpected_channel"] = p
    p = copy.deepcopy(base); p["time_s"] = p["time_s"][:500]; p["signals"] = {k: v[:500] for k, v in p["signals"].items()}; bad["length_out_of_range"] = p
    for k, v in bad.items():
        (EXAMPLES_DIR / "invalid" / f"{k}.json").write_text(json.dumps(v), encoding="utf-8")  # NaN written as NaN token

    demo = df.merge(man[["anomaly", "category_name", "setting"]], left_on="sample", right_index=True)
    demo.to_parquet(PROCESSED_DIR / "demo_cycles.parquet", index=False)
    print(f"exported {len(index)} valid examples, {len(bad)} invalid payloads, demo rows {len(demo)}")


if __name__ == "__main__":
    main()
