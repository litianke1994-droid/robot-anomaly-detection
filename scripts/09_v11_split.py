"""v1.1: split the official test part into validation and holdout (see docs/validation_protocol_v1_1.md).

Output: reports/v1_1/split_manifest_v1_1.csv, reports/v1_1/split_summary_v1_1.json
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from robot_ad.config import PROJECT_ROOT, REPORTS_DIR
from robot_ad.data import sha256_bytes

OUT = REPORTS_DIR / "v1_1"


def block_halves(ids: np.ndarray, block: int, rng: np.random.Generator, frac: float) -> tuple[list, list]:
    """Cut sorted ids into consecutive blocks, shuffle blocks, give round(frac * n_blocks) blocks to the first part."""
    ids = np.sort(ids)
    blocks = [ids[i : i + block] for i in range(0, len(ids), block)]
    order = rng.permutation(len(blocks))
    k = int(round(len(blocks) * frac))
    first = [int(s) for b in order[:k] for s in blocks[b]]
    second = [int(s) for b in order[k:] for s in blocks[b]]
    return first, second


def main() -> None:
    cfg = json.loads((PROJECT_ROOT / "configs" / "v1_1.json").read_text(encoding="utf-8"))
    sp = cfg["split"]
    man = pd.read_csv(REPORTS_DIR / "split_manifest.csv", index_col="sample")
    rng = np.random.default_rng(sp["seed"])
    role = man["split"].copy()  # fit_normal / dev_normal / cal_normal keep their v1 roles

    test = man[man["split"] == "test"]
    for _setting, g in test[~test["anomaly"]].groupby("setting"):
        val, hold = block_halves(g.index.to_numpy(), sp["block_size"], rng, sp["validation_fraction"])
        vfit, vcal = block_halves(np.array(val), sp["block_size"], rng, sp["val_normal_fit_fraction"])
        role.loc[vfit] = "val_normal_fit"
        role.loc[vcal] = "val_normal_cal"
        role.loc[hold] = "holdout"
    for _cat, g in test[test["anomaly"]].groupby("category_name"):
        val, hold = block_halves(g.index.to_numpy(), sp["block_size"], rng, sp["validation_fraction"])
        role.loc[val] = "val_anomaly"
        role.loc[hold] = "holdout"

    out = man.copy()
    out["role_v1_1"] = role
    assert set(out.loc[out["split"] == "test", "role_v1_1"]) == {
        "val_normal_fit",
        "val_normal_cal",
        "val_anomaly",
        "holdout",
    }
    OUT.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT / "split_manifest_v1_1.csv", lineterminator="\n")

    t = out[out["split"] == "test"]
    counts = {
        r: {
            "n": int((t["role_v1_1"] == r).sum()),
            "n_anomaly": int(t.loc[t["role_v1_1"] == r, "anomaly"].sum()),
            "normal_by_setting": {
                str(k): int(v) for k, v in t[(t["role_v1_1"] == r) & ~t["anomaly"]].groupby("setting").size().items()
            },
        }
        for r in ["val_normal_fit", "val_normal_cal", "val_anomaly", "holdout"]
    }
    summary = {
        "counts": counts,
        "holdout_anomaly_per_category": {
            k: int(v)
            for k, v in t[(t["role_v1_1"] == "holdout") & t["anomaly"]].groupby("category_name").size().items()
        },
        "manifest_sha256": sha256_bytes((OUT / "split_manifest_v1_1.csv").read_bytes()),
    }
    (OUT / "split_summary_v1_1.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
