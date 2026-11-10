"""Week 2: freeze the fit/dev/cal/test sample manifest.

Rule (pre-registered in docs/validation_protocol.md):
- test = official test part (setting != PRE_A), normal + anomalous, untouched until evaluation.
- The 948 official PRE_A normal cycles are cut into blocks of `block_size`
  consecutive sample ids. Whole blocks are shuffled with a fixed seed and
  assigned 60/20/20 to fit_normal / dev_normal / cal_normal.

Why blocks: sample ids are contiguous in the file and probably follow the
recording order, but the recording order is not documented. Blocks keep
neighbouring cycles together (limits near-duplicate leakage between roles)
while the shuffle avoids putting all late cycles into one role.

Outputs: reports/split_manifest.csv, reports/split_summary.json
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from robot_ad.config import REPORTS_DIR, load_config
from robot_ad.data import sha256_bytes


def main() -> None:
    cfg = load_config()
    sp = cfg["split"]
    meta = pd.read_csv(REPORTS_DIR / "cycle_metadata.csv", index_col="sample")

    train_ids = np.sort(meta.index[(meta["setting"] == cfg["data"]["train_setting_code"]) & (~meta["anomaly"])].to_numpy())
    test_ids = np.sort(meta.index[meta["setting"] != cfg["data"]["train_setting_code"]].to_numpy())
    assert not meta.loc[meta["setting"] == cfg["data"]["train_setting_code"], "anomaly"].any()

    bs = int(sp["block_size"])
    blocks = [train_ids[i : i + bs] for i in range(0, len(train_ids), bs)]
    rng = np.random.default_rng(int(sp["seed"]))
    order = rng.permutation(len(blocks))
    n_fit = round(len(blocks) * sp["fractions"]["fit_normal"])
    n_dev = round(len(blocks) * sp["fractions"]["dev_normal"])

    rows = []
    for rank, b in enumerate(order):
        role = "fit_normal" if rank < n_fit else "dev_normal" if rank < n_fit + n_dev else "cal_normal"
        for s in blocks[b]:
            rows.append((int(s), role, int(b)))
    for s in test_ids:
        rows.append((int(s), "test", -1))

    man = pd.DataFrame(rows, columns=["sample", "split", "block"]).set_index("sample").sort_index()
    man = man.join(meta[["anomaly", "category", "category_name", "setting", "n_points"]])

    # integrity checks
    assert man.index.is_unique
    assert len(man) == len(meta)
    assert not man.loc[man["split"] != "test", "anomaly"].any()

    man.to_csv(REPORTS_DIR / "split_manifest.csv", lineterminator="\n")
    digest = sha256_bytes((REPORTS_DIR / "split_manifest.csv").read_bytes())

    counts = man.groupby("split").agg(n=("anomaly", "size"), n_anomaly=("anomaly", "sum"))
    summary = {
        "rule": "blocks of consecutive PRE_A normal sample ids, shuffled with fixed seed, 60/20/20",
        "block_size": bs,
        "seed": int(sp["seed"]),
        "n_blocks": len(blocks),
        "counts": {k: {"n": int(v["n"]), "n_anomaly": int(v["n_anomaly"])} for k, v in counts.iterrows()},
        "sample_overlap_between_splits": 0,
        "manifest_sha256": digest,
    }
    with open(REPORTS_DIR / "split_summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
