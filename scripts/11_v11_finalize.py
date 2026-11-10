"""v1.1: pick the best validation candidate and build the final bundle (no holdout data).

Selection = top of reports/v1_1/search_log.jsonl by the registered criterion
(mean TPR at 5 % FPR, tie-break mean AUC), ignoring superseded rows.
Final fit uses the canonical v1.1 split from the manifest:
    fit  = fit_normal (+ val_normal_fit if reference is "multi")
    cal  = cal_normal + val_normal_cal   -> warning = q95, alarm = q99

Outputs:
    models/v1_1__selected.joblib (+ .json)     (not in Git)
    configs/v1_1_selected.json                 frozen choice, committed before the holdout run
    reports/v1_1/final_model.json
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from robot_ad.bundle import dependency_versions, git_commit, now_utc, save_bundle
from robot_ad.config import MODELS_DIR, PROCESSED_DIR, PROJECT_ROOT, REPORTS_DIR, load_config
from robot_ad.data import sha256_bytes
from robot_ad.features import feature_names
from robot_ad.models import ZScoreMax, make_detector
from robot_ad.schema import ValidationRules
from robot_ad.scorer import CycleScorer

OUT = REPORTS_DIR / "v1_1"


def build(spec: dict):
    if spec["name"] == "seq_ae":
        from robot_ad.sequence import SeqAutoencoder

        return SeqAutoencoder(**spec.get("kw", {}))
    return make_detector(spec["name"], load_config()["models"], **spec.get("kw", {}))


def label_of(spec: dict) -> str:
    kw = ",".join(f"{k}={v}" for k, v in spec.get("kw", {}).items())
    return f"{spec['name']}({kw})" if kw else spec["name"]


def main() -> None:
    cfg = load_config()
    cfg11 = json.loads((PROJECT_ROOT / "configs" / "v1_1.json").read_text(encoding="utf-8"))
    rows = [json.loads(x) for x in (OUT / "search_log.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    valid = [r for r in rows if not r.get("superseded")]
    valid.sort(key=lambda d: (d["metrics"]["tpr_at_5fpr_mean"], d["metrics"]["auc_mean"]), reverse=True)
    best = valid[0]
    print("selected:", best["label"], best["metrics"])

    man = pd.read_csv(OUT / "split_manifest_v1_1.csv", index_col="sample")
    man = man[man["role_v1_1"] != "holdout"]
    roles = man["role_v1_1"]
    fit_roles = ["fit_normal"] + (["val_normal_fit"] if best["ref"] == "multi" else [])
    fit_ids = man.index[roles.isin(fit_roles)]
    cal_ids = man.index[roles.isin(["cal_normal", "val_normal_cal"])]

    channels = cfg["channels"]["full_18"]
    names = feature_names(channels)
    feats = pd.read_parquet(PROCESSED_DIR / "features_v1.parquet").loc[man.index, names]
    members = [(label_of(m), build(m)) for m in best["members"]]
    scorer = CycleScorer(members)
    S_all = None
    if scorer.needs_sequence:
        z = np.load(PROCESSED_DIR / "seq_1024_nonholdout.npz")
        assert np.array_equal(z["samples"], man.index.to_numpy())
        S_all = z["S"]

    pos_fit = man.index.get_indexer(fit_ids)
    pos_cal = man.index.get_indexer(cal_ids)
    F = feats.to_numpy()
    scorer.fit(F[pos_fit], None if S_all is None else S_all[pos_fit], names)
    cal_scores = scorer.score(F[pos_cal], None if S_all is None else S_all[pos_cal])

    lv = cfg11["calibration"]["levels"]
    method = cfg11["calibration"]["quantile_method"]
    levels = {k: float(np.quantile(cal_scores, q, method=method)) for k, q in lv.items()}
    primary = levels[cfg11["calibration"]["primary_level"]]
    cal_settings = man.loc[cal_ids, "setting"].to_numpy()
    per_setting = {
        str(s): {
            "n": int((cal_settings == s).sum()),
            "warning_rate": float(np.mean(cal_scores[cal_settings == s] > primary)),
        }
        for s in np.unique(cal_settings)
    }

    deviation = ZScoreMax().fit(F[pos_fit], names)
    slug = "+".join(
        label_of(m).replace("(", "-").replace(")", "").replace("=", "").replace(",", "-") for m in best["members"]
    )
    model_version = f"robot-ad-1.1.0-{slug}-{best['ref']}"
    manifest_sha = sha256_bytes((OUT / "split_manifest_v1_1.csv").read_bytes())
    bundle = {
        "model_version": model_version,
        "schema_version": cfg["schema_version"],
        "model_name": best["label"],
        "channel_set": "full_18",
        "is_primary": True,
        "channels": channels,
        "feature_names": names,
        "feature_spec": {k: cfg["features"][k] for k in ("stats", "std_ddof", "quantile_method")},
        "scorer": scorer,
        "detector": scorer,
        "detector_params": scorer.params(),
        "deviation": deviation,
        "deviation_top_k": cfg["deviation_hint"]["top_k"],
        "threshold": {
            "threshold": primary,
            "quantile": lv[cfg11["calibration"]["primary_level"]],
            "quantile_method": method,
            "rule": "score > threshold",
            "n_cal": int(len(cal_scores)),
            "cal_alarm_rate": float(np.mean(cal_scores > primary)),
        },
        "threshold_levels": levels,
        "reference": best["ref"],
        "n_fit": int(len(fit_ids)),
        "input_validation": ValidationRules.from_config(cfg, channels).to_dict(),
        "fingerprints": {
            "split_manifest_v1_1_sha256": manifest_sha,
            "raw_data_sha256": cfg["data"]["expected_sha256"],
        },
        "validation_metrics": best["metrics"],
        "git_commit": git_commit(),
        "dependencies": dependency_versions(),
        "created_utc": now_utc(),
    }
    save_bundle(bundle, MODELS_DIR / "v1_1__selected.joblib")

    frozen = {
        "selected_label": best["label"],
        "reference": best["ref"],
        "members": best["members"],
        "model_version": model_version,
        "threshold_levels": levels,
        "validation_metrics": best["metrics"],
        "split_manifest_v1_1_sha256": manifest_sha,
        "selected_from_rounds": sorted({r["round"] for r in valid}),
        "n_candidates_evaluated": len(rows),
        "n_candidates_valid": len(valid),
    }
    (PROJECT_ROOT / "configs" / "v1_1_selected.json").write_text(json.dumps(frozen, indent=2), encoding="utf-8")
    final = {
        **frozen,
        "n_fit": int(len(fit_ids)),
        "n_cal": int(len(cal_ids)),
        "calibration_warning_rate_by_setting": per_setting,
        "member_params": scorer.params(),
        "top10_validation": [{"round": r["round"], "label": r["label"], **r["metrics"]} for r in valid[:10]],
    }
    (OUT / "final_model.json").write_text(json.dumps(final, indent=2), encoding="utf-8")
    print(json.dumps({"model_version": model_version, "levels": levels, "per_setting": per_setting}, indent=2))


if __name__ == "__main__":
    main()
