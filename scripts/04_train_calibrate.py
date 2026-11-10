"""Weeks 4-5: fit the 3 baselines x 2 channel sets on fit_normal, check on
dev_normal, calibrate on cal_normal, save frozen model bundles.

This script never loads test rows. Outputs:
    models/<model>__<channel_set>.joblib (+ .json metadata)   (not in Git)
    reports/training_summary.json
    reports/figures/normal_score_distributions.png
"""

from __future__ import annotations

import json
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from robot_ad.bundle import dependency_versions, git_commit, now_utc, save_bundle
from robot_ad.config import DEFAULT_CONFIG, FIGURES_DIR, MODELS_DIR, PROCESSED_DIR, REPORTS_DIR, load_config
from robot_ad.data import sha256_bytes
from robot_ad.features import feature_names
from robot_ad.models import IForest, ZScoreMax, calibrate_threshold, make_detector
from robot_ad.schema import ValidationRules

STABILITY_SEEDS = [0, 1, 2, 3, 4]


def main() -> None:
    cfg = load_config()
    man = pd.read_csv(REPORTS_DIR / "split_manifest.csv", index_col="sample")
    normal_roles = ["fit_normal", "dev_normal", "cal_normal"]
    feats_all = pd.read_parquet(PROCESSED_DIR / "features_v1.parquet")
    feats = feats_all.loc[man.index[man["split"].isin(normal_roles)]]  # test rows dropped here
    del feats_all
    roles = man.loc[feats.index, "split"]
    assert not man.loc[feats.index, "anomaly"].any()

    fingerprints = {
        "raw_data_sha256": cfg["data"]["expected_sha256"],
        "split_manifest_sha256": sha256_bytes((REPORTS_DIR / "split_manifest.csv").read_bytes()),
        "config_sha256": sha256_bytes(DEFAULT_CONFIG.read_bytes()),
        "feature_spec_sha256": sha256_bytes((REPORTS_DIR / "feature_spec.json").read_bytes()),
    }
    commit = git_commit()
    deps = dependency_versions()
    MODELS_DIR.mkdir(exist_ok=True)

    results = []
    score_store = {}
    for exp in cfg["experiments"]:
        model_name, ch_set = exp["model"], exp["channel_set"]
        channels = cfg["channels"][ch_set]
        names = feature_names(channels)
        X = {r: feats.loc[roles.index[roles == r], names].to_numpy() for r in normal_roles}

        det = make_detector(model_name, cfg["models"])
        t0 = time.perf_counter()
        det.fit(X["fit_normal"], names)
        t_fit = time.perf_counter() - t0
        t0 = time.perf_counter()
        scores = {r: det.score(X[r]) for r in normal_roles}
        t_score = (time.perf_counter() - t0) / sum(len(v) for v in X.values())

        thr = calibrate_threshold(scores["cal_normal"], cfg["threshold"]["cal_quantile"], cfg["threshold"]["quantile_method"])
        tau = thr["threshold"]
        dev_alarm = float(np.mean(scores["dev_normal"] > tau))
        fit_alarm = float(np.mean(scores["fit_normal"] > tau))

        stability = None
        if model_name == "iforest":
            dev_scores, dev_rates = [], []
            for s in STABILITY_SEEDS:
                d = IForest(**{**cfg["models"]["iforest"], "random_state": s}).fit(X["fit_normal"])
                t_s = calibrate_threshold(d.score(X["cal_normal"]))["threshold"]
                ds = d.score(X["dev_normal"])
                dev_scores.append(ds)
                dev_rates.append(float(np.mean(ds > t_s)))
            rho = [float(spearmanr(scores["dev_normal"], ds).statistic) for ds in dev_scores]
            stability = {
                "seeds": STABILITY_SEEDS,
                "dev_spearman_vs_frozen_seed_min": min(rho),
                "dev_alarm_rate_per_seed": dev_rates,
            }

        deviation = ZScoreMax(**cfg["models"]["zscore_max"]).fit(X["fit_normal"], names)
        model_version = f"robot-ad-{cfg['project_version']}-{model_name}-{ch_set}"
        bundle = {
            "model_version": model_version,
            "schema_version": cfg["schema_version"],
            "model_name": model_name,
            "channel_set": ch_set,
            "is_primary": model_name == cfg["primary"]["model"] and ch_set == cfg["primary"]["channel_set"],
            "channels": channels,
            "feature_names": names,
            "feature_spec": {
                "stats": cfg["features"]["stats"],
                "std_ddof": cfg["features"]["std_ddof"],
                "quantile_method": cfg["features"]["quantile_method"],
            },
            "detector": det,
            "detector_params": det.params(),
            "deviation": deviation,
            "deviation_top_k": cfg["deviation_hint"]["top_k"],
            "threshold": thr,
            "input_validation": ValidationRules.from_config(cfg, channels).to_dict(),
            "n_fit_normal": int(len(X["fit_normal"])),
            "fingerprints": fingerprints,
            "git_commit": commit,
            "dependencies": deps,
            "created_utc": now_utc(),
        }
        save_bundle(bundle, MODELS_DIR / f"{model_name}__{ch_set}.joblib")
        score_store[(model_name, ch_set)] = (scores, tau)

        results.append(
            {
                "model": model_name,
                "channel_set": ch_set,
                "model_version": model_version,
                "n_features": len(names),
                "params": det.params(),
                "threshold": thr,
                "dev_alarm_rate_at_threshold": dev_alarm,
                "fit_alarm_rate_at_threshold": fit_alarm,
                "normal_score_quantiles": {
                    r: {q: float(np.quantile(scores[r], float(q))) for q in ("0.5", "0.95", "0.99")} for r in normal_roles
                },
                "fit_seconds": round(t_fit, 3),
                "score_ms_per_cycle": round(t_score * 1e3, 4),
                "iforest_seed_stability": stability,
            }
        )
        print(f"{model_version:40s} tau={tau:.5g} dev_alarm={dev_alarm:.3f} fit={t_fit:.2f}s")

    summary = {
        "created_utc": now_utc(),
        "git_commit": commit,
        "test_data_used": False,
        "primary": cfg["primary"],
        "split_counts": man[man["split"].isin(normal_roles)]["split"].value_counts().to_dict(),
        "fingerprints": fingerprints,
        "dependencies": deps,
        "experiments": results,
    }
    with open(REPORTS_DIR / "training_summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    for ax, ((m, c), (scores, tau)) in zip(axes.ravel(), score_store.items()):
        lo = min(v.min() for v in scores.values())
        hi = max(v.max() for v in scores.values())
        bins = np.linspace(lo, hi, 40)
        for r, col in zip(normal_roles, ["#4C72B0", "#55A868", "#C44E52"]):
            ax.hist(scores[r], bins=bins, alpha=0.5, label=r, color=col, density=True)
        ax.axvline(tau, color="k", ls="--", lw=1, label="threshold (cal q95)")
        ax.set_title(f"{m} / {c}")
        ax.set_xlabel("anomaly score (larger = more anomalous)")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Normal-only score distributions (no test data)")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "normal_score_distributions.png", dpi=120)


if __name__ == "__main__":
    main()
