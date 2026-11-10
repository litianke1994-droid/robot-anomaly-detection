"""Week 6: error analysis of the primary model after the frozen test run.

- 8 cases: 3 highest-scoring normal cycles (false alarms), 3 lowest-scoring
  anomalies (misses), 2 borderline cycles closest to the threshold.
- Drift check: how far test-normal cycles of each setting sit from fit_normal
  in each channel group (median |z| of the features) and in cycle duration.

Outputs: reports/error_analysis.json, reports/figures/case_*.png, reports/figures/normal_drift.png
"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from robot_ad.bundle import load_bundle
from robot_ad.config import (
    FIGURES_DIR,
    MODELS_DIR,
    NORMAL_SETTING_NAMES,
    PROCESSED_DIR,
    REPORTS_DIR,
    load_config,
    raw_path,
)
from robot_ad.predict import Predictor

GROUPS = {"motor_iq": "current", "joint_velocity": "velocity", "torque_sensor_a": "torque"}


def read_cycles(path, samples, columns):
    df = pd.read_parquet(
        path, columns=["sample", "time"] + columns, filters=[("sample", "in", list(map(int, samples)))]
    )
    return {int(s): g.reset_index(drop=True) for s, g in df.groupby("sample")}


def main() -> None:
    cfg = load_config()
    p = cfg["primary"]
    bundle = load_bundle(MODELS_DIR / f"{p['model']}__{p['channel_set']}.joblib")
    pred = Predictor(bundle)
    pv, tau = bundle["model_version"], pred.threshold
    channels = bundle["channels"]

    scores = pd.read_csv(REPORTS_DIR / "test_scores.csv", index_col="sample")
    man = pd.read_csv(REPORTS_DIR / "split_manifest.csv", index_col="sample")
    feats = pd.read_parquet(PROCESSED_DIR / "features_v1.parquet")
    s = scores[pv]
    normal = scores[~scores["anomaly"]]
    anom = scores[scores["anomaly"]]

    picks = []
    for sid in normal[pv].sort_values(ascending=False).index[:3]:
        picks.append((int(sid), "false_alarm"))
    for sid in anom[pv].sort_values().index[:3]:
        picks.append((int(sid), "missed_anomaly"))
    gap = (s - tau).abs().sort_values()
    for sid in gap.index:
        if len([k for k in picks if k[1] == "borderline"]) == 2:
            break
        if int(sid) not in [k[0] for k in picks]:
            picks.append((int(sid), "borderline"))

    # normal reference: 30 fit_normal cycles
    fit_ids = man.index[man["split"] == "fit_normal"].to_numpy()
    ref_ids = np.random.default_rng(0).choice(fit_ids, size=30, replace=False)
    cycles = read_cycles(raw_path(cfg), [k[0] for k in picks] + list(ref_ids), channels)

    cases = []
    for sid, kind in picks:
        x = feats.loc[sid, bundle["feature_names"]].to_numpy()
        hints = pred.deviation_hints(x)
        row = scores.loc[sid]
        case = {
            "sample": sid,
            "kind": kind,
            "true_label": "anomaly" if row["anomaly"] else "normal",
            "category": row["category_name"],
            "setting": NORMAL_SETTING_NAMES.get(int(row["setting"]), "anomaly setting"),
            "n_points": int(man.loc[sid, "n_points"]),
            "score": float(row[pv]),
            "threshold": tau,
            "predicted": "anomaly" if row[pv] > tau else "normal",
            "other_model_scores_vs_threshold": {},
            "top_deviating_channels": hints,
        }
        for other in ("zscore_max", "pca_recon"):
            ob = json.loads((MODELS_DIR / f"{other}__full_18.json").read_text(encoding="utf-8"))
            case["other_model_scores_vs_threshold"][other] = {
                "score": float(row[ob["model_version"]]),
                "threshold": ob["threshold"]["threshold"],
                "alarm": bool(row[ob["model_version"]] > ob["threshold"]["threshold"]),
            }
        cases.append(case)

        # figure: top-3 deviating channels vs normal reference
        fig, axes = plt.subplots(3, 1, figsize=(9, 7), sharex=True)
        cyc = cycles[sid]
        for ax, h in zip(axes, hints):
            ch = h["channel"]
            for r in ref_ids:
                ax.plot(cycles[int(r)]["time"], cycles[int(r)][ch], color="0.75", lw=0.6)
            ax.plot(cyc["time"], cyc[ch], color="C3", lw=1.2, label=f"sample {sid}")
            ax.set_ylabel(ch, fontsize=8)
            ax.set_title(f"{ch}: max |z| = {h['max_abs_z']} ({h['feature'].split('__')[1]})", fontsize=9)
            ax.grid(alpha=0.3)
        axes[0].legend(fontsize=8, loc="upper right")
        axes[-1].set_xlabel("time [s] (grey: 30 fit_normal cycles)")
        fig.suptitle(
            f"{kind}: sample {sid}, true={case['true_label']} ({case['category']}, {case['setting']}), "
            f"score={case['score']:.3f}, tau={tau:.3f}",
            fontsize=10,
        )
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / f"case_{kind}_{sid}.png", dpi=110)
        plt.close(fig)

    # ---- drift of test-normal settings relative to fit_normal ---------------
    dev = bundle["deviation"]
    normal_ids = man.index[(~man["anomaly"])]
    Z = np.abs(dev.zscores(feats.loc[normal_ids, bundle["feature_names"]].to_numpy()))
    role = (
        man.loc[normal_ids]
        .apply(lambda r: r["split"] if r["split"] != "test" else NORMAL_SETTING_NAMES[int(r["setting"])], axis=1)
        .to_numpy()
    )
    names = bundle["feature_names"]
    drift = []
    for g in ["fit_normal", "dev_normal", "cal_normal", "PRE_B", "BETWEEN_A", "BETWEEN_B", "BETWEEN_C"]:
        m = role == g
        row = {"group": g, "n": int(m.sum())}
        for prefix, label in GROUPS.items():
            cols = [i for i, n in enumerate(names) if n.startswith(prefix)]
            row[f"median_abs_z_{label}"] = float(np.median(Z[np.ix_(m, cols)]))
        row["median_n_points"] = float(np.median(man.loc[normal_ids[m], "n_points"]))
        if g in NORMAL_SETTING_NAMES.values():
            row["primary_fpr"] = float(np.mean(scores.loc[normal_ids[m], pv] > tau))
        drift.append(row)

    # which features moved most for BETWEEN_B
    mb = role == "BETWEEN_B"
    zmean = np.mean(dev.zscores(feats.loc[normal_ids[mb], names].to_numpy()), axis=0)
    top = np.argsort(-np.abs(zmean))[:10]
    between_b_top = [{"feature": names[i], "mean_z": round(float(zmean[i]), 2)} for i in top]

    out = {
        "model_version": pv,
        "threshold": tau,
        "cases": cases,
        "normal_drift_vs_fit_normal": drift,
        "between_b_most_shifted_features": between_b_top,
        "note": "Analysis after the frozen test run. Nothing here changes the v1 models or thresholds.",
    }
    with open(REPORTS_DIR / "error_analysis.json", "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)

    fig, ax = plt.subplots(figsize=(9, 4))
    labels = [d["group"] for d in drift]
    w = 0.27
    for k, lab in enumerate(GROUPS.values()):
        ax.bar(np.arange(len(labels)) + (k - 1) * w, [d[f"median_abs_z_{lab}"] for d in drift], w, label=lab)
    ax.set_xticks(np.arange(len(labels)), [f"{d['group']}\n(n={d['n']})" for d in drift], fontsize=8)
    ax.set_ylabel("median |z| vs fit_normal")
    ax.set_title("Normal cycles: distance from the training distribution, per feature group")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "normal_drift.png", dpi=130)
    plt.close(fig)

    for c in cases:
        print(
            c["kind"],
            c["sample"],
            c["true_label"],
            c["category"],
            c["setting"],
            round(c["score"], 3),
            [(h["channel"], h["max_abs_z"]) for h in c["top_deviating_channels"]],
            {k: v["alarm"] for k, v in c["other_model_scores_vs_threshold"].items()},
        )
    for d in drift:
        print(d)
    print(between_b_top)


if __name__ == "__main__":
    main()
