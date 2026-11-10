"""v1.1: the single holdout evaluation (exploratory; see docs/validation_protocol_v1_1.md).

Refuses to run unless the selected configuration is committed and tracked files are clean.
Compares on the same holdout: v1.1 selected, v1 Isolation Forest (primary), v1 z-score, v1 PCA.

Outputs: reports/v1_1/holdout_evaluation.json, reports/v1_1/holdout_scores.csv,
         reports/figures/v1_1_*.png
"""

from __future__ import annotations

import json
import subprocess

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

from robot_ad.bundle import git_commit, load_bundle, now_utc
from robot_ad.config import (
    FIGURES_DIR,
    MODELS_DIR,
    NORMAL_SETTING_NAMES,
    PROCESSED_DIR,
    PROJECT_ROOT,
    REPORTS_DIR,
    load_config,
    raw_path,
)
from robot_ad.metrics import operating_point, per_group_recall, ranking

OUT = REPORTS_DIR / "v1_1"
N_BOOT = 2000


def guard() -> None:
    tracked_dirty = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"], cwd=PROJECT_ROOT, capture_output=True, text=True
    ).stdout.strip()
    in_head = (
        subprocess.run(
            ["git", "cat-file", "-e", "HEAD:configs/v1_1_selected.json"], cwd=PROJECT_ROOT, capture_output=True
        ).returncode
        == 0
    )
    if tracked_dirty or not in_head:
        raise SystemExit("Refusing: commit configs/v1_1_selected.json and a clean tree before scoring the holdout.")
    if (OUT / "holdout_evaluation.json").exists():
        raise SystemExit("Refusing: the holdout has already been scored once (reports/v1_1/holdout_evaluation.json).")


def bootstrap(y, scores: dict, taus: dict, seed: int = 0) -> dict:
    """Percentile 95 % CIs for recall/FPR/AUC/AP and paired differences vs. the v1.1 model."""
    rng = np.random.default_rng(seed)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    stats = {k: {"recall": [], "fpr": [], "auc": [], "ap": []} for k in scores}
    for _ in range(N_BOOT):
        idx = np.r_[rng.choice(pos, len(pos)), rng.choice(neg, len(neg))]  # stratified resample
        yb = y[idx]
        for k, s in scores.items():
            sb = s[idx]
            pred = sb > taus[k]
            stats[k]["recall"].append(pred[yb == 1].mean())
            stats[k]["fpr"].append(pred[yb == 0].mean())
            stats[k]["auc"].append(roc_auc_score(yb, sb))
            stats[k]["ap"].append(average_precision_score(yb, sb))
    ci = lambda a: [float(np.quantile(a, 0.025)), float(np.quantile(a, 0.975))]  # noqa: E731
    out = {k: {m: ci(v) for m, v in d.items()} for k, d in stats.items()}
    ref = "v1_1"
    out["paired_difference_v1_1_minus"] = {
        k: {m: ci(np.array(stats[ref][m]) - np.array(stats[k][m])) for m in ("recall", "fpr", "auc", "ap")}
        for k in scores
        if k != ref
    }
    return out


def main() -> None:
    guard()
    cfg = load_config()
    man = pd.read_csv(OUT / "split_manifest_v1_1.csv", index_col="sample")
    hold = man[man["role_v1_1"] == "holdout"]
    y = hold["anomaly"].to_numpy().astype(int)
    feats_all = pd.read_parquet(PROCESSED_DIR / "features_v1.parquet").loc[hold.index]

    b11 = load_bundle(MODELS_DIR / "v1_1__selected.joblib")
    frozen = json.loads((PROJECT_ROOT / "configs" / "v1_1_selected.json").read_text(encoding="utf-8"))
    assert b11["model_version"] == frozen["model_version"]
    F = feats_all[b11["feature_names"]].to_numpy()
    S = None
    if b11["scorer"].needs_sequence:
        from robot_ad.sequence import resample_cycle

        ch = b11["channels"]
        df = pd.read_parquet(
            raw_path(cfg), columns=["sample", "time"] + ch, filters=[("sample", "in", hold.index.tolist())]
        )
        S = np.stack(
            [resample_cycle(g["time"].to_numpy(), g[ch].to_numpy()) for _, g in df.groupby("sample", sort=True)]
        )
    scores = {"v1_1": b11["scorer"].score(F, S)}
    taus = {"v1_1": b11["threshold_levels"]["warning"]}
    versions = {"v1_1": b11["model_version"]}
    for key, fname in [
        ("v1_iforest", "iforest__full_18"),
        ("v1_zscore", "zscore_max__full_18"),
        ("v1_pca", "pca_recon__full_18"),
    ]:
        b = load_bundle(MODELS_DIR / f"{fname}.joblib")
        scores[key] = b["detector"].score(feats_all[b["feature_names"]].to_numpy())
        taus[key] = b["threshold"]["threshold"]
        versions[key] = b["model_version"]

    cat = hold["category_name"].to_numpy()
    setting = np.array([NORMAL_SETTING_NAMES.get(int(s), "anomaly") for s in hold["setting"]])
    results = {}
    for k, s in scores.items():
        r = {
            "model_version": versions[k],
            "operating_point_warning": operating_point(y, s, taus[k]),
            "ranking": ranking(y, s),
            "per_category": per_group_recall(cat[y == 1], y[y == 1], s[y == 1], taus[k]),
            "per_normal_setting": per_group_recall(setting[y == 0], y[y == 0], s[y == 0], taus[k]),
        }
        if k == "v1_1":
            a = b11["threshold_levels"]["alarm"]
            r["operating_point_alarm"] = operating_point(y, s, a)
            r["per_normal_setting_alarm"] = per_group_recall(setting[y == 0], y[y == 0], s[y == 0], a)
        results[k] = r

    out = {
        "created_utc": now_utc(),
        "evaluation_git_commit": git_commit(),
        "label": "EXPLORATORY: holdout is half of the v1 test part, whose aggregate metrics were seen in v1",
        "holdout_counts": {
            "n": int(len(y)),
            "n_anomaly": int(y.sum()),
            "n_normal": int((1 - y).sum()),
            "anomaly_rate": float(y.mean()),
        },
        "thresholds": {"v1_1_levels": b11["threshold_levels"], **{k: taus[k] for k in taus if k != "v1_1"}},
        "results": results,
        "bootstrap_95ci": bootstrap(y, scores, taus),
        "n_bootstrap": N_BOOT,
    }
    (OUT / "holdout_evaluation.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    sc = pd.DataFrame({k: v for k, v in scores.items()}, index=hold.index)
    sc.insert(0, "anomaly", hold["anomaly"])
    sc.insert(1, "category_name", hold["category_name"])
    sc.insert(2, "setting", hold["setting"])
    sc.to_csv(OUT / "holdout_scores.csv", lineterminator="\n")

    print(f"{'model':12s} {'recall':>7s} {'fpr':>6s} {'prec':>6s} {'auc':>6s} {'ap':>6s}")
    for k, r in results.items():
        op, rk = r["operating_point_warning"], r["ranking"]
        print(
            f"{k:12s} {op['recall']:7.3f} {op['fpr']:6.3f} {op['precision']:6.3f} {rk['roc_auc']:6.3f} {rk['average_precision']:6.3f}"
        )
    op = results["v1_1"]["operating_point_alarm"]
    print(f"v1_1 alarm    {op['recall']:7.3f} {op['fpr']:6.3f} {op['precision']:6.3f}")
    print(json.dumps(out["bootstrap_95ci"]["paired_difference_v1_1_minus"], indent=1))

    # ---- figures ---------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for k, s in scores.items():
        fpr, tpr, _ = roc_curve(y, s)
        axes[0].plot(fpr, tpr, label=f"{k} AUC={results[k]['ranking']['roc_auc']:.3f}")
        op = results[k]["operating_point_warning"]
        axes[0].plot(op["fpr"], op["recall"], "o", color=axes[0].lines[-1].get_color())
    axes[0].plot([0, 1], [0, 1], "k:", lw=0.8)
    axes[0].set(xlabel="FPR", ylabel="recall", title="Holdout ROC (dots = operating point)")
    axes[0].legend(fontsize=8)
    sets = [row["group"] for row in results["v1_1"]["per_normal_setting"]]
    w = 0.2
    for i, k in enumerate(scores):
        axes[1].bar(
            np.arange(len(sets)) + (i - 1.5) * w, [row["fpr"] for row in results[k]["per_normal_setting"]], w, label=k
        )
    axes[1].axhline(0.05, color="k", ls="--", lw=0.8, label="5 % target")
    axes[1].set_xticks(
        np.arange(len(sets)), [f"{s}\n(n={row['n']})" for s, row in zip(sets, results["v1_1"]["per_normal_setting"])]
    )
    axes[1].set(ylabel="false-alarm rate", title="Holdout normal cycles by setting")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "v1_1_holdout_roc_fpr.png", dpi=130)
    plt.close(fig)

    cats = [row["group"] for row in results["v1_1"]["per_category"]]
    fig, ax = plt.subplots(figsize=(12, 4.5))
    for i, k in enumerate(["v1_1", "v1_iforest", "v1_zscore"]):
        ax.bar(
            np.arange(len(cats)) + (i - 1) * 0.27, [row["recall"] for row in results[k]["per_category"]], 0.27, label=k
        )
    ax.set_xticks(
        np.arange(len(cats)),
        [f"{c}\n(n={row['n']})" for c, row in zip(cats, results["v1_1"]["per_category"])],
        rotation=45,
        ha="right",
        fontsize=8,
    )
    ax.set(ylabel="recall at warning threshold", ylim=(0, 1.05), title="Holdout recall per anomaly category")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "v1_1_holdout_category_recall.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
