"""Week 6: one formal evaluation of the frozen bundles on the official test part.

Outputs:
    reports/evaluation.json
    reports/test_scores.csv
    reports/figures/{confusion_primary,category_recall,test_scores_primary,roc_pr}.png
"""

from __future__ import annotations

import json
import os
import platform
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve

from robot_ad.bundle import dependency_versions, git_commit, load_bundle, now_utc
from robot_ad.config import (
    DEFAULT_CONFIG,
    FIGURES_DIR,
    MODELS_DIR,
    NORMAL_SETTING_NAMES,
    PROCESSED_DIR,
    REPORTS_DIR,
    load_config,
)
from robot_ad.data import sha256_bytes
from robot_ad.metrics import operating_point, per_group_recall, ranking


def main() -> None:
    cfg = load_config()
    man = pd.read_csv(REPORTS_DIR / "split_manifest.csv", index_col="sample")
    test = man[man["split"] == "test"]
    feats = pd.read_parquet(PROCESSED_DIR / "features_v1.parquet").loc[test.index]
    y = test["anomaly"].to_numpy().astype(int)
    cat = test["category_name"].to_numpy()
    setting = np.array([NORMAL_SETTING_NAMES.get(int(s), "anomaly_setting") for s in test["setting"]])
    feature_ms = json.loads((REPORTS_DIR / "feature_spec.json").read_text(encoding="utf-8"))["timing"][
        "feature_ms_per_cycle"
    ]

    manifest_sha = sha256_bytes((REPORTS_DIR / "split_manifest.csv").read_bytes())
    config_sha = sha256_bytes(DEFAULT_CONFIG.read_bytes())

    results, scores_out = [], pd.DataFrame(index=test.index)
    scores_out["anomaly"] = test["anomaly"]
    scores_out["category_name"] = test["category_name"]
    scores_out["setting"] = test["setting"]
    for exp in cfg["experiments"]:
        b = load_bundle(MODELS_DIR / f"{exp['model']}__{exp['channel_set']}.joblib")
        assert b["fingerprints"]["split_manifest_sha256"] == manifest_sha, "manifest changed after freezing"
        assert b["fingerprints"]["config_sha256"] == config_sha, "config changed after freezing"
        X = feats[b["feature_names"]].to_numpy()
        t0 = time.perf_counter()
        s = b["detector"].score(X)
        score_ms = (time.perf_counter() - t0) * 1e3 / len(X)
        tau = b["threshold"]["threshold"]
        scores_out[b["model_version"]] = s

        normal_rows = per_group_recall(setting[y == 0], y[y == 0], s[y == 0], tau)
        results.append(
            {
                "model_version": b["model_version"],
                "model": b["model_name"],
                "channel_set": b["channel_set"],
                "is_primary": b["is_primary"],
                "bundle_git_commit": b["git_commit"],
                "operating_point": operating_point(y, s, tau),
                "ranking": ranking(y, s),
                "per_category": per_group_recall(cat[y == 1], y[y == 1], s[y == 1], tau),
                "per_normal_setting": normal_rows,
                "n_ties_at_threshold": int(np.sum(s == tau)),
                "timing_ms_per_cycle": {"feature_extraction": feature_ms, "scoring_batch": round(score_ms, 4)},
            }
        )

    out = {
        "created_utc": now_utc(),
        "evaluation_git_commit": git_commit(),
        "protocol": "docs/validation_protocol.md",
        "positive_class": "anomaly",
        "test_counts": {
            "n": int(len(y)),
            "n_anomaly": int(y.sum()),
            "n_normal": int((1 - y).sum()),
            "anomaly_rate": float(y.mean()),
        },
        "environment": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "cpu_count": os.cpu_count(),
            "dependencies": dependency_versions(),
        },
        "results": results,
    }
    with open(REPORTS_DIR / "evaluation.json", "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    scores_out.to_csv(REPORTS_DIR / "test_scores.csv", lineterminator="\n")

    # ---- console table -----------------------------------------------------
    print(f"{'model_version':40s} {'recall':>7s} {'fpr':>6s} {'prec':>6s} {'auc':>6s} {'ap':>6s}")
    for r in results:
        op, rk = r["operating_point"], r["ranking"]
        print(
            f"{r['model_version']:40s} {op['recall']:7.3f} {op['fpr']:6.3f} {op['precision']:6.3f} {rk['roc_auc']:6.3f} {rk['average_precision']:6.3f}"
        )

    _figures(results, scores_out, y)


def _figures(results: list[dict], scores: pd.DataFrame, y: np.ndarray) -> None:
    primary = next(r for r in results if r["is_primary"])
    pv = primary["model_version"]
    full = [r for r in results if r["channel_set"] == "full_18"]

    # confusion matrix of the primary model
    c = primary["operating_point"]["confusion"]
    m = np.array([[c["TN"], c["FP"]], [c["FN"], c["TP"]]])
    fig, ax = plt.subplots(figsize=(4.2, 3.8))
    ax.imshow(m, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(
                j,
                i,
                str(m[i, j]),
                ha="center",
                va="center",
                fontsize=14,
                color="white" if m[i, j] > m.max() / 2 else "black",
            )
    ax.set_xticks([0, 1], ["pred normal", "pred anomaly"])
    ax.set_yticks([0, 1], ["true normal", "true anomaly"])
    ax.set_title(f"{pv}\nfrozen threshold, test set")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "confusion_primary.png", dpi=130)
    plt.close(fig)

    # per-category recall, 3 models on full_18
    cats = [row["group"] for row in full[0]["per_category"]]
    w = 0.27
    fig, ax = plt.subplots(figsize=(12, 4.5))
    for k, r in enumerate(full):
        rec = [row["recall"] for row in r["per_category"]]
        ax.bar(np.arange(len(cats)) + (k - 1) * w, rec, w, label=r["model"])
    n = [row["n"] for row in full[0]["per_category"]]
    ax.set_xticks(np.arange(len(cats)), [f"{c}\n(n={k})" for c, k in zip(cats, n)], rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("recall at frozen threshold")
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.set_title("Recall per anomaly category (18 channels)")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "category_recall.png", dpi=130)
    plt.close(fig)

    # primary score distribution on test
    s = scores[pv].to_numpy()
    tau = primary["operating_point"]["threshold"]
    bins = np.linspace(s.min(), s.max(), 50)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(s[y == 0], bins=bins, alpha=0.6, label="test normal", density=True)
    ax.hist(s[y == 1], bins=bins, alpha=0.6, label="test anomaly", density=True)
    ax.axvline(tau, color="k", ls="--", label="frozen threshold")
    ax.set_xlabel("anomaly score")
    ax.legend()
    ax.set_title(f"Test score distribution — {pv}")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "test_scores_primary.png", dpi=130)
    plt.close(fig)

    # ROC + PR, 3 models full_18
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for r in full:
        sc = scores[r["model_version"]].to_numpy()
        fpr, tpr, _ = roc_curve(y, sc)
        p, rc, _ = precision_recall_curve(y, sc)
        axes[0].plot(fpr, tpr, label=f"{r['model']} AUC={r['ranking']['roc_auc']:.3f}")
        axes[1].plot(rc, p, label=f"{r['model']} AP={r['ranking']['average_precision']:.3f}")
        op = r["operating_point"]
        axes[0].plot(op["fpr"], op["recall"], "o", color=axes[0].lines[-1].get_color())
    axes[0].plot([0, 1], [0, 1], "k:", lw=0.8)
    axes[1].axhline(y.mean(), color="k", ls=":", lw=0.8, label=f"anomaly rate {y.mean():.3f}")
    axes[0].set(xlabel="FPR", ylabel="recall (TPR)", title="ROC (dots = frozen operating point)")
    axes[1].set(xlabel="recall", ylabel="precision", title="Precision-recall")
    for ax in axes:
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "roc_pr.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
