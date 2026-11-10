"""Evaluation metrics. Positive class = anomaly (y = 1)."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def confusion(y: np.ndarray, pred: np.ndarray) -> dict:
    y = np.asarray(y).astype(bool)
    pred = np.asarray(pred).astype(bool)
    return {
        "TP": int(np.sum(y & pred)),
        "FN": int(np.sum(y & ~pred)),
        "FP": int(np.sum(~y & pred)),
        "TN": int(np.sum(~y & ~pred)),
    }


def _ratio(a: int, b: int) -> float | None:
    return float(a / b) if b else None


def operating_point(y: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    pred = np.asarray(scores) > threshold
    c = confusion(y, pred)
    recall = _ratio(c["TP"], c["TP"] + c["FN"])
    fpr = _ratio(c["FP"], c["FP"] + c["TN"])
    precision = _ratio(c["TP"], c["TP"] + c["FP"])
    f1 = None
    if precision is not None and recall is not None and (precision + recall) > 0:
        f1 = 2 * precision * recall / (precision + recall)
    return {"threshold": float(threshold), "confusion": c, "recall": recall, "fpr": fpr, "precision": precision, "f1": f1}


def ranking(y: np.ndarray, scores: np.ndarray) -> dict:
    y = np.asarray(y).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y, scores)),
        "average_precision": float(average_precision_score(y, scores)),
        "ap_reference_anomaly_rate": float(y.mean()),
        "n": int(y.size),
        "n_anomaly": int(y.sum()),
        "n_normal": int((1 - y).sum()),
    }


def per_group_recall(groups: np.ndarray, y: np.ndarray, scores: np.ndarray, threshold: float) -> list[dict]:
    """For anomalous groups: TP/FN/recall. For normal groups: FP/TN/FPR. Same frozen threshold."""
    rows = []
    groups = np.asarray(groups)
    y = np.asarray(y).astype(bool)
    pred = np.asarray(scores) > threshold
    for g in sorted(set(groups.tolist()), key=str):
        m = groups == g
        if y[m].all():
            tp, fn = int(np.sum(pred[m])), int(np.sum(~pred[m]))
            rows.append({"group": g, "kind": "anomaly", "n": int(m.sum()), "TP": tp, "FN": fn, "recall": tp / (tp + fn)})
        elif (~y[m]).all():
            fp, tn = int(np.sum(pred[m])), int(np.sum(~pred[m]))
            rows.append({"group": g, "kind": "normal", "n": int(m.sum()), "FP": fp, "TN": tn, "fpr": fp / (fp + tn)})
        else:
            raise ValueError(f"group {g!r} mixes normal and anomalous cycles")
    return rows
