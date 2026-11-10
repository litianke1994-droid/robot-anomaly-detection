"""v1.1 model search on the validation half only. Run round by round:

    python scripts/10_v11_search.py --round 1

Holdout cycles are dropped before any data is loaded into memory. Every
candidate result is appended to reports/v1_1/search_log.jsonl.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import time

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from robot_ad.bundle import now_utc
from robot_ad.config import PROCESSED_DIR, PROJECT_ROOT, REPORTS_DIR, load_config, raw_path
from robot_ad.features import feature_names
from robot_ad.models import make_detector

OUT = REPORTS_DIR / "v1_1"
LOG = OUT / "search_log.jsonl"
CACHE = PROCESSED_DIR / "v11_member_cache.pkl"
SEQ_CACHE = PROCESSED_DIR / "seq_1024_nonholdout.npz"
CFG11 = json.loads((PROJECT_ROOT / "configs" / "v1_1.json").read_text(encoding="utf-8"))
CFG = load_config()


# --------------------------------------------------------------------------- data
def load_data():
    man = pd.read_csv(OUT / "split_manifest_v1_1.csv", index_col="sample")
    man = man[man["role_v1_1"] != "holdout"]  # holdout never enters this script
    names = feature_names(CFG["channels"]["full_18"])
    F = pd.read_parquet(PROCESSED_DIR / "features_v1.parquet").loc[man.index, names]
    return man, F.to_numpy(), names


def load_sequences(man: pd.DataFrame) -> np.ndarray:
    from robot_ad.sequence import resample_cycle

    if SEQ_CACHE.exists():
        z = np.load(SEQ_CACHE)
        if np.array_equal(z["samples"], man.index.to_numpy()):
            return z["S"]
    ch = CFG["channels"]["full_18"]
    df = pd.read_parquet(raw_path(CFG), columns=["sample", "time"] + ch, filters=[("sample", "in", man.index.tolist())])
    S = np.stack([resample_cycle(g["time"].to_numpy(), g[ch].to_numpy()) for _, g in df.groupby("sample", sort=True)])
    np.savez_compressed(SEQ_CACHE, S=S, samples=man.index.to_numpy())
    return S


def block_split(ids: np.ndarray, seed: int, block: int = 5) -> tuple[np.ndarray, np.ndarray]:
    ids = np.sort(ids)
    blocks = [ids[i : i + block] for i in range(0, len(ids), block)]
    order = np.random.default_rng(seed).permutation(len(blocks))
    k = round(len(blocks) / 2)
    return (np.concatenate([blocks[b] for b in order[:k]]), np.concatenate([blocks[b] for b in order[k:]]))


def repeat_splits(man: pd.DataFrame) -> list[dict]:
    vn = man[man["role_v1_1"].isin(["val_normal_fit", "val_normal_cal"])]
    out = []
    for r in range(CFG11["search"]["repeats"]):
        fit_parts, cal_parts = [], []
        for _s, g in vn.groupby("setting"):
            a, b = block_split(g.index.to_numpy(), CFG11["search"]["repeat_seed"] + r)
            fit_parts.append(a)
            cal_parts.append(b)
        out.append({"vfit": np.concatenate(fit_parts), "vcal": np.concatenate(cal_parts)})
    return out


# ------------------------------------------------------------------------ members
def member_key(spec: dict, ref: str, fit_ids: np.ndarray) -> str:
    # hashlib, not hash(): Python's hash() of bytes is salted per process, which silently disabled the cache
    fit = hashlib.sha1(np.asarray(fit_ids, dtype=np.int64).tobytes()).hexdigest()[:16]
    return json.dumps({"spec": spec, "ref": ref, "fit": fit, "v": 2}, sort_keys=True)


def build_member(spec: dict):
    if spec["name"] == "seq_ae":
        from robot_ad.sequence import SeqAutoencoder

        return SeqAutoencoder(**spec.get("kw", {}))
    return make_detector(spec["name"], CFG["models"], **spec.get("kw", {}))


def member_scores(spec, ref, fit_ids, man, F, S, names, cache) -> dict:
    """Fit member on fit_ids; return scores on every non-holdout cycle + normalization stats."""
    fit_ids_key = np.sort(fit_ids)
    key = member_key(spec, ref, fit_ids_key)
    if key in cache:
        return cache[key]
    pos = man.index.get_indexer(fit_ids_key)
    det = build_member(spec)
    is_seq = getattr(det, "input_kind", "features") == "sequence"
    X = S if is_seq else F
    t0 = time.perf_counter()
    det.fit(X[pos], names)
    t_fit = time.perf_counter() - t0
    s_all = det.score(X)
    s_fit = getattr(det, "train_scores_", None)
    s_fit = s_all[pos] if s_fit is None else s_fit  # leave-one-out for kNN / LOF
    med, q95 = float(np.median(s_fit)), float(np.quantile(s_fit, 0.95))
    res = {"scores": s_all, "norm": (med, max(q95 - med, 1e-12)), "fit_seconds": t_fit}
    cache[key] = res
    return res


# ----------------------------------------------------------------------- evaluate
def evaluate(cand: dict, man, F, S, names, splits, cache) -> dict:
    roles = man["role_v1_1"]
    pos_mask = (roles == "val_anomaly").to_numpy()
    cal_pre_a = (roles == "cal_normal").to_numpy()
    tprs, aucs, fit_s = [], [], 0.0
    nonprea_fpr = []
    for sp in splits:
        fit_ids = man.index[roles == "fit_normal"].to_numpy()
        if cand["ref"] == "multi":
            fit_ids = np.concatenate([fit_ids, sp["vfit"]])
        parts = []
        for spec in cand["members"]:
            m = member_scores(spec, cand["ref"], fit_ids, man, F, S, names, cache)
            fit_s += m["fit_seconds"]
            med, scale = m["norm"]
            parts.append((m["scores"] - med) / scale)
        score = np.mean(parts, axis=0)
        neg_mask = cal_pre_a | man.index.isin(sp["vcal"])
        tau = np.quantile(score[neg_mask], 0.95)
        tprs.append(float(np.mean(score[pos_mask] > tau)))
        y = np.r_[np.ones(pos_mask.sum()), np.zeros(neg_mask.sum())]
        aucs.append(float(roc_auc_score(y, np.r_[score[pos_mask], score[neg_mask]])))
        other = man.index.isin(sp["vcal"])
        nonprea_fpr.append(float(np.mean(score[other] > tau)))
    return {
        "tpr_at_5fpr_mean": float(np.mean(tprs)),
        "tpr_at_5fpr_std": float(np.std(tprs)),
        "auc_mean": float(np.mean(aucs)),
        "auc_std": float(np.std(aucs)),
        "fpr_on_non_pre_a_cal_normals_mean": float(np.mean(nonprea_fpr)),
        "fit_seconds_total": round(fit_s, 2),
    }


# ------------------------------------------------------------------------- rounds
def M(name, **kw):
    return {"name": name, "kw": kw} if kw else {"name": name}


def C(label, ref, *members):
    return {"label": label, "ref": ref, "members": list(members)}


def round_candidates(r: int, best: list[dict]) -> list[dict]:
    if r == 1:  # all detector families, default settings, both references
        fams = ["zscore_max", "pca_recon", "iforest", "mahalanobis", "knn", "lof", "ocsvm", "ecod"]
        return [C(f"{f}|{ref}", ref, M(f)) for f in fams for ref in ("pre_a", "multi")]
    if r == 2:  # hyper-parameters of the families, multi reference
        c = []
        c += [C(f"pca_recon(var={v})|multi", "multi", M("pca_recon", variance=v)) for v in (0.8, 0.9, 0.99)]
        c += [C(f"knn(k={k})|multi", "multi", M("knn", k=k)) for k in (1, 3, 10, 20)]
        c += [C(f"lof(n={k})|multi", "multi", M("lof", n_neighbors=k)) for k in (5, 10, 50)]
        c += [C(f"ocsvm(nu={nu})|multi", "multi", M("ocsvm", nu=nu)) for nu in (0.01, 0.1)]
        c += [C(f"iforest(max_features={m})|multi", "multi", M("iforest", max_features=m)) for m in (0.25, 0.5)]
        return c
    if r == 3:  # sequence autoencoder (PyTorch, GPU)
        return [
            C(f"seq_ae({mode},w={w})|{ref}", ref, M("seq_ae", score_mode=mode, width=w))
            for ref in ("pre_a", "multi")
            for mode in ("peak", "mean")
            for w in (32,)
        ] + [C("seq_ae(peak,w=64)|multi", "multi", M("seq_ae", score_mode="peak", width=64))]
    if (
        r in (4, 6)
    ):  # combinations (round 6 = round 4 rerun after the leave-one-out normalization fix) of the best single members so far (validation ranking)
        singles = [b for b in best if len(b["members"]) == 1 and b["ref"] == "multi"]
        seen, uniq = set(), []
        for b in singles:  # one per family
            fam = b["members"][0]["name"]
            if fam not in seen:
                seen.add(fam)
                uniq.append(b)
        top = uniq[:5]
        c = []
        for i in range(len(top)):
            for j in range(i + 1, len(top)):
                a, b = top[i], top[j]
                c.append(C(f"combo[{a['label']} + {b['label']}]", "multi", a["members"][0], b["members"][0]))
        if len(top) >= 3:
            c.append(C("combo[top3]", "multi", *[t["members"][0] for t in top[:3]]))
        # feature model + sequence autoencoder (different information source)
        ae = M("seq_ae", score_mode="mean", width=32)
        for t in top[:3]:
            c.append(C(f"combo[{t['label']} + seq_ae(mean)]", "multi", t["members"][0], ae))
        return c
    if r == 5:  # sequence AE (mean score) robustness: seeds, capacity, training length
        return [
            C(f"seq_ae(mean,w=32,seed={s})|multi", "multi", M("seq_ae", score_mode="mean", width=32, seed=s))
            for s in (1, 2)
        ] + [
            C("seq_ae(mean,w=64)|multi", "multi", M("seq_ae", score_mode="mean", width=64)),
            C("seq_ae(mean,w=32,epochs=160)|multi", "multi", M("seq_ae", score_mode="mean", width=32, epochs=160)),
        ]
    raise SystemExit(f"no round {r}")


def read_log() -> list[dict]:
    if not LOG.exists():
        return []
    rows = [json.loads(line) for line in LOG.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows = [d for d in rows if not d.get("superseded")]
    rows.sort(key=lambda d: (d["metrics"]["tpr_at_5fpr_mean"], d["metrics"]["auc_mean"]), reverse=True)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", type=int, required=True)
    args = ap.parse_args()

    man, F, names = load_data()
    best = read_log()
    cands = round_candidates(args.round, best)
    S = load_sequences(man) if any(m["name"] == "seq_ae" for c in cands for m in c["members"]) else None
    splits = repeat_splits(man)
    cache = pickle.loads(CACHE.read_bytes()) if CACHE.exists() else {}

    with open(LOG, "a", encoding="utf-8") as fh:
        for cand in cands:
            t0 = time.perf_counter()
            met = evaluate(cand, man, F, S, names, splits, cache)
            row = {
                "round": args.round,
                **cand,
                "metrics": met,
                "seconds": round(time.perf_counter() - t0, 1),
                "logged_utc": now_utc(),
            }
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            print(
                f"R{args.round} {cand['label']:55s} TPR@5%={met['tpr_at_5fpr_mean']:.3f}±{met['tpr_at_5fpr_std']:.3f} "
                f"AUC={met['auc_mean']:.3f} nonPRE_A-FPR={met['fpr_on_non_pre_a_cal_normals_mean']:.3f}"
            )
    CACHE.write_bytes(pickle.dumps(cache))

    print("\nTop 8 so far (validation only):")
    for d in read_log()[:8]:
        print(
            f"  R{d['round']} {d['label']:55s} TPR@5%={d['metrics']['tpr_at_5fpr_mean']:.3f} AUC={d['metrics']['auc_mean']:.3f}"
        )


if __name__ == "__main__":
    main()
