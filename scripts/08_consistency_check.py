"""Week 7-9: software evidence.

1. Consistency: for every exported example, compare the score from
   (a) the batch evaluation (reports/test_scores.csv),
   (b) a freshly reloaded bundle on the raw payload,
   (c) the CLI in a subprocess,
   (d) the FastAPI app (in-process TestClient).
   Pass: identical decision and max |score difference| <= 1e-6.
2. Input validation: every invalid payload must be rejected by CLI (exit 2)
   and API (HTTP 422) with the expected error code.
3. /health response.

Output: reports/consistency.json
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from robot_ad.bundle import now_utc
from robot_ad.cli import DEFAULT_BUNDLE
from robot_ad.config import EXAMPLES_DIR, PROCESSED_DIR, PROJECT_ROOT, REPORTS_DIR, load_config, raw_path
from robot_ad.predict import Predictor

TOL = 1e-6


def batch_scores(pred: Predictor, samples: list[int]) -> pd.Series:
    """Batch path: precomputed feature table (+ resampled raw sequences if the model needs them)."""
    feats = pd.read_parquet(PROCESSED_DIR / "features_v1.parquet").loc[samples, pred.bundle["feature_names"]]
    det = pred.scorer if pred.scorer is not None else pred.bundle["detector"]
    if pred.scorer is not None and pred.scorer.needs_sequence:
        from robot_ad.sequence import resample_cycle

        cfg = load_config()
        df = pd.read_parquet(
            raw_path(cfg), columns=["sample", "time"] + pred.channels, filters=[("sample", "in", samples)]
        )
        seq = {s: resample_cycle(g["time"].to_numpy(), g[pred.channels].to_numpy()) for s, g in df.groupby("sample")}
        S = np.stack([seq[s] for s in samples])
        return pd.Series(det.score(feats.to_numpy(), S), index=samples)
    if pred.scorer is not None:
        return pd.Series(det.score(feats.to_numpy(), None), index=samples)
    return pd.Series(det.score(feats.to_numpy()), index=samples)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", default=str(DEFAULT_BUNDLE))
    bundle_path = Path(ap.parse_args().bundle)
    os.environ["ROBOT_AD_BUNDLE"] = str(bundle_path)  # the API must serve the same bundle
    pred = Predictor.from_path(bundle_path)
    pv = pred.model_version
    index = json.loads((EXAMPLES_DIR / "index.json").read_text(encoding="utf-8"))
    batch = batch_scores(pred, [item["sample"] for item in index])

    sys.path.insert(0, str(PROJECT_ROOT))  # make the top-level api package importable
    from api.main import app

    rows = []
    with TestClient(app) as client:
        health = client.get("/health")
        for item in index:
            path = EXAMPLES_DIR / item["file"]
            payload = json.loads(path.read_text(encoding="utf-8"))
            a = float(batch.loc[item["sample"]])
            b = pred.predict_payload(payload)
            cli = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "robot_ad.cli",
                    "--bundle",
                    str(bundle_path),
                    "predict",
                    "--input",
                    str(path),
                    "--compact",
                ],
                capture_output=True,
                text=True,
                cwd=PROJECT_ROOT,
                check=True,
            )
            c = json.loads(cli.stdout)
            d = client.post("/predict", content=path.read_bytes(), headers={"content-type": "application/json"}).json()
            scores = [a, b["anomaly_score"], c["anomaly_score"], d["anomaly_score"]]
            decisions = [a > pred.threshold, b["is_anomaly"], c["is_anomaly"], d["is_anomaly"]]
            rows.append(
                {
                    "file": item["file"],
                    "sample": item["sample"],
                    "known_label_for_reference": "anomaly" if item["anomaly"] else "normal",
                    "score_batch": a,
                    "score_reload": scores[1],
                    "score_cli": scores[2],
                    "score_api": scores[3],
                    "max_abs_diff": max(scores) - min(scores),
                    "decision": bool(decisions[0]),
                    "decisions_agree": len(set(decisions)) == 1,
                }
            )

        invalid = []
        for f in sorted((EXAMPLES_DIR / "invalid").glob("*.json")):
            expected = f.stem
            cli = subprocess.run(
                [sys.executable, "-m", "robot_ad.cli", "--bundle", str(bundle_path), "predict", "--input", str(f)],
                capture_output=True,
                text=True,
                cwd=PROJECT_ROOT,
            )
            r = client.post("/predict", content=f.read_bytes(), headers={"content-type": "application/json"})
            api_code = r.json().get("error", {}).get("code")
            cli_code = json.loads(cli.stderr.strip().splitlines()[-1])["error"]["code"] if cli.returncode == 2 else None
            invalid.append(
                {
                    "case": expected,
                    "cli_exit_code": cli.returncode,
                    "cli_error_code": cli_code,
                    "api_status": r.status_code,
                    "api_error_code": api_code,
                    "api_message": r.json().get("error", {}).get("message"),
                    "pass": cli.returncode == 2
                    and cli_code == expected
                    and r.status_code == 422
                    and api_code == expected,
                }
            )

    max_diff = max(r["max_abs_diff"] for r in rows)
    out = {
        "created_utc": now_utc(),
        "model_version": pv,
        "tolerance": TOL,
        "n_examples": len(rows),
        "max_abs_score_diff": max_diff,
        "all_decisions_agree": all(r["decisions_agree"] for r in rows),
        "consistency_pass": max_diff <= TOL and all(r["decisions_agree"] for r in rows),
        "examples": rows,
        "invalid_inputs": invalid,
        "invalid_inputs_pass": all(i["pass"] for i in invalid),
        "health": {"status_code": health.status_code, "body": health.json()},
    }
    with open(REPORTS_DIR / "consistency.json", "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    print(f"examples={len(rows)} max_abs_diff={max_diff:.3e} consistency_pass={out['consistency_pass']}")
    for i in invalid:
        print(
            f"  {i['case']:28s} cli_exit={i['cli_exit_code']} api={i['api_status']} {i['api_error_code']} pass={i['pass']}"
        )
    print("health", health.status_code, health.json()["model_version"])


if __name__ == "__main__":
    main()
