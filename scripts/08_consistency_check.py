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

import json
import subprocess
import sys

import pandas as pd
from fastapi.testclient import TestClient

from robot_ad.bundle import now_utc
from robot_ad.config import EXAMPLES_DIR, MODELS_DIR, PROJECT_ROOT, REPORTS_DIR
from robot_ad.predict import Predictor

TOL = 1e-6


def main() -> None:
    bundle_path = MODELS_DIR / "iforest__full_18.joblib"
    pred = Predictor.from_path(bundle_path)
    pv = pred.model_version
    batch = pd.read_csv(REPORTS_DIR / "test_scores.csv", index_col="sample")[pv]
    index = json.loads((EXAMPLES_DIR / "index.json").read_text(encoding="utf-8"))

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
