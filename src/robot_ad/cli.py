"""Command line prediction for one cycle file.

    python -m robot_ad.cli predict --input data/examples/test_normal_0001.json
    python -m robot_ad.cli info

Exit codes: 0 = predicted, 2 = invalid input, 1 = other error.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from .config import MODELS_DIR
from .predict import Predictor
from .schema import InputError

DEFAULT_BUNDLE = os.environ.get("ROBOT_AD_BUNDLE", str(MODELS_DIR / "iforest__full_18.joblib"))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="robot-ad", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bundle", default=DEFAULT_BUNDLE, help="path to a model bundle (.joblib)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("predict", help="score one cycle JSON file")
    p.add_argument("--input", required=True, type=Path)
    p.add_argument("--compact", action="store_true", help="single-line JSON output")
    sub.add_parser("info", help="print model version, threshold and input contract")
    args = ap.parse_args(argv)

    predictor = Predictor.from_path(args.bundle)
    if args.cmd == "info":
        b = predictor.bundle
        print(json.dumps({
            "model_version": b["model_version"],
            "schema_version": b["schema_version"],
            "threshold": b["threshold"],
            "input_validation": b["input_validation"],
            "git_commit": b["git_commit"],
        }, indent=2))
        return 0

    try:
        payload = json.loads(args.input.read_text(encoding="utf-8"))
        result = predictor.predict_payload(payload)
    except InputError as e:
        print(json.dumps({"error": e.to_dict()}), file=sys.stderr)
        return 2
    except json.JSONDecodeError as e:
        print(json.dumps({"error": {"code": "invalid_json", "message": str(e)}}), file=sys.stderr)
        return 2
    print(json.dumps(result, indent=None if args.compact else 2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
