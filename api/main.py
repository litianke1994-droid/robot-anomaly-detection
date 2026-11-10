"""FastAPI service. Uses the same Predictor as the CLI and the replay page.

    uvicorn api.main:app --port 8000

Environment:
    ROBOT_AD_BUNDLE  path to the model bundle (default models/iforest__full_18.joblib)
"""

from __future__ import annotations

import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Any

from fastapi import Body, FastAPI
from fastapi.responses import JSONResponse

from robot_ad.config import MODELS_DIR
from robot_ad.predict import Predictor
from robot_ad.schema import InputError

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger("robot_ad.api")

BUNDLE_PATH = os.environ.get("ROBOT_AD_BUNDLE", str(MODELS_DIR / "iforest__full_18.joblib"))

_state: dict[str, Any] = {"predictor": None, "load_error": None}


@asynccontextmanager
async def lifespan(_: FastAPI):
    get_predictor()  # load once at startup; /health reports failures
    yield


app = FastAPI(title="Robot cycle anomaly detection", version="1.0.0", lifespan=lifespan)


def get_predictor() -> Predictor | None:
    if _state["predictor"] is None and _state["load_error"] is None:
        try:
            _state["predictor"] = Predictor.from_path(BUNDLE_PATH)
            log.info("bundle loaded model_version=%s", _state["predictor"].model_version)
        except Exception as e:  # reported by /health
            _state["load_error"] = f"{type(e).__name__}: {e}"
            log.error("bundle load failed: %s", _state["load_error"])
    return _state["predictor"]


@app.get("/health")
def health() -> JSONResponse:
    p = get_predictor()
    if p is None:
        return JSONResponse(status_code=503, content={"status": "not_ready", "error": _state["load_error"]})
    return JSONResponse(content={
        "status": "ok",
        "model_ready": True,
        "model_version": p.model_version,
        "schema_version": p.schema_version,
        "threshold": p.threshold,
        "channels": p.channels,
    })


@app.post("/predict")
def predict(payload: Any = Body(...)) -> JSONResponse:
    p = get_predictor()
    if p is None:
        return JSONResponse(status_code=503, content={"error": {"code": "model_not_ready", "message": _state["load_error"]}})
    t0 = time.perf_counter()
    try:
        result = p.predict_payload(payload)
    except InputError as e:
        log.info("predict rejected code=%s elapsed_ms=%.2f", e.code, (time.perf_counter() - t0) * 1e3)
        return JSONResponse(status_code=422, content={"error": e.to_dict()})
    log.info(
        "predict ok model_version=%s n_points=%d score=%.6f is_anomaly=%s elapsed_ms=%.2f",
        result["model_version"], result["n_points"], result["anomaly_score"], result["is_anomaly"],
        (time.perf_counter() - t0) * 1e3,
    )
    return JSONResponse(content=result)
