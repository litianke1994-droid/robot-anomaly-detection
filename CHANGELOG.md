# Changelog

## 1.1.0 — 2026-11-10
### Model (exploratory, see docs/validation_protocol_v1_1.md)
- New served model: Mahalanobis + LOF(n=10) ensemble on a multi-condition normal
  reference, with two-tier thresholds (warning = q95, alarm = q99 of a
  multi-condition calibration pool).
- Holdout (580 cycles, scored once): recall 0.827, FPR 0.095, AUC 0.948
  (v1 Isolation Forest: 0.759 / 0.455 / 0.724).
- Validation search: 68 candidates in 6 logged rounds. Detectors added:
  Mahalanobis, kNN, LOF, one-class SVM, ECOD, score combinations, and a 1D-CNN
  autoencoder (PyTorch).

### API / CLI / app
- `alert_level` (normal / warning / alarm) and `alert_levels` in responses and `/health`.
- Invalid JSON now returns `{"error": {"code": "invalid_json"}}`, like every other rejection.
- Signal arrays must hold JSON numbers only; strings, booleans and null are rejected.
- `timing_ms.features` renamed to `timing_ms.validate_and_features`.
- Model warm-up at API start: the first request no longer pays ~1.4 s.
- Replay page: alert level, 2×3 signal grid, cycle-mean line against the normal band.

### Fixes found along the way
- The bundle JSON sidecar crashed on bundles holding a scorer object.
- The cross-round search cache was keyed with salted `hash()`; it now uses SHA-1.
- kNN/LOF combination normalization used in-sample scores; it now uses leave-one-out scores.

### Tooling
- ruff config and formatting, package metadata, `robot-ad` console script,
  pinned httpx/ruff, 39 tests.

## 1.0.0 — 2026-11-10
- First complete version: audit, frozen split, 108 features, three baselines,
  calibration, frozen test evaluation (primary Isolation Forest: not passed),
  error analysis, CLI, FastAPI, Streamlit replay, Dockerfile.
