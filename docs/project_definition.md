# Project definition — robot cycle anomaly detection (v1)

## Question
After a six-axis robot finishes one pick-and-place cycle, decide from that
cycle's motor current, joint velocity and torque-sensor signals whether the
whole cycle was anomalous, and point to the signals worth inspecting.

## Contract
| Item | v1 decision |
|---|---|
| Sample unit | One complete pick-and-place cycle (`sample` id). Its time rows are never split across sets. |
| Input X | 18 whitelisted channels → 6 statistics each → 108 features (frozen in `configs/v1.json`). |
| Label y | Evaluation only: normal = 0, anomaly = 1. The anomaly `category` is used for stratified reporting only. |
| Training | Normal cycles only (`fit_normal`). No anomaly examples are needed or used for fitting. |
| Output | `is_anomaly`, `anomaly_score` (larger = more anomalous), `threshold`, `model_version`, `schema_version`, plus a separate reference-deviation hint. |
| Primary model | Isolation Forest on 18 channels (pre-specified before any test evaluation). |

## Why train on normal operation only
Normal cycles are plentiful and stable, while faults are rare, diverse and
partly unknown in advance. A model of normal behaviour can flag any cycle that
departs from it, including fault types that were never seen. The price is that
it flags *any* departure, including harmless changes in operating conditions,
so the false-alarm rate on new normal data has to be measured, not assumed.

## Scope boundary
v1 checks a completed cycle offline. Out of scope: real-time collision
warning, locating the moment an anomaly starts, root-cause diagnosis, robot
control, trajectory planning, vision. Results hold only for the voraus-AD
setup that was actually tested; there is no cross-robot data. A detection
score is not a fault probability.

## Data source and licence
- Dataset: voraus-AD, 100 Hz version (Brockmann et al., *The voraus-AD Dataset
  for Anomaly Detection in Robot Applications*), from the official repository
  `vorausrobotik/voraus-ad-dataset`.
- Data licence: CC BY-NC-SA 4.0. Reference code licence: MIT.
- Local file: `data/raw/voraus-ad-dataset-100hz.parquet`, 1 115 942 833 bytes,
  SHA-256 `c90ab1c78af52651b954d41787f7e89d750f0a128b57600b0e5ceec22621f704`
  (downloaded 2026-10-10, see `reports/download_check_2026-10-10.json`).
- The raw data is read-only and is not committed to this repository.
