# Robot cycle anomaly detection (voraus-AD)

Offline anomaly detection for a six-axis pick-and-place robot. After a cycle
finishes, the system scores the whole cycle from its motor currents, joint
velocities and torque signals, compares the score with a calibrated threshold,
and points to the channels worth inspecting. Models learn **normal operation
only**; anomalies are used for evaluation, never for training.

**Status: v1 complete, with a negative main result reported as it came out.**
The pre-registered primary model (Isolation Forest) did *not* beat a simple
z-score baseline, and every model raises far more false alarms on normal
cycles from other recording settings than the 5 % calibration target. Details
are in the [technical report](docs/technical_report.md).

## Results at a glance (frozen test set: 1174 cycles, 755 anomalous)

| Model (18 channels) | Recall | False-alarm rate | ROC-AUC | AP |
|---|---|---|---|---|
| max \|z\| baseline | 0.743 | 0.253 | 0.829 | 0.911 |
| PCA reconstruction | 0.824 | 0.272 | 0.877 | 0.939 |
| Isolation Forest (primary, pre-specified) | 0.760 | 0.425 | 0.744 | 0.841 |

Random ranking: AUC 0.5, AP 0.643 (test anomaly rate). Thresholds: 95 %
quantile of 190 held-out normal training cycles, fixed before testing.

![per-category recall](reports/figures/category_recall.png)

## How it works

```text
Parquet (100 Hz) ──► one cycle (T×18) ──► 108 statistics ──► detector ──► score > τ ? ──► CLI / API / replay page
                         audit            mean std min max     fitted on     τ from separate
                                          RMS q95|x|           normal only   normal cycles
```

| Role | Cycles | Purpose |
|---|---|---|
| fit_normal | 570 | fit scalers / PCA / Isolation Forest |
| dev_normal | 188 | stability checks |
| cal_normal | 190 | threshold only |
| test (official) | 1174 | one evaluation after freezing ([protocol](docs/validation_protocol.md)) |

The protocol was committed before the test set was scored; the Git history
shows the order (`cef9b9a` freeze → `5cfbc83` evaluation).

## Repository layout

```text
configs/v1.json          channels, features, split, models, threshold, input contract
src/robot_ad/            features, detectors, schema validation, bundle, Predictor, CLI
scripts/01–08            audit → split → features → train/calibrate → evaluate → errors → examples → consistency
api/main.py              FastAPI: GET /health, POST /predict
app/replay.py            Streamlit replay page
tests/                   24 unit tests on synthetic data (no dataset needed)
docs/                    definition, data dictionary, protocol, technical report, demo script, acceptance
reports/                 audit, split manifest, training/evaluation JSON, figures, software evidence
```

## Reproduce

Requires Python 3.12. The dataset (≈1.1 GB, CC BY-NC-SA 4.0) is not in this
repository; download the **100 Hz** Parquet from the official
[voraus-AD repository](https://github.com/vorausrobotik/voraus-ad-dataset) to
`data/raw/voraus-ad-dataset-100hz.parquet`.

```bash
python -m venv .venv
```

```bash
.venv/Scripts/python -m pip install -r requirements.txt -e .
```

Use `.venv/bin/python` on Linux/macOS. Then run the pipeline in order:

```bash
python scripts/01_audit.py
```

```bash
python scripts/02_make_split.py
```

```bash
python scripts/03_extract_features.py
```

```bash
python scripts/04_train_calibrate.py
```

```bash
python scripts/05_evaluate.py
```

```bash
python scripts/06_error_analysis.py
```

```bash
python scripts/07_export_examples.py
```

```bash
python scripts/08_consistency_check.py
```

```bash
python -m pytest
```

Each step reads only the outputs of earlier steps; the raw file is never modified.

## Use

CLI (exit code 2 on invalid input):

```bash
python -m robot_ad.cli predict --input data/examples/test_normal_1963.json
```

API:

```bash
uvicorn api.main:app --port 8000
```

```bash
curl -X POST http://127.0.0.1:8000/predict -H "content-type: application/json" --data @data/examples/test_normal_1963.json
```

Replay page:

```bash
streamlit run app/replay.py
```

Request (`robot-cycle-v1`): `schema_version`, `time_s` (T values, seconds,
strictly increasing) and `signals` with exactly the 18 channels
`motor_iq_1..6`, `joint_velocity_1..6`, `torque_sensor_a_1..6`, each of
length T (900–1300). Response: `is_anomaly`, `anomaly_score`, `threshold`,
`model_version`, `schema_version`, plus `reference_deviations` — the channels
whose statistics deviate most from normal training cycles (an inspection
hint, not an explanation of the model).

## Docker

```bash
docker build -t robot-ad-api:1.0.0 .
```

```bash
docker run --rm -p 8000:8000 robot-ad-api:1.0.0
```

The image serves the bundle in `models/` and never retrains. **Not yet built
on the development machine** (no Docker installed); see
[reports/docker_status.md](reports/docker_status.md).

## Scope and limitations
Offline, whole-cycle detection on one robot cell. No real-time warning, no
localisation of the anomaly in time, no root-cause diagnosis. A score is not a
fault probability. Results are limited to the tested voraus-AD settings.

## Data and licences
- Data: voraus-AD dataset, Brockmann et al., *The voraus-AD Dataset for
  Anomaly Detection in Robot Applications*, released under CC BY-NC-SA 4.0
  by voraus robotik. Not redistributed here; derived artefacts in `reports/`
  (cycle ids, labels, aggregate statistics, figures) are shared under the same
  licence for non-commercial use.
- Category and setting names follow the official loader `voraus_ad.py` (MIT).
- Code in this repository: MIT ([LICENSE](LICENSE)).
