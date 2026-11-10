# Validation protocol v1.1 (registered before any v1.1 model is fitted)

## Why a v1.1
v1 failed its effectiveness criterion: the pre-specified Isolation Forest did
not beat the z-score baseline, and every model raised 20–54 % false alarms on
normal cycles from settings it had never seen (technical report §5). v1
results stay unchanged in `reports/evaluation.json`.

v1.1 asks a production-shaped question: **if the normal reference and the
alarm threshold include normal cycles from several operating conditions (as a
deployed system would collect during commissioning), which detector works
best, and how well does it do on cycles it has never seen?**

## Honesty constraint
The official test part was scored in v1, so it is no longer an untouched test
set. v1.1 therefore splits it once into **validation** and **holdout**:

- All model search, as many rounds as needed, uses the validation half only.
  Every round is appended to `reports/v1_1/search_log.jsonl`, including
  rounds that did not help.
- The final configuration is committed to Git **before** the holdout is scored.
- The holdout is scored exactly once.
- Because v1 already showed aggregate metrics on the whole test part,
  v1.1 holdout numbers are labelled **exploratory**, not an independent
  confirmation.

## Split (`scripts/09_v11_split.py`, frozen in `reports/v1_1/split_manifest_v1_1.csv`)
| v1 role | v1.1 role | Content |
|---|---|---|
| fit_normal | fit_normal | PRE_A 570 |
| dev_normal | dev_normal | PRE_A 188 (checks only) |
| cal_normal | cal_normal | PRE_A 190 |
| test | val_normal_fit | ¼ of test normals (stratified by setting) |
| test | val_normal_cal | ¼ of test normals |
| test | val_anomaly | ½ of test anomalies (stratified by category) |
| test | holdout | the other ½ of test normals and anomalies |

Units are blocks of 5 consecutive sample ids inside each stratum (setting for
normal, category for anomaly), shuffled with seed 20261110, so neighbouring
cycles never sit on both sides of a boundary. Holdout normals come from the
same settings as validation normals (a "known conditions" scenario);
anomaly variants also appear on both sides. Generalisation to new settings or
new fault variants is **not** measured by v1.1.

## Reference and threshold
- Reference options: `pre_a` (fit_normal only) or `multi` (fit_normal +
  val_normal_fit).
- Calibration pool: cal_normal + val_normal_cal, never used for fitting.
- Two-tier alerting: **warning** = 95 % quantile of the pool,
  **alarm** = 99 % quantile; `is_anomaly` ⇔ score > warning. Both operating
  points are reported.

## Search on validation
For each candidate (detector × reference option × hyper-parameters):
5 repeats; in each repeat the validation normals are re-split at block
level into fit/cal halves (seed 7 + repeat). Criterion: mean TPR at 5 % FPR
(val anomalies vs calibration-pool normals); tie-break: mean ROC-AUC.
Candidate families: v1 detectors (z-score, PCA, Isolation Forest), Mahalanobis
(Ledoit-Wolf), kNN distance, LOF, one-class SVM, ECOD, score combinations,
and — if PyTorch is available — a 1D-CNN autoencoder on the raw sequences.

## Holdout report (`scripts/12_v11_holdout.py`)
For the selected v1.1 model, v1 Isolation Forest and v1 z-score, all on the
same holdout: recall/FPR/precision/F1 at both levels with counts, ROC-AUC, AP
(+ anomaly-rate reference), per category, per normal setting, and 95 %
bootstrap intervals (2000 resamples) including paired differences vs v1.
