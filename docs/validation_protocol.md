# Validation protocol v1 (frozen before test evaluation)

This file is committed **before** `scripts/05_evaluate.py` is first run. The
Git history is the evidence that models, features, thresholds and metrics did
not depend on the test set.

## 1. Data roles
| Role | Source | n | Used for |
|---|---|---|---|
| `fit_normal` | official PRE_A normal | 570 | fit scalers, PCA, Isolation Forest, deviation statistics |
| `dev_normal` | official PRE_A normal | 188 | pipeline and stability checks (no test labels) |
| `cal_normal` | official PRE_A normal | 190 | threshold calibration only |
| `test` | official non-PRE_A part | 1174 (419 normal, 755 anomalous) | one formal evaluation after freezing |

Split rule: the 948 PRE_A cycles, sorted by `sample`, are cut into 95 blocks of
10 consecutive ids. The blocks are shuffled with seed 20261010 and assigned
60/20/20 (57/19/19 blocks). The recording order is not documented, so we do
not claim the blocks are acquisition sessions; they only keep neighbouring
cycles together. The manifest is `reports/split_manifest.csv`
(SHA-256 in `reports/split_summary.json`). No sample appears in two roles.

## 2. Features
18 channels × {mean, std (ddof=0), min, max, RMS, q95 of |x| (linear)} = 108
columns, one row per cycle, no padding or truncation (`reports/feature_spec.json`).
Cycle duration is not a feature.

## 3. Registered experiments (all fitted on `fit_normal` only)
| Model | Score (larger = more anomalous) | Channel sets |
|---|---|---|
| `zscore_max` | max |z| over features; zero-variance features excluded | full_18, current_6 |
| `pca_recon` | standardize, PCA keeping 95 % variance, mean squared reconstruction error | full_18, current_6 |
| `iforest` | Isolation Forest (300 trees, seed 42), `-score_samples` | full_18, current_6 |

Primary (served) model: **iforest / full_18**, fixed in `configs/v1.json`
before testing. The current_6 vs full_18 comparison is the pre-registered
ablation. No experiment is dropped because of its result.

## 4. Threshold rule
For each configuration separately: τ = 95 % quantile (linear) of its
`cal_normal` scores; alarm iff `score > τ`. Ties at τ are counted and
reported. The 5 % is a calibration target on PRE_A normal data; it does not
guarantee 5 % false alarms on other settings. A refitted model must be
recalibrated; v1 does not merge fit/dev/cal and refit.

## 5. Metrics (positive class = anomaly)
At the frozen threshold: Recall and FPR with raw TP/FN/FP/TN, Precision, F1.
Threshold-free: ROC-AUC and Average Precision, with the test anomaly rate as
the AP reference (random ranking AUC ≈ 0.5). Per anomaly category: n, TP, FN,
recall. Per test normal setting: n, FP, FPR. Timing: feature extraction and
scoring per cycle, with the runtime environment.

## 6. Pass criteria for model effectiveness
The primary model should improve on the simple `zscore_max` baseline in the
recall/FPR trade-off at the frozen operating point. If not, or if test FPR is
far above the calibration target, model effectiveness is reported as
“not passed / limited evidence” with a reason analysis. No fixed detection
rate is promised.

## 7. Discipline
- Preprocessing is fitted only on `fit_normal`.
- `test` is not used to choose channels, models, thresholds or to drop samples.
- After the test run, at least 5 false-alarm / missed / borderline cases are analysed.
- Any change after the test run is a new version (v1.1+) and its results are
  labelled exploratory, because the test set has been seen.

## 8. Checks done on normal data before freezing (from `reports/training_summary.json`)
- Dev alarm rate at the calibrated threshold is 2.7–8.5 % across the six
  configurations (target 5 %). With 190 calibration cycles, the 95 % quantile
  rests on about 10 cycles, so the alarm rate on new normal data will vary
  by several percentage points.
- Isolation Forest with 5 other seeds: dev-score Spearman ≥ 0.93 against the
  frozen seed; dev alarm rate 1.1–6.4 % — the operating point is seed-sensitive
  at this sample size. The frozen seed stays 42.
