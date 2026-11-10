# Demo video script (3–5 min) — to be recorded

Order follows the brief: task → signals/sample → normal training and
threshold → baselines and errors → API/replay → limits. Only measured numbers.

| Time | Show | Say (key points) |
|---|---|---|
| 0:00–0:30 | README top | A robot picks, moves and places cans. Question: after a cycle, was it abnormal? Offline, whole-cycle, one robot cell. |
| 0:30–1:00 | `reports/figures/single_cycle_normal.png` | One sample = one cycle, ~1100 points at 100 Hz, 18 signals. Each cycle becomes 108 statistics, so short events can be diluted. |
| 1:00–1:45 | `docs/validation_protocol.md`, `normal_score_distributions.png` | Train on 570 normal cycles only; threshold = 95 % quantile of 190 other normal cycles; protocol committed before the test run (Git history). |
| 1:45–2:45 | `roc_pr.png`, `category_recall.png`, `normal_drift.png` | Honest result: the pre-specified Isolation Forest did not beat the z-score baseline (FPR 0.425 vs 0.253). PCA ranked best (AUC 0.877) but is not promoted post hoc. False alarms come from normal drift between settings, mostly in motor currents. |
| 2:45–3:45 | Streamlit replay: sample 382, then a PRE_B false alarm | Score vs threshold, deviation hints, known label shown separately. Sample 382: short torque spike the forest barely notices. |
| 3:45–4:15 | Terminal: `/health`, valid `/predict`, `missing_channel` → 422 | Same Predictor in CLI, API and page; scores identical to 1e-16; malformed input rejected with a code. |
| 4:15–4:45 | Technical report §7–8 | Limits: one cell, threshold from 190 cycles, no root cause. Next: drift-aware calibration, phase features, 1D-CNN autoencoder. |
