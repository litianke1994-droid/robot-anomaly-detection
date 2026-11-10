# Acceptance record v1 (2026-11-10)

Method, model effectiveness and software are judged separately (brief §8).

| ID | Deliverable | Status | Evidence |
|---|---|---|---|
| D01 | Definition and data audit | passed | `docs/project_definition.md`, `docs/data_dictionary.md`, `reports/audit_summary.json`, `reports/cycle_metadata.csv` |
| D02 | Features and split | passed | `reports/feature_spec.json` (108 columns), `reports/split_manifest.csv` (no overlap, no labels in X) |
| D03 | Three baselines and calibration | passed | `scripts/04_train_calibrate.py`, `reports/training_summary.json` (normal-only fit, separate cal set) |
| D04 | Evaluation and cases | passed (method) | `reports/evaluation.json`, figures, 6 configurations incl. ablation, 8 cases in `reports/error_analysis.json` |
| — | Model effectiveness (§8.1) | **not passed** | Primary Isolation Forest does not improve on max-\|z\| at the frozen point; test FPR 0.425 vs 5 % target. Reason analysis: technical report §5 |
| D05 | Prediction and demo | passed | model bundles, CLI, FastAPI, Streamlit replay; `reports/consistency.json` (max \|Δ\| 5.6e-17, 7/7 invalid rejected), `reports/api_run_evidence.json` |
| D06 | Reproduction and portfolio | partial | README, technical report, tests: done. Docker image: written, **not built** (`reports/docker_status.md`). 3–5 min video: **not recorded** (`docs/demo_script.md`). |

## Deviations from the brief
- Brief weeks 1–10 were executed in one implementation session on
  2026-11-10; the brief planned the learning to be done by the author over
  10 weeks. Parts marked "self-implemented" in the brief (e.g. `features.py`)
  were written with an AI assistant.
- pyarrow pinned to 24.0.0 (26.x is blocked by Windows Smart App Control on
  the dev machine; streamlit 1.67 excludes 25.0.0).
- Optional PyTorch extension (weeks 11–12): a 1D-CNN autoencoder was tried in
  the v1.1 validation search; it did not beat the feature models and is not
  served (docs/technical_report_v1_1.md §2).

# Acceptance update v1.1 (2026-11-10)

| Item | Status | Evidence |
|---|---|---|
| Model effectiveness (v1.1, exploratory) | improved over all v1 models on holdout; **exploratory** label kept | `reports/v1_1/holdout_evaluation.json`: recall 0.827, FPR 0.095, AUC 0.948; paired bootstrap FPR/AUC differences exclude 0 |
| No test/holdout leakage into choices | evidenced | commits `0f0956b` (protocol + split) → `acc5fc7` (selection frozen) → `6c266ef` (single holdout run); `scripts/12_v11_holdout.py` refuses a second run |
| Remaining failure | open | BETWEEN_B normal cycles: 56 % warnings, 32 % alarms; collisions recall 0.56–0.63 |
| Software (v1.1 served) | passed | `reports/consistency.json` (max Δ 9.4e-13, 7/7 invalid rejected), `reports/api_run_evidence.json`, 39 tests, ruff clean |
| Docker | still not built | `reports/docker_status.md` (clean-venv check added) |
| Video | still not recorded | `docs/demo_script.md` |
