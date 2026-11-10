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
- Optional PyTorch extension (weeks 11–12): not started.
