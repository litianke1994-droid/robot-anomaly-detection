# Data dictionary (columns used in v1)

Source: voraus-AD 100 Hz Parquet, 2 321 690 rows × 137 columns. Only the
columns below are read. Measured facts come from `reports/audit_summary.json`.

## Meta columns (never model features)
| Column | Type | Meaning | Use in v1 |
|---|---|---|---|
| `sample` | int64 | Cycle id, 0–2121, stored contiguously | Grouping key; split unit |
| `time` | double | Seconds since cycle start (starts at 0–0.010 s) | Ordering and sampling checks only |
| `anomaly` | bool | True for anomalous cycles; constant within a cycle | Evaluation label |
| `category` | int64 | 0–11 anomaly types, 12 = normal operation | Stratified evaluation only |
| `setting` | int64 | Experimental setting; the official loader renames it `variant`. 72 = `PRE_A` | Official train/test boundary |
| `action`, `active` | uint8 / int64 | Motion phase / activity flag | Not read; excluded from features |

## Signal whitelist (18 channels, axis 1–6)
| Group | Meaning | Unit* | Observed range (all data) |
|---|---|---|---|
| `motor_iq_1..6` | Motor q-axis current | A | about −7 to 9.5 |
| `joint_velocity_1..6` | Joint angular velocity | rad/s | about −2.3 to 2.3 |
| `torque_sensor_a_1..6` | Joint torque sensor A | N·m | about −53 to 76 |

\*Units follow the paper's signal table and the field names; the Parquet file
carries no unit metadata. The observed magnitudes are physically plausible
for these units (待核实 against the paper table).

## Measured audit facts
- 2122 cycles; labels, category and setting are constant inside every cycle.
- Official split (setting == PRE_A for training): 948 train normal; 419 test
  normal + 755 test anomalous — identical to the paper.
- Test normal settings: PRE_B 319, BETWEEN_A 37, BETWEEN_B 44, BETWEEN_C 19.
- Test anomalies per category: AXIS_FRICTION 144, AXIS_WEIGHT 156,
  COLLISION_FOAM 72, COLLISION_CABLE 48, COLLISION_CARTON 22, MISS_CAN 11,
  LOSE_CAN 74, CAN_WEIGHT 80, ENTANGLED 10, INVALID_POSITION 12,
  MOTOR_COMMUTATION 89, WOBBLING_STATION 37.
- Cycle length 986–1164 points (train normal 1028–1164, median 1083).
- Sampling interval 8.45–12.11 ms, median 10.07 ms; no non-increasing steps.
- No NaN/inf in the 18 channels, no duplicate (sample, time) rows, no
  duplicate cycles, no channel constant within any cycle.
