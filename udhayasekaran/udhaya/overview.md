# DG Failure / Anomaly Detection — Quick Overview

## What this project does
- Trains a model from historical DG telemetry + humidity + temperature.
- **Auto mode**
  - If failures are detected (`failure_event` exists): trains a **supervised classifier** to predict failure in **next 30 minutes**.
  - If failures are NOT present (current case): trains an **unsupervised anomaly model (IsolationForest)** to learn “normal” behavior and outputs **risk score (0..1)**.

---

## Inputs (place in same folder)
- `dg.csv`
  - Must contain: `_time`, `dev_id`, and numeric sensor columns
  - Optional: `DG_st` (used to infer failure starts)
- `humidity.csv`
  - Must contain a time column: `Time` / `time` / `timestamp` / `Timestamp`
  - One value column (any name)
- `temperature.csv`
  - Same as humidity format

---

## Outputs (generated)
- `model_artifact.joblib` — trained model
- `model_meta.json` — metadata (feature list, thresholds, mode, params)
- (optional for testing) `latest_window.csv` — recent window extracted from old data

---

## How it works (high level pipeline)
1. Load CSVs and parse timestamps
2. Resample DG telemetry per device to **1-minute**
3. Join humidity + temperature using nearest timestamp (`merge_asof`, 2-min tolerance)
4. (If possible) detect `failure_event` from `DG_st`
5. Create `y_30m` target (failure within next 30 mins)
6. Create features:
   - time: hour/dayofweek/minute
   - lags: 1,2,5,10,15,30
   - rolling windows: 5,10,30,60 (mean/std/min/max/trend)
7. Drop “leaky” columns (status/alarm/trip/diag etc.)
8. Train:
   - supervised classifier **OR**
   - anomaly model (IsolationForest)
9. Save model + meta

---

## Quick Run
### 1) Install
```bash
pip install -r requirements.txt
