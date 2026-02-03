"""
model.py (v4) - AUTO MODE

If failure_event exists:
  -> Supervised model predicts failure in next 30 minutes (classification)

If failure_event does NOT exist (your current case):
  -> Unsupervised anomaly model learns "normal" behavior and outputs anomaly risk.
     For the 30-min requirement, we compute risk from the latest rolling window,
     and you can alert when risk exceeds threshold.

FILES expected in same folder:
  dg.csv
  humidity.csv
  temperature.csv

OUTPUTS:
  model_artifact.joblib
  model_meta.json
"""

from __future__ import annotations

import os, re, json, warnings, joblib
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from sklearn.ensemble import HistGradientBoostingClassifier, IsolationForest
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report, confusion_matrix

warnings.filterwarnings("ignore")

# =========================
# Config
# =========================

DG_PATH = "dg.csv"
HUM_PATH = "humidity.csv"
TMP_PATH = "temperature.csv"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_OUT = os.path.join(BASE_DIR, "model_artifact.joblib")
META_OUT  = os.path.join(BASE_DIR, "model_meta.json")

TIME_COL = "_time"
DEV_COL  = "dev_id"

RESAMPLE_FREQ = "1min"
ASOF_TOL = pd.Timedelta("2min")

HORIZON_MINUTES = 30
ROLL_WINDOWS = [5, 10, 30, 60]
LAGS = [1, 2, 5, 10, 15, 30]

# leakage control
LEAKY_BASE = ["DG_st", "IO_diag", "No_of_trips", "No_of_starts", "gw_pwr_st", "ts"]
LEAKY_KEYWORDS = ["alarm", "fault", "trip", "shutdown", "breaker", "diag", "status", "dg_st", "io_diag"]

DROP_GPS = True

# anomaly settings
ANOM_CONTAMINATION = 0.02  # expected fraction of anomalies (tune later)
ANOM_THRESHOLD = 0.6       # 0..1 risk threshold for alert (tune later)

NON_FEATURE_COLS = [TIME_COL, DEV_COL, "failure_event", "failure_type", "y_30m"]


# =========================
# Helpers
# =========================

def to_datetime_safe(s: pd.Series, dayfirst: bool = True) -> pd.Series:
    return pd.to_datetime(s, dayfirst=dayfirst, errors="coerce")

def parse_number_from_text(x: str) -> float:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return np.nan
    s = str(x).strip()
    m = re.search(r"[-+]?\d*\.?\d+", s)
    return float(m.group(0)) if m else np.nan

def numericize_df(df: pd.DataFrame, exclude: List[str]) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        if c in exclude:
            continue
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out

def pick_existing_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


# =========================
# Loaders
# =========================

def load_dg(path: str) -> pd.DataFrame:
    dg = pd.read_csv(path, encoding="utf-8-sig")
    dg = dg.dropna(subset=[DEV_COL]).copy()
    dg[TIME_COL] = to_datetime_safe(dg[TIME_COL], dayfirst=True)
    dg = dg.dropna(subset=[TIME_COL])
    dg = dg.sort_values([DEV_COL, TIME_COL]).reset_index(drop=True)
    dg = numericize_df(dg, exclude=[TIME_COL, DEV_COL])
    return dg

def load_humidity(path: str) -> pd.DataFrame:
    h = pd.read_csv(path)
    time_col = pick_existing_col(h, ["Time", "time", "timestamp", "Timestamp"])
    if time_col is None:
        raise ValueError("humidity.csv must contain a Time column.")
    val_col = [c for c in h.columns if c != time_col][0]
    h = h[[time_col, val_col]].copy()
    h.rename(columns={time_col: "time", val_col: "humidity"}, inplace=True)
    h["time"] = to_datetime_safe(h["time"], dayfirst=True)
    h["humidity"] = h["humidity"].apply(parse_number_from_text)
    h = h.dropna(subset=["time"]).sort_values("time").reset_index(drop=True)
    return h

def load_temperature(path: str) -> pd.DataFrame:
    t = pd.read_csv(path)
    time_col = pick_existing_col(t, ["Time", "time", "timestamp", "Timestamp"])
    if time_col is None:
        raise ValueError("temperature.csv must contain a Time column.")
    val_col = [c for c in t.columns if c != time_col][0]
    t = t[[time_col, val_col]].copy()
    t.rename(columns={time_col: "time", val_col: "temperature"}, inplace=True)
    t["time"] = to_datetime_safe(t["time"], dayfirst=True)
    t["temperature"] = t["temperature"].apply(parse_number_from_text)
    t = t.dropna(subset=["time"]).sort_values("time").reset_index(drop=True)
    return t


# =========================
# Gold table
# =========================

def resample_per_device(dg: pd.DataFrame, freq: str = RESAMPLE_FREQ) -> pd.DataFrame:
    def _resample(df: pd.DataFrame) -> pd.DataFrame:
        df = df.set_index(TIME_COL)
        out = df.resample(freq).mean(numeric_only=True)
        out[DEV_COL] = df[DEV_COL].iloc[0]
        return out.reset_index()

    gold = dg.groupby(DEV_COL, group_keys=False).apply(_resample)
    gold = gold.sort_values([DEV_COL, TIME_COL]).reset_index(drop=True)
    return gold

def join_exogenous(gold: pd.DataFrame, h: pd.DataFrame, t: pd.DataFrame) -> pd.DataFrame:
    g = gold.sort_values(TIME_COL).copy()
    h2 = h.sort_values("time").copy()
    t2 = t.sort_values("time").copy()

    g = pd.merge_asof(g, h2, left_on=TIME_COL, right_on="time", direction="nearest", tolerance=ASOF_TOL).drop(columns=["time"])
    g = pd.merge_asof(g, t2, left_on=TIME_COL, right_on="time", direction="nearest", tolerance=ASOF_TOL).drop(columns=["time"])
    return g


# =========================
# Predictive Labeling (if possible)
# =========================

@dataclass
class FailureLabelConfig:
    dg_status_col: str = "DG_st"
    running_value: float = 1.0
    treat_any_non_running_as_failure: bool = True
    explicit_failure_values: Optional[List[float]] = None
    min_off_minutes: int = 2

def label_failure_start_from_dg_st(df: pd.DataFrame, cfg: FailureLabelConfig) -> pd.DataFrame:
    out = df.sort_values([DEV_COL, TIME_COL]).copy()
    if cfg.dg_status_col not in out.columns:
        # No status -> cannot label
        out["failure_event"] = 0
        out["failure_type"] = ""
        return out

    st = out[cfg.dg_status_col]
    if cfg.explicit_failure_values is not None:
        is_fail_state = st.isin(cfg.explicit_failure_values)
    elif cfg.treat_any_non_running_as_failure:
        is_fail_state = (st != cfg.running_value)
    else:
        is_fail_state = (st == 0)

    out["_is_fail_state"] = is_fail_state.astype(int)

    def _confirm(df_dev: pd.DataFrame) -> pd.DataFrame:
        x = df_dev["_is_fail_state"].to_numpy()
        confirmed = np.zeros_like(x)
        win = max(1, cfg.min_off_minutes)
        for i in range(len(x)):
            j0 = max(0, i - win + 1)
            if (i - j0 + 1) >= win and x[j0:i+1].sum() == (i - j0 + 1):
                confirmed[i] = 1
        prev = np.r_[0, confirmed[:-1]]
        df_dev["failure_event"] = ((confirmed == 1) & (prev == 0)).astype(int)
        df_dev["failure_type"] = np.where(df_dev["failure_event"] == 1, "dg_stop_or_fault", "")
        return df_dev

    out = out.groupby(DEV_COL, group_keys=False).apply(_confirm)
    out = out.drop(columns=["_is_fail_state"])
    return out

def create_target_30m(df: pd.DataFrame, horizon_minutes: int = HORIZON_MINUTES) -> pd.DataFrame:
    out = df.sort_values([DEV_COL, TIME_COL]).copy()
    steps = horizon_minutes

    def _make(df_dev: pd.DataFrame) -> pd.DataFrame:
        x = df_dev["failure_event"].fillna(0).astype(int).to_numpy()
        y = np.zeros_like(x)
        for i in range(len(x)):
            j_end = min(len(x), i + steps + 1)
            if i + 1 < j_end and x[i+1:j_end].max() == 1:
                y[i] = 1
        df_dev["y_30m"] = y
        return df_dev

    return out.groupby(DEV_COL, group_keys=False).apply(_make)


# =========================
# Features
# =========================

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    ts = out[TIME_COL]
    out["hour"] = ts.dt.hour
    out["dayofweek"] = ts.dt.dayofweek
    out["minute"] = ts.dt.minute
    return out

def add_lags(df: pd.DataFrame) -> pd.DataFrame:
    out = df.sort_values([DEV_COL, TIME_COL]).copy()
    base_cols = [c for c in out.columns if c not in NON_FEATURE_COLS and c not in ["hour","dayofweek","minute"]]
    for lag in LAGS:
        for c in base_cols:
            out[f"{c}_lag{lag}"] = out.groupby(DEV_COL)[c].shift(lag)
    return out

def add_rolls(df: pd.DataFrame) -> pd.DataFrame:
    out = df.sort_values([DEV_COL, TIME_COL]).copy()
    base_cols = [c for c in out.columns if c not in NON_FEATURE_COLS and not any(c.endswith(f"_lag{l}") for l in LAGS)]
    num_cols = [c for c in base_cols if pd.api.types.is_numeric_dtype(out[c])]
    for w in ROLL_WINDOWS:
        grp = out.groupby(DEV_COL, group_keys=False)
        for c in num_cols:
            out[f"{c}_roll{w}_mean"] = grp[c].rolling(w, min_periods=max(2, w//3)).mean().reset_index(level=0, drop=True)
            out[f"{c}_roll{w}_std"]  = grp[c].rolling(w, min_periods=max(2, w//3)).std().reset_index(level=0, drop=True)
            out[f"{c}_roll{w}_min"]  = grp[c].rolling(w, min_periods=max(2, w//3)).min().reset_index(level=0, drop=True)
            out[f"{c}_roll{w}_max"]  = grp[c].rolling(w, min_periods=max(2, w//3)).max().reset_index(level=0, drop=True)
            out[f"{c}_roll{w}_trend"] = out[f"{c}_roll{w}_mean"] - grp[c].shift(w).reset_index(level=0, drop=True)
    return out

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    out = add_time_features(df)
    out = add_lags(out)
    out = add_rolls(out)
    return out


# =========================
# Leakage removal
# =========================

def keyword_suspicious(col: str) -> bool:
    lc = col.lower()
    return any(k in lc for k in LEAKY_KEYWORDS)

def drop_leaky_columns(X: pd.DataFrame) -> pd.DataFrame:
    patterns = []
    for c in LEAKY_BASE:
        patterns += [
            rf"^{re.escape(c)}$",
            rf"^{re.escape(c)}_lag\d+$",
            rf"^{re.escape(c)}_roll\d+_.*$",
        ]

    drop_cols = set()
    for col in X.columns:
        if any(re.match(p, col) for p in patterns):
            drop_cols.add(col)
        elif keyword_suspicious(col):
            drop_cols.add(col)

    if DROP_GPS:
        for gps in ["lat","long"]:
            for col in list(X.columns):
                if col == gps or col.startswith(f"{gps}_lag") or col.startswith(f"{gps}_roll"):
                    drop_cols.add(col)

    return X.drop(columns=list(drop_cols), errors="ignore")


def make_X(df: pd.DataFrame) -> pd.DataFrame:
    X = df.drop(columns=[c for c in NON_FEATURE_COLS if c in df.columns], errors="ignore").copy()
    # numeric only
    for c in list(X.columns):
        if not pd.api.types.is_numeric_dtype(X[c]):
            X = X.drop(columns=[c])
    X = drop_leaky_columns(X)
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median(numeric_only=True))
    X = X.fillna(0)
    return X


# =========================
# Training: supervised OR anomaly
# =========================

def train_supervised(feat: pd.DataFrame) -> Tuple[object, Dict, List[str]]:
    feat = feat.sort_values(TIME_COL).reset_index(drop=True)

    X = make_X(feat)
    y = feat["y_30m"].astype(int)

    # Simple time holdout: last 20% (but should now include positives if labels exist)
    cut = int(0.8 * len(feat))
    X_tr, X_te = X.iloc[:cut], X.iloc[cut:]
    y_tr, y_te = y.iloc[:cut], y.iloc[cut:]

    model = HistGradientBoostingClassifier(max_depth=6, learning_rate=0.05, max_iter=400, random_state=42)
    model.fit(X_tr, y_tr)

    proba = model.predict_proba(X_te)[:, 1]
    pred = (proba >= 0.5).astype(int)

    metrics = {
        "mode": "supervised_failure_prediction",
        "test_size": int(len(y_te)),
        "test_pos": int(y_te.sum()),
        "test_neg": int((y_te == 0).sum()),
    }

    if len(np.unique(y_te)) > 1:
        metrics["roc_auc"] = float(roc_auc_score(y_te, proba))
        metrics["avg_precision"] = float(average_precision_score(y_te, proba))
    else:
        metrics["roc_auc"] = None
        metrics["avg_precision"] = None

    print("\n=== SUPERVISED EVAL (last 20%) ===")
    print(metrics)
    print("Confusion Matrix:\n", confusion_matrix(y_te, pred))
    print(classification_report(y_te, pred, digits=4))

    # final fit on all
    final_model = HistGradientBoostingClassifier(max_depth=6, learning_rate=0.05, max_iter=400, random_state=42)
    final_model.fit(X, y)

    return final_model, metrics, list(X.columns)


def train_anomaly(feat: pd.DataFrame) -> Tuple[object, Dict, List[str]]:
    """
    Unsupervised fallback when you have NO failures.
    Uses IsolationForest to learn normal behavior.
    Produces a risk score in [0,1] where higher = more anomalous.
    """
    feat = feat.sort_values(TIME_COL).reset_index(drop=True)
    X = make_X(feat)

    model = IsolationForest(
        n_estimators=400,
        contamination=ANOM_CONTAMINATION,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X)

    # score_samples: higher means more normal; convert to anomaly risk
    s = model.score_samples(X)
    # normalize to 0..1 risk
    risk = (s.max() - s) / (s.max() - s.min() + 1e-9)

    metrics = {
        "mode": "unsupervised_anomaly_detection",
        "train_rows": int(len(X)),
        "contamination": float(ANOM_CONTAMINATION),
        "suggested_alert_threshold": float(ANOM_THRESHOLD),
        "risk_p95": float(np.quantile(risk, 0.95)),
        "risk_p99": float(np.quantile(risk, 0.99)),
    }

    print("\n=== ANOMALY TRAINED (no failures in data) ===")
    print("Risk p95:", metrics["risk_p95"], "Risk p99:", metrics["risk_p99"])
    print("Suggested threshold:", ANOM_THRESHOLD)

    return model, metrics, list(X.columns)


def save_artifacts(model: object, feature_cols: List[str], label_cfg: Dict, metrics: Dict):
    joblib.dump(model, MODEL_OUT)
    meta = {
        "feature_columns": feature_cols,
        "horizon_minutes": HORIZON_MINUTES,
        "resample_freq": RESAMPLE_FREQ,
        "roll_windows": ROLL_WINDOWS,
        "lags": LAGS,
        "labeling": label_cfg,
        "metrics": metrics,
        "leakage_dropped_base": LEAKY_BASE,
        "drop_gps": DROP_GPS,
        "artifact_file": os.path.basename(MODEL_OUT),
    }
    with open(META_OUT, "w") as f:
        json.dump(meta, f, indent=2)
    print("\nSaved:", MODEL_OUT)
    print("Saved:", META_OUT)


# =========================
# Realtime scoring helpers
# =========================

def load_artifacts():
    model = joblib.load(MODEL_OUT)
    with open(META_OUT, "r") as f:
        meta = json.load(f)
    return model, meta

def anomaly_risk_from_window(model: IsolationForest, meta: Dict, recent_window: pd.DataFrame) -> float:
    """
    Input: recent_window dataframe already containing same raw cols as dg + humidity/temp merged.
    Output: anomaly risk in [0,1]
    """
    recent_window = recent_window.copy()
    recent_window[TIME_COL] = pd.to_datetime(recent_window[TIME_COL], errors="coerce")
    recent_window = recent_window.sort_values([DEV_COL, TIME_COL])

    feat = build_features(recent_window)
    feat["_row_in_dev"] = feat.groupby(DEV_COL).cumcount()
    feat = feat[feat["_row_in_dev"] >= max(ROLL_WINDOWS)].drop(columns=["_row_in_dev"])

    X = feat[meta["feature_columns"]].copy()
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median(numeric_only=True)).fillna(0)

    if len(X) == 0:
        return 0.0

    s = model.score_samples(X)
    risk = (s.max() - s[-1]) / (s.max() - s.min() + 1e-9)
    return float(np.clip(risk, 0, 1))


# =========================
# Main
# =========================

def main():
    dg = load_dg(DG_PATH)
    h  = load_humidity(HUM_PATH)
    t  = load_temperature(TMP_PATH)

    gold = resample_per_device(dg, RESAMPLE_FREQ)
    gold = join_exogenous(gold, h, t)

    label_cfg = FailureLabelConfig(
        dg_status_col="DG_st",
        running_value=1.0,
        treat_any_non_running_as_failure=True,
        explicit_failure_values=None,
        min_off_minutes=2,
    )

    labeled = label_failure_start_from_dg_st(gold, label_cfg)
    labeled = create_target_30m(labeled, HORIZON_MINUTES)

    feat = build_features(labeled)
    feat["_row_in_dev"] = feat.groupby(DEV_COL).cumcount()
    feat = feat[feat["_row_in_dev"] >= max(ROLL_WINDOWS)].drop(columns=["_row_in_dev"])

    print("\nOverall y_30m counts:")
    print(feat["y_30m"].value_counts(dropna=False))
    print("\nFailure_event count:", int(feat["failure_event"].sum()))

    if int(feat["failure_event"].sum()) > 0 and int(feat["y_30m"].sum()) > 0:
        model, metrics, feature_cols = train_supervised(feat)
    else:
        # Your current case: no failures -> anomaly fallback
        model, metrics, feature_cols = train_anomaly(feat)

    save_artifacts(model, feature_cols, label_cfg.__dict__, metrics)
    print("\nDone.")

if __name__ == "__main__":
    main()
