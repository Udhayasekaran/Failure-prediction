import json
import joblib

import numpy as np
import pandas as pd

TIME_COL = "_time"
DEV_COL = "dev_id"
ROLL_WINDOWS = [5, 10, 30, 60]


def build_features(
    df: pd.DataFrame,
    lags=(1, 2, 5, 10, 15, 30),
    rolls=(5, 10, 30, 60),
):
    df = df.copy()
    df[TIME_COL] = pd.to_datetime(df[TIME_COL], errors="coerce")
    df = df.dropna(subset=[TIME_COL, DEV_COL]).sort_values([DEV_COL, TIME_COL]).reset_index(drop=True)

    # time features
    df["hour"] = df[TIME_COL].dt.hour.astype(float)
    df["dayofweek"] = df[TIME_COL].dt.dayofweek.astype(float)
    df["minute"] = df[TIME_COL].dt.minute.astype(float)

    non_feature_cols = {TIME_COL, DEV_COL, "failure_event", "failure_type", "y_30m"}

    # base columns: everything except non-feature
    base_cols = [c for c in df.columns if c not in non_feature_cols]

    # split numeric columns for rolling/lag
    num_cols = [c for c in base_cols if pd.api.types.is_numeric_dtype(df[c])]

    grp = df.groupby(DEV_COL, group_keys=False)

    # ---------- LAGS (build into dict -> concat once) ----------
    lag_feats = {}
    for lag in lags:
        shifted = grp[num_cols].shift(lag)
        for c in num_cols:
            lag_feats[f"{c}_lag{lag}"] = shifted[c]

    # ---------- ROLLS (build into dict -> concat once) ----------
    roll_feats = {}
    for w in rolls:
        minp = max(2, w // 3)
        rolled_mean = grp[num_cols].rolling(w, min_periods=minp).mean().reset_index(level=0, drop=True)
        rolled_std  = grp[num_cols].rolling(w, min_periods=minp).std().reset_index(level=0, drop=True)
        rolled_min  = grp[num_cols].rolling(w, min_periods=minp).min().reset_index(level=0, drop=True)
        rolled_max  = grp[num_cols].rolling(w, min_periods=minp).max().reset_index(level=0, drop=True)
        shifted_w   = grp[num_cols].shift(w)

        for c in num_cols:
            roll_feats[f"{c}_roll{w}_mean"]  = rolled_mean[c]
            roll_feats[f"{c}_roll{w}_std"]   = rolled_std[c]
            roll_feats[f"{c}_roll{w}_min"]   = rolled_min[c]
            roll_feats[f"{c}_roll{w}_max"]   = rolled_max[c]
            roll_feats[f"{c}_roll{w}_trend"] = rolled_mean[c] - shifted_w[c]

    feat_df = pd.concat([df, pd.DataFrame(lag_feats), pd.DataFrame(roll_feats)], axis=1)

    # one copy to defragment (optional but helps)
    feat_df = feat_df.copy()
    return feat_df


def anomaly_risk(model, meta, recent_window: pd.DataFrame) -> float:
    feat = build_features(recent_window)

    # Drop early rows so roll windows have enough history
    feat["_row_in_dev"] = feat.groupby(DEV_COL).cumcount()
    feat = feat[feat["_row_in_dev"] >= max(ROLL_WINDOWS)].drop(columns=["_row_in_dev"])

    if len(feat) == 0:
        return 0.0

    # Ensure all expected columns exist (avoid KeyError)
    expected = meta["feature_columns"]
    for col in expected:
        if col not in feat.columns:
            feat[col] = np.nan

    X = feat[expected].copy()
    X = X.replace([np.inf, -np.inf], np.nan)

    # robust fill: median for numeric, then 0
    med = X.median(numeric_only=True)
    X = X.fillna(med).fillna(0)

    # IsolationForest score_samples: higher = more normal -> convert to risk
    scores = model.score_samples(X)
    smax, smin = float(np.max(scores)), float(np.min(scores))
    risk = (smax - float(scores[-1])) / (smax - smin + 1e-9)
    return float(np.clip(risk, 0, 1))


def alert_level(risk: float, p95: float, p99: float):
    if risk >= p99:
        return "CRITICAL"
    if risk >= p95:
        return "WARNING"
    return "NORMAL"


def risk_bar(risk: float, p95: float, p99: float, width: int = 30) -> str:
    filled = int(round(risk * width))
    bar = "█" * filled + "░" * (width - filled)

    def mark(pos):
        i = min(width - 1, max(0, int(round(pos * width)) - 1))
        return i

    m95 = mark(p95)
    m99 = mark(p99)
    bar_list = list(bar)
    bar_list[m95] = "│"  # threshold marker
    bar_list[m99] = "┃"  # threshold marker
    return "".join(bar_list)


def compute_risk_series(model, meta, window: pd.DataFrame) -> pd.Series:
    feat = build_features(window)
    feat["_row_in_dev"] = feat.groupby(DEV_COL).cumcount()
    feat = feat[feat["_row_in_dev"] >= max(ROLL_WINDOWS)].drop(columns=["_row_in_dev"])
    if len(feat) == 0:
        return pd.Series(dtype=float)

    expected = meta["feature_columns"]
    for col in expected:
        if col not in feat.columns:
            feat[col] = np.nan

    X = feat[expected].copy().replace([np.inf, -np.inf], np.nan)
    med = X.median(numeric_only=True)
    X = X.fillna(med).fillna(0)

    scores = model.score_samples(X)
    smax, smin = float(np.max(scores)), float(np.min(scores))
    risks = (smax - scores) / (smax - smin + 1e-9)
    return pd.Series(np.clip(risks, 0, 1), index=feat.index)


def top_anomaly_drivers(window: pd.DataFrame, k: int = 8) -> pd.DataFrame:
    """
    Model-agnostic 'reason' approximation:
    Compare last row to recent median using robust z-like score.
    """
    feat = build_features(window)
    feat = feat.sort_values([DEV_COL, TIME_COL])

    out = []
    for dev, g in feat.groupby(DEV_COL):
        g = g.tail(200)  # recent context
        numeric_cols = [c for c in g.columns if pd.api.types.is_numeric_dtype(g[c]) and c not in ["hour", "minute", "dayofweek"]]
        if len(g) < 10 or not numeric_cols:
            continue

        last = g.iloc[-1][numeric_cols].astype(float)
        med = g[numeric_cols].median(numeric_only=True)
        mad = (g[numeric_cols] - med).abs().median(numeric_only=True) + 1e-9

        robust_z = ((last - med).abs() / mad).sort_values(ascending=False).head(k)

        df = pd.DataFrame({
            "feature": robust_z.index,
            "deviation": robust_z.values,
            "last": last[robust_z.index].values,
            "median": med[robust_z.index].values,
        })
        df.insert(0, DEV_COL, dev)
        out.append(df)

    if not out:
        return pd.DataFrame(columns=[DEV_COL, "feature", "deviation", "last", "median"])
    return pd.concat(out, ignore_index=True)


def print_report(window: pd.DataFrame, risk: float, level: str, p95: float, p99: float, model=None, meta=None):
    window = window.copy()
    window[TIME_COL] = pd.to_datetime(window[TIME_COL], errors="coerce")
    last_time = window[TIME_COL].max()
    devs = window[DEV_COL].nunique() if DEV_COL in window.columns else 1

    print("\n" + "=" * 72)
    print("REALTIME HEALTH REPORT")
    print("=" * 72)
    print(f"Last timestamp : {last_time}")
    print(f"Devices in window: {devs}")
    print("-" * 72)

    bar = risk_bar(risk, p95, p99, width=34)
    print(f"Risk score     : {risk:.3f}")
    print(f"Thresholds     : p95={p95:.3f} (WARNING), p99={p99:.3f} (CRITICAL)")
    print(f"Status         : {level}")
    print(f"Risk bar       : {bar}")
    print("                │=p95  ┃=p99")
    print("-" * 72)

    if model is not None and meta is not None:
        rs = compute_risk_series(model, meta, window)
        if len(rs) >= 5:
            tail = rs.tail(10).to_list()
            trend = np.polyfit(np.arange(len(tail)), tail, 1)[0]
            arrows = "↑" if trend > 0.002 else ("↓" if trend < -0.002 else "→")
            print(f"Recent risk (last {len(tail)} points): " + ", ".join(f"{v:.2f}" for v in tail))
            print(f"Trend          : {arrows} (slope={trend:+.4f} per step)")
            print("-" * 72)

    drivers = top_anomaly_drivers(window, k=8)
    if len(drivers) > 0:
        print("Top deviating signals (approx reasons):")
        for dev, g in drivers.groupby(DEV_COL):
            print(f"\nDevice: {dev}")
            for _, r in g.iterrows():
                print(f"  - {r['feature']}: deviation={r['deviation']:.2f} | last={r['last']:.3f} | median={r['median']:.3f}")
    else:
        print("Top deviating signals: not enough data to compute.")

    print("=" * 72 + "\n")


if __name__ == "__main__":
    model = joblib.load("model_artifact.joblib")
    meta = json.load(open("model_meta.json", "r"))

    window = pd.read_csv("latest_window.csv")
    risk = anomaly_risk(model, meta, window)

    p95 = meta["metrics"]["risk_p95"]
    p99 = meta["metrics"]["risk_p99"]
    level = alert_level(risk, p95, p99)

    print_report(window, risk, level, p95, p99, model=model, meta=meta)
