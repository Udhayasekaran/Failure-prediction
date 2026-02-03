import re
import numpy as np
import pandas as pd

TIME_COL = "_time"
DEV_COL  = "dev_id"
ASOF_TOL = pd.Timedelta("2min")

def to_dt(s):
    return pd.to_datetime(s, dayfirst=True, errors="coerce")

def parse_number(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return np.nan
    m = re.search(r"[-+]?\d*\.?\d+", str(x))
    return float(m.group(0)) if m else np.nan

def pick_existing_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None

def build_gold(dg_path="dg.csv", hum_path="humidity.csv", tmp_path="temperature.csv"):
    # DG
    dg = pd.read_csv(dg_path, encoding="utf-8-sig")
    dg[TIME_COL] = to_dt(dg[TIME_COL])
    dg = dg.dropna(subset=[TIME_COL, DEV_COL]).sort_values([DEV_COL, TIME_COL]).reset_index(drop=True)

    # resample each device to 1-min (numeric means)
    def _res(df):
        df = df.set_index(TIME_COL)
        out = df.resample("1min").mean(numeric_only=True)
        out[DEV_COL] = df[DEV_COL].iloc[0]
        return out.reset_index()

    gold = dg.groupby(DEV_COL, group_keys=False).apply(_res)
    gold = gold.sort_values([DEV_COL, TIME_COL]).reset_index(drop=True)

    # humidity
    h = pd.read_csv(hum_path)
    h_time = pick_existing_col(h, ["Time", "time", "timestamp", "Timestamp"])
    if h_time is None:
        raise ValueError("humidity.csv must have a time column (Time/time/timestamp).")
    h_val = [c for c in h.columns if c != h_time][0]
    h = h[[h_time, h_val]].copy()
    h.columns = ["time", "humidity"]
    h["time"] = to_dt(h["time"])
    h["humidity"] = h["humidity"].apply(parse_number)
    h = h.dropna(subset=["time"]).sort_values("time").reset_index(drop=True)

    # temperature
    t = pd.read_csv(tmp_path)
    t_time = pick_existing_col(t, ["Time", "time", "timestamp", "Timestamp"])
    if t_time is None:
        raise ValueError("temperature.csv must have a time column (Time/time/timestamp).")
    t_val = [c for c in t.columns if c != t_time][0]
    t = t[[t_time, t_val]].copy()
    t.columns = ["time", "temperature"]
    t["time"] = to_dt(t["time"])
    t["temperature"] = t["temperature"].apply(parse_number)
    t = t.dropna(subset=["time"]).sort_values("time").reset_index(drop=True)

    # merge_asof nearest
    gold = pd.merge_asof(
        gold.sort_values(TIME_COL),
        h.sort_values("time"),
        left_on=TIME_COL, right_on="time",
        direction="nearest",
        tolerance=ASOF_TOL,
    ).drop(columns=["time"])

    gold = pd.merge_asof(
        gold.sort_values(TIME_COL),
        t.sort_values("time"),
        left_on=TIME_COL, right_on="time",
        direction="nearest",
        tolerance=ASOF_TOL,
    ).drop(columns=["time"])

    return gold.sort_values([DEV_COL, TIME_COL]).reset_index(drop=True)

if __name__ == "__main__":
    WINDOW_MINUTES = 180  # use 120–240 (must be >= 80 for your roll windows)
    out_path = "latest_window.csv"

    gold = build_gold()

    # pick a device automatically (first one)
    dev = gold[DEV_COL].dropna().unique()[0]
    d = gold[gold[DEV_COL] == dev].sort_values(TIME_COL).reset_index(drop=True)

    # take last WINDOW_MINUTES rows
    start = max(0, len(d) - WINDOW_MINUTES)
    latest_window = d.iloc[start:].copy()

    latest_window.to_csv(out_path, index=False)
    print("✅ Created", out_path)
    print("Device:", dev, "Rows:", len(latest_window))
    print("Time range:", latest_window[TIME_COL].min(), "->", latest_window[TIME_COL].max())
