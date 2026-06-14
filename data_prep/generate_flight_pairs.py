import pandas as pd
import numpy as np
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASETS_DIR = PROJECT_ROOT / "datasets"
OUTPUT_FILE  = PROJECT_ROOT / "data_prep" / "flight_pairs.csv"

MAX_TURNAROUND_HOURS = 8

def bts_time_to_minutes(value):
    """Convert BTS HHMM time format to minutes since midnight.

    Examples
    --------
        5     -> 00:05 ->   5 minutes
        945   -> 09:45 -> 585 minutes
        1530  -> 15:30 -> 930 minutes
    """
    if pd.isna(value):
        return np.nan
    try:
        value = int(float(value))
    except (ValueError, TypeError):
        return np.nan

    hours, minutes = divmod(value, 100)

    if hours == 24:
        hours = 0
    if not (0 <= hours <= 23 and 0 <= minutes <= 59):
        return np.nan

    return hours * 60 + minutes


def build_datetime(date_col, time_col):
    """Combine a BTS date column and HHMM time column into datetimes."""
    dates   = pd.to_datetime(date_col, format="%m/%d/%Y %I:%M:%S %p", errors="coerce")
    minutes = time_col.apply(bts_time_to_minutes)
    return dates + pd.to_timedelta(minutes, unit="m")

# Loading data
print("Loading raw BTS data...")

csv_files = sorted(DATASETS_DIR.glob("*.csv"))
if not csv_files:
    raise FileNotFoundError(f"No CSV files found in {DATASETS_DIR}")

print(f"Found {len(csv_files)} monthly files")

raw_dfs = []
for f in csv_files:
    chunk = pd.read_csv(f, low_memory=False)
    chunk.columns = chunk.columns.str.strip()
    raw_dfs.append(chunk)

df = pd.concat(raw_dfs, ignore_index=True)
print(f"Total rows loaded: {len(df)}")

# Clean & Filter
print("Cleaning data...")

REQUIRED_COLS = [
    "MONTH", "DAY_OF_WEEK", "FL_DATE", "OP_UNIQUE_CARRIER",
    "TAIL_NUM", "ORIGIN", "DEST",
    "CRS_DEP_TIME", "DEP_TIME", "DEP_DELAY_NEW",
    "ARR_DELAY_NEW", "CANCELLED", "DIVERTED", "CRS_ELAPSED_TIME",
]

missing = [c for c in REQUIRED_COLS if c not in df.columns]
if missing:
    raise ValueError(f"Missing columns in raw data: {missing}")

NUMERIC_RAW = [
    "CRS_DEP_TIME", "DEP_TIME", "DEP_DELAY_NEW", "ARR_DELAY_NEW",
    "CANCELLED", "DIVERTED", "CRS_ELAPSED_TIME",
]
for col in NUMERIC_RAW:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Valid flights
df = df[
    (df["CANCELLED"] == 0)
    & (df["DIVERTED"] == 0)
    & df["TAIL_NUM"].notna()
    & df["FL_DATE"].notna()
    & df["CRS_DEP_TIME"].notna()
    & df["DEP_TIME"].notna()
    & df["DEP_DELAY_NEW"].notna()
    & df["ARR_DELAY_NEW"].notna()
    & df["CRS_ELAPSED_TIME"].notna()
].copy()

print(f"Rows after cleaning: {len(df)}")

# Datetime Columns
print("Building datetime columns...")

df["sched_dep_dt"]  = build_datetime(df["FL_DATE"], df["CRS_DEP_TIME"])
df["actual_dep_dt"] = build_datetime(df["FL_DATE"], df["DEP_TIME"])

df["sched_arr_dt"]  = df["sched_dep_dt"] + pd.to_timedelta(df["CRS_ELAPSED_TIME"], unit="m")
df["actual_arr_dt"] = df["sched_arr_dt"] + pd.to_timedelta(df["ARR_DELAY_NEW"], unit="m")

df = df.dropna(subset=["sched_dep_dt", "actual_dep_dt", "sched_arr_dt", "actual_arr_dt"])
print(f"Rows with valid datetimes: {len(df)}")

# Aircraft Rotation Pairs
print("Building flight pairs...")

df = df.sort_values(["TAIL_NUM", "sched_dep_dt"])

PAIR_COLS = [
    "MONTH", "DAY_OF_WEEK", "FL_DATE", "OP_UNIQUE_CARRIER", "TAIL_NUM",
    "ORIGIN", "DEST", "DEP_DELAY_NEW", "ARR_DELAY_NEW", "CRS_ELAPSED_TIME",
    "sched_dep_dt", "actual_dep_dt", "sched_arr_dt", "actual_arr_dt",
]
required_cols_df = df[PAIR_COLS].copy()

flight_i = required_cols_df.add_suffix("_i")
flight_j = (
    required_cols_df
    .groupby("TAIL_NUM", group_keys=False)
    .apply(lambda g: g.shift(-1))
    .add_suffix("_j")
)

pairs = pd.concat([flight_i, flight_j], axis=1)
pairs["TAIL_NUM"] = pairs["TAIL_NUM_i"]

# Drop rows where there is no subsequent flight
pairs = pairs.dropna(subset=["FL_DATE_j"])

# Valid rotation: aircraft must land where it next departs
pairs = pairs[pairs["DEST_i"] == pairs["ORIGIN_j"]].copy()

# Turnaround Calculations
pairs["scheduled_turnaround_min"] = (
    (pairs["sched_dep_dt_j"] - pairs["sched_arr_dt_i"]).dt.total_seconds() / 60
)

pairs["actual_available_turnaround_min"] = (
    (pairs["sched_dep_dt_j"] - pairs["actual_arr_dt_i"]).dt.total_seconds() / 60
)

pairs = pairs[
    (pairs["scheduled_turnaround_min"] >= 0)
    & (pairs["scheduled_turnaround_min"] <= MAX_TURNAROUND_HOURS * 60)
].copy()

print(f"Valid flight pairs: {len(pairs)}")

# Feature Engineering
print("Engineering features...")

output = pd.DataFrame({
    "TAIL_NUM": pairs["TAIL_NUM"],
    "FL_DATE":  pd.to_datetime(pairs["sched_dep_dt_i"]).dt.strftime("%Y-%m-%d"),
    "carrier":  pairs["OP_UNIQUE_CARRIER_i"],

    "turnaround_airport":              pairs["DEST_i"],
    "dest_j":                          pairs["DEST_j"],
    
    "month":                           pairs["MONTH_i"],
    "day_of_week":                     pairs["DAY_OF_WEEK_i"],
    "next_dep_hour":                   (pairs["sched_dep_dt_j"].dt.hour + pairs["sched_dep_dt_j"].dt.minute / 60).round(3),
    
    "upstream_arr_delay_min":          pairs["ARR_DELAY_NEW_i"],
    
    "scheduled_turnaround_min":        pairs["scheduled_turnaround_min"],
    "actual_available_turnaround_min": pairs["actual_available_turnaround_min"],
    
    "crs_elapsed_time_j":              pairs["CRS_ELAPSED_TIME_j"],

    "target_dep_delay_min":            pairs["DEP_DELAY_NEW_j"],
})

# Save
output.to_csv(OUTPUT_FILE, index=False)

print(f"Saved {len(output)} flight pairs to {OUTPUT_FILE}")
print(f"Mean departure delay: {output['target_dep_delay_min'].mean():.1f} min")
print("Process complete.")
