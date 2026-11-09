"""Convert the Endomondo parquet files into small CSV files for the website.

Run from the repo root after scripts/download_data.py:

    python scripts/build_site_data.py
    python scripts/build_site_data.py --per-type 100   # more example workouts

Writes to site/data/:

    workouts.csv     one row per workout (no time series), ~40k rows
    users.csv        one row per athlete with aggregate stats
    timeseries.csv   full time series for a small, stratified sample of workouts
                     (long format: one row per workout per timestep)
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data" / "endomondo-hr-prediction-v2" / "dataset.parquet"
OUT = ROOT / "site" / "data"

SCALAR_COLS = [
    "workout_id", "user_id", "split", "workout_type", "duration_min", "data_points",
    "hr_mean", "hr_std", "hr_min", "hr_max", "speed_mean", "speed_max", "altitude_gain",
]


def build_workouts(df: pd.DataFrame) -> pd.DataFrame:
    w = df[SCALAR_COLS].copy()
    start = df["timestamp"].map(lambda ts: ts[0] if len(ts) else np.nan)
    w.insert(4, "start_date", pd.to_datetime(start, unit="s").dt.strftime("%Y-%m-%d"))
    w["distance_km"] = w["speed_mean"] * w["duration_min"] / 60
    return w.round(2)


def build_users(w: pd.DataFrame) -> pd.DataFrame:
    u = w.groupby("user_id").agg(
        workouts=("workout_id", "size"),
        total_km=("distance_km", "sum"),
        total_hours=("duration_min", lambda m: m.sum() / 60),
        hr_mean=("hr_mean", "mean"),
        hr_max=("hr_max", "max"),
        speed_mean=("speed_mean", "mean"),
        first_date=("start_date", "min"),
        last_date=("start_date", "max"),
    )
    return u.reset_index().sort_values("workouts", ascending=False).round(2)


def build_timeseries(df: pd.DataFrame, per_type: int, seed: int) -> pd.DataFrame:
    sample = (
        df.groupby("workout_type", group_keys=False)
        .apply(lambda g: g.sample(min(per_type, len(g)), random_state=seed))
    )
    rows = []
    for r in sample.itertuples():
        ts = np.asarray(r.timestamp, dtype="int64")
        rows.append(pd.DataFrame({
            "workout_id": r.workout_id,
            "t_min": (ts - ts[0]) / 60,
            "heart_rate": r.heart_rate,
            "speed": r.speed,
            "altitude": r.altitude,
        }))
    return pd.concat(rows, ignore_index=True).round(2)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--per-type", type=int, default=40, help="example workouts per workout type (default 40)")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    if not SRC.exists():
        raise SystemExit(f"{SRC} not found. Run: python scripts/download_data.py")

    print(f"Reading {SRC.relative_to(ROOT)} ...")
    df = pd.read_parquet(SRC)
    OUT.mkdir(parents=True, exist_ok=True)

    workouts = build_workouts(df)
    outputs = {
        "workouts.csv": workouts,
        "users.csv": build_users(workouts),
        "timeseries.csv": build_timeseries(df, args.per_type, args.seed),
    }
    for name, table in outputs.items():
        path = OUT / name
        table.to_csv(path, index=False)
        print(f"  {name:<15} {len(table):>8,} rows  {path.stat().st_size / 1e6:6.1f} MB")


if __name__ == "__main__":
    main()
