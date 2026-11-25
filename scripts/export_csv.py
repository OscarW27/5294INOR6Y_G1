"""Convert the Endomondo parquet dataset into two plain CSV files.

Run from the repo root after scripts/download_data.py:

    python scripts/export_csv.py

Writes to data/csv/:

    workouts.csv       one row per workout (all scalar columns), 40,186 rows
    measurements.csv   one row per workout per timestep (long format), ~20M rows
                       columns: workout_id, step, timestamp, t_sec, heart_rate, speed, altitude

Join the two files on workout_id.
"""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data" / "endomondo-hr-prediction-v2" / "dataset.parquet"
OUT = ROOT / "data" / "csv"

SERIES_COLS = ["heart_rate", "speed", "altitude", "timestamp"]


def main() -> None:
    if not SRC.exists():
        raise SystemExit(f"{SRC} not found. Run scripts/download_data.py first.")
    OUT.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(SRC)

    # workouts.csv: everything except the list columns, plus a readable start time
    workouts = df.drop(columns=SERIES_COLS)
    workouts.insert(
        2, "start_time",
        pd.to_datetime(df["timestamp"].str[0], unit="s").dt.strftime("%Y-%m-%d %H:%M:%S"),
    )
    workouts.round(3).to_csv(OUT / "workouts.csv", index=False)
    print(f"workouts.csv      {len(workouts):>11,} rows")

    # measurements.csv: explode the list columns into one row per timestep
    lengths = df["timestamp"].str.len().to_numpy()
    ts = np.concatenate(df["timestamp"].to_numpy())
    starts = np.repeat(df["timestamp"].str[0].to_numpy(), lengths)
    m = pd.DataFrame({
        "workout_id": np.repeat(df["workout_id"].to_numpy(), lengths),
        "step": np.concatenate([np.arange(n) for n in lengths]),
        "timestamp": ts,
        "t_sec": ts - starts,
        "heart_rate": np.concatenate(df["heart_rate"].to_numpy()).round(2),
        "speed": np.concatenate(df["speed"].to_numpy()).round(3),
        "altitude": np.concatenate(df["altitude"].to_numpy()).round(2),
    })
    m.to_csv(OUT / "measurements.csv", index=False, chunksize=1_000_000)
    print(f"measurements.csv  {len(m):>11,} rows")

    for f in sorted(OUT.glob("*.csv")):
        print(f"  {f.name:<18} {f.stat().st_size / 1e6:8.1f} MB")


if __name__ == "__main__":
    main()
