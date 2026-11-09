"""Use the Endomondo HR dataset without downloading it.

The dataset is streamed straight from Hugging Face: workouts are fetched as you
iterate over them, so nothing is saved to disk.

Usage (from the repo root):
    python examples/stream_dataset.py
"""

from itertools import islice

import numpy as np
from datasets import load_dataset

REPO_ID = "rricc22/endomondo-hr-prediction-v2"

# Name the split files explicitly so the combined dataset.parquet is not mixed in.
DATA_FILES = {
    "train": "train.parquet",
    "validation": "validation.parquet",
    "test": "test.parquet",
}


def main() -> None:
    # streaming=True gives an IterableDataset: rows are fetched as you loop over them.
    dataset = load_dataset(REPO_ID, data_files=DATA_FILES, streaming=True)
    train = dataset["train"]

    # --- Look at one workout ---------------------------------------------
    workout = next(iter(train))
    print("First training workout")
    print(f"  workout_id:   {workout['workout_id']}")
    print(f"  user_id:      {workout['user_id']}")
    print(f"  type:         {workout['workout_type']}")
    print(f"  duration:     {workout['duration_min']:.1f} min")
    print(f"  data points:  {workout['data_points']}")
    print(f"  avg HR:       {workout['hr_mean']:.1f} BPM")
    print(f"  avg speed:    {workout['speed_mean']:.1f} km/h")

    # The time-series columns are plain Python lists; turn them into arrays.
    hr = np.asarray(workout["heart_rate"])
    speed = np.asarray(workout["speed"])
    altitude = np.asarray(workout["altitude"])
    print(f"  HR series:    {hr[:5].round(1)} ... ({len(hr)} values)")
    print(f"  speed series: {speed[:5].round(1)} ...")
    print(f"  altitude:     {altitude[:5].round(1)} ...")

    # --- Filter and take a small sample --------------------------------------
    # Operations on a streamed dataset are lazy; nothing is fetched until you iterate.
    intensive = train.filter(lambda w: w["workout_type"] == "INTENSIVE")
    sample = list(islice(intensive, 100))
    mean_hr = np.mean([w["hr_mean"] for w in sample])
    print(f"\nMean HR over the first {len(sample)} INTENSIVE workouts: {mean_hr:.1f} BPM")

    # --- Load a small sample as a pandas DataFrame ---------------------------
    df = (
        dataset["validation"]
        .select_columns(["workout_id", "workout_type", "duration_min", "hr_mean", "speed_mean"])
        .take(1000)
        .to_pandas()
    )
    print("\nFirst 1000 validation workouts, averaged per type:")
    print(df.groupby("workout_type")[["duration_min", "hr_mean", "speed_mean"]].mean().round(1))


if __name__ == "__main__":
    main()
