# 5294INOR6Y_G1

## Setup

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

## Website

The website lives in [site/](site/) and is plain HTML/CSS/JS (charts with [Observable Plot](https://observablehq.com/plot/)). It reads small CSV files from [site/data/](site/data/), which are generated from the parquet files:

```bash
python scripts/build_site_data.py                 # writes site/data/*.csv (~6 MB)
python -m http.server 8000 -d site                # preview at http://localhost:8000
```

| CSV | Contents |
|---|---|
| `workouts.csv` | one row per workout: type, date, duration, distance, HR/speed stats (40,186 rows) |
| `users.csv` | one row per athlete with totals and averages (761 rows) |
| `timeseries.csv` | full HR/speed/altitude series for 120 example workouts (40 per type), one row per timestep |

Opening `index.html` by double-clicking does not work, because browsers block loading CSVs from `file://`. Use the local server above.

Every push to `main` that changes `site/` is deployed to GitHub Pages by [.github/workflows/pages.yml](.github/workflows/pages.yml).

## Data

We use the [Endomondo Heart Rate Prediction Dataset V2](https://huggingface.co/datasets/rricc22/endomondo-hr-prediction-v2): 40,186 running workouts from 761 athletes, with heart rate, speed and altitude time series. See [data/endomondo-hr-prediction-v2/README.md](data/endomondo-hr-prediction-v2/README.md) for the full dataset card.

The data files are **not in this repo** because they are too large for git (~680 MB). There are two ways to get them.

### Option 1: download the data (recommended for training)

Run this once from the repo root:

```bash
python scripts/download_data.py            # all files (~680 MB)
python scripts/download_data.py --splits   # only train/validation/test (~340 MB)
```

This saves the files to `data/endomondo-hr-prediction-v2/`:

| File | Contents |
|---|---|
| `train.parquet` | 28,130 workouts (70%) |
| `validation.parquet` | 6,027 workouts (15%) |
| `test.parquet` | 6,029 workouts (15%) |
| `dataset.parquet` | all three splits combined, with a `split` column |

You can safely re-run the script; files that are already complete are skipped. Load the files with pandas:

```python
import pandas as pd

train = pd.read_parquet("data/endomondo-hr-prediction-v2/train.parquet")
```

### Option 2: stream the data without downloading

[examples/stream_dataset.py](examples/stream_dataset.py) reads the dataset directly from Hugging Face, fetching workouts only as you loop over them. Nothing is saved to disk. This is handy for quick exploration; for repeated training runs, Option 1 is faster.

```bash
python examples/stream_dataset.py
```
