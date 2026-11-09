---
license: mit
task_categories:
- time-series-forecasting
tags:
- heart-rate
- running
- physiological-modeling
- lstm
- endomondo
size_categories:
- 10K<n<100K
---

# Endomondo Heart Rate Prediction Dataset V2

## Dataset Summary

This dataset contains **40,186 running workouts** from **761** athletes, designed for heart rate prediction from speed and altitude time-series.

Each workout includes:
- **Time-series**: Heart rate (target), speed, altitude, timestamps
- **Metadata**: Workout type, duration, user ID
- **Statistics**: Pre-computed HR/speed metrics for filtering

## Dataset Structure

### Splits

| Split | Workouts | Description |
|-------|----------|-------------|
| Train | 28,130 | Training set (70%) |
| Validation | 6,027 | Validation set (15%) |
| Test | 6,029 | Test set (15%) |

### Features

| Feature | Type | Description |
|---------|------|-------------|
| `workout_id` | int | Unique workout identifier |
| `user_id` | int | Anonymous user identifier |
| `workout_type` | string | RECOVERY, STEADY, or INTENSIVE |
| `duration_min` | float | Workout duration in minutes |
| `data_points` | int | Number of timesteps (max 500) |
| `heart_rate` | list[float] | Heart rate time-series [BPM] |
| `speed` | list[float] | Speed time-series [km/h] |
| `altitude` | list[float] | Altitude time-series [meters] |
| `timestamp` | list[float] | Unix timestamps [seconds] |
| `hr_mean` | float | Average heart rate [BPM] |
| `hr_std` | float | HR standard deviation |
| `hr_min` | float | Minimum HR [BPM] |
| `hr_max` | float | Maximum HR [BPM] |
| `speed_mean` | float | Average speed [km/h] |
| `speed_max` | float | Maximum speed [km/h] |
| `altitude_gain` | float | Cumulative elevation gain [m] |
| `split` | string | train / validation / test |

### Workout Type Distribution

| Type | Count | Description |
|------|-------|-------------|
| RECOVERY | 15,095 | Easy runs (low intensity) |
| STEADY | 22,991 | Moderate pace runs |
| INTENSIVE | 2,100 | High intensity workouts |

## Data Quality

All workouts have been:
1. **Filtered** for quality (removed HR anomalies, corrupted data)
2. **Smoothed** with 7-point moving average (reduces GPS noise)
3. **Validated** against physiological constraints:
   - HR mean ≥ 120 BPM
   - HR max ≤ 200 BPM
   - HR std ≥ 5 BPM
   - Speed-HR correlation ≥ -0.3

Removed: 6,064 low-quality workouts

## Usage Example

```python
from datasets import load_dataset

# Load full dataset
dataset = load_dataset("rricc22/endomondo-hr-prediction-v2")

# Access splits
train_data = dataset['train']
test_data = dataset['test']

# Example workout
workout = train_data[0]
print(f"Workout Type: {workout['workout_type']}")
print(f"Duration: {workout['duration_min']:.1f} min")
print(f"Avg HR: {workout['hr_mean']:.1f} BPM")
print(f"Avg Speed: {workout['speed_mean']:.1f} km/h")

# Access time-series
heart_rate = workout['heart_rate']  # List of HR values
speed = workout['speed']            # List of speed values
```

## Model Performance

This dataset was used to train an LSTM model achieving:
- **7.42 BPM** Mean Absolute Error
- **17% improvement** over baseline

See the model card: [rricc22/heart-rate-prediction-lstm](https://huggingface.co/rricc22/heart-rate-prediction-lstm)

Try the demo: [Heart Rate Predictor](https://huggingface.co/spaces/rricc22/heart-rate-predictor)

## Source

- **Original Data**: Endomondo dataset
- **Processing Pipeline**: Quality filtering → Smoothing → Feature engineering
- **Version**: V2 (January 2026)
- **License**: MIT

## Citation

If you use this dataset, please cite:

```bibtex
@dataset{endomondo_hr_v2,
  title={Endomondo Heart Rate Prediction Dataset V2},
  author={Riccardo},
  year={2026},
  publisher={Hugging Face},
  url={https://huggingface.co/datasets/rricc22/endomondo-hr-prediction-v2}
}
```

## Related Resources

- 🤗 **Model**: [heart-rate-prediction-lstm](https://huggingface.co/rricc22/heart-rate-prediction-lstm)
- 🚀 **Demo**: [Interactive Predictor](https://huggingface.co/spaces/rricc22/heart-rate-predictor)
- 📊 **GitHub**: [SUB3_V2 Repository](https://github.com/rricc22/SUB3_V2)
