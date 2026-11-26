# Master prompt: k-means runner profiles notebook

Paste the prompt below into Claude Code (or another coding assistant) from the root of this repository.

---

I want to use unsupervised machine learning to group the athletes in this project into **running profiles**, and I need a simple, well-explained Jupyter notebook that supports the "Machine-Learning Approach" section of our report.

**Data.** Use the Endomondo HR Prediction V2 dataset in `data/endomondo-hr-prediction-v2/dataset.parquet` (download it with `python scripts/download_data.py` if it's missing). Each row is one running workout with 500-step time series for `heart_rate`, `speed` and `timestamp`, plus summary columns (`hr_mean`, `hr_max`, `speed_mean`, `duration_min`, `altitude_gain`, `workout_type`).

**Where to put it.** Create the notebook in `analysis/report_ml/` and make sure `analysis/` is gitignored. Use the project's `.venv`. Install scikit-learn, matplotlib and ipykernel there if they're missing, but don't change `requirements.txt`. Execute the notebook so it's saved with its outputs, and make it as visual as possible.

**Steps the notebook must contain, each with a short explanation:**

1. **Features per workout**, computed from the time series:
   - `efficiency` = 100 × speed_mean / hr_mean (km/h per 100 bpm)
   - `opening_pace_ratio` = mean speed over the first 15% of the series / mean over the rest
   - `cardiac_drift_pct` = % drop in speed/HR efficiency from the first half to the second half
   - `speed_cv` = std / mean of speed within the run
   - `distance_km` = speed_mean × duration_min / 60
2. **Features per athlete.** Only use athletes with **at least 10 workouts** (fewer gives noisy profiles). Aggregate with medians:
   - `runs_per_week` = workouts / weeks between first and last workout (minimum 1 week)
   - `distance_km`, `efficiency`, `speed_cv`, `opening_pace_ratio`, `cardiac_drift_pct`: median per athlete
   - `rel_intensity` = median hr_mean / 95th percentile of hr_max
   - `share_intensive` = share of workouts with `workout_type == "INTENSIVE"`
   - `climb_m_per_km` = total altitude gain / total distance

   Don't include mean speed or the 90th-percentile distance: they correlate strongly with efficiency and distance, so k-means would count those traits twice.
3. **Scaling:** log1p of `runs_per_week`, `distance_km` and `climb_m_per_km`; clip every feature to its 1st–99th percentile; then standardise to z-scores.
4. **Choosing k** for k = 2…10, shown as one figure with four panels saved as `output/chart.png`:
   - inertia (elbow)
   - silhouette score
   - Davies-Bouldin index
   - bootstrap stability: the mean adjusted Rand index between the full-data clustering and 20 clusterings refitted on bootstrap samples

   Use `random_state=42`, `n_init=20` for the scored models, and `n_init=5` for the bootstrap refits. Highlight k = 3 and explain why it's the best trade-off, not a perfect fit. Silhouette scores are low overall, so the profiles overlap.
5. **K-means with k = 3** (`n_init=50`, `random_state=42`). Show the cluster centres as a z-score heatmap (0 = the average athlete).
6. **Naming the clusters with a fixed rule**, because k-means numbers clusters arbitrarily:
   1. The highest `share_intensive` → *Interval-focused*.
   2. Of the remaining two, the highest `efficiency` → *Efficient & consistent*.
   3. The last one → *Fast starter / developing*.
7. **The profiles in real units** (medians per profile), plus a check table that recalculates every number quoted in the report text.
8. **A 2D PCA map** of the athletes coloured by profile, showing that the groups overlap.

**Expected results, to check that your recreation matches:**
- 569 athletes with at least 10 workouts
- At k = 3: silhouette 0.158, Davies-Bouldin 1.94, bootstrap stability ≈ 0.55
- Cluster sizes 306 / 216 / 47

| Profile | Key medians |
|---|---|
| Efficient & consistent | 0.72 runs/week, 12.0 km, cardiac drift 4.0% |
| Fast starter / developing | 0.36 runs/week, 9.8 km, drift 6.8%, opening pace ratio 1.01 |
| Interval-focused | ≈ 33% intensive workouts |

**Good to know:**
- The dataset's `workout_type` is essentially a heart-rate threshold. "Interval-focused" therefore means "often runs at high heart rate", not proven interval training.
- Present the profiles as tendencies, not hard categories.
