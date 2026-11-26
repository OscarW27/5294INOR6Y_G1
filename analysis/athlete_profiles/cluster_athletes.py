"""Group athletes into running profiles with k-means.

Pipeline
    1. Per workout: derive features from the HR / speed / altitude time series
       (efficiency, pacing, cardiac drift, within-run speed variability).
    2. Per athlete: aggregate workouts into one feature vector (medians/shares),
       keeping only athletes with enough workouts for a stable profile.
    3. Transform: log-transform skewed features, then standardise (z-scores).
    4. Pick k: compare k = 2..10 on inertia (elbow), silhouette, Davies-Bouldin
       and stability (adjusted Rand index between runs on bootstrap samples).
    5. Fit k-means, describe each cluster in real units, save plots + CSVs.

Run from the repo root:

    .venv/Scripts/python analysis/athlete_profiles/cluster_athletes.py          # pick k by silhouette
    .venv/Scripts/python analysis/athlete_profiles/cluster_athletes.py --k 4    # force k
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, davies_bouldin_score, silhouette_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "data" / "endomondo-hr-prediction-v2" / "dataset.parquet"
OUT = Path(__file__).resolve().parent / "output"

# Features used for clustering, with the reason each one is in there.
FEATURES = {
    # Volume / habit
    "runs_per_week":      "training frequency over the athlete's active period",
    "distance_km":        "typical run length (median)",
    "long_run_km":        "longest regular run (90th percentile distance)",
    # Speed & fitness
    "speed_kmh":          "typical pace (median mean speed)",
    "efficiency":         "speed per heartbeat: km/h per 100 bpm (aerobic fitness)",
    # Intensity
    "rel_intensity":      "median HR as % of the athlete's own peak HR (p95 of hr_max)",
    "share_intensive":    "fraction of workouts labelled INTENSIVE",
    # Run style
    "speed_cv":           "within-run speed variation (intervals/fartlek vs steady)",
    "opening_pace_ratio": "speed first 15% / rest of run (>1 = starts too fast), measure M-01",
    "cardiac_drift_pct":  "efficiency drop 1st->2nd half of run in %, measure M-04",
    "climb_m_per_km":     "elevation gain per km (hilly vs flat routes)",
}
LOG_FEATURES = ["runs_per_week", "distance_km", "long_run_km", "climb_m_per_km"]


# ---------------------------------------------------------------- features --

def workout_features(df: pd.DataFrame) -> pd.DataFrame:
    """One row per workout with time-series derived measures."""
    hr = np.stack(df["heart_rate"].to_numpy()).astype("float32")   # (n, 500)
    sp = np.stack(df["speed"].to_numpy()).astype("float32")
    n_pts = sp.shape[1]
    cut15, half = int(n_pts * 0.15), n_pts // 2

    ef = sp / np.clip(hr, 1, None)                                  # efficiency per timestep
    ef1, ef2 = ef[:, :half].mean(1), ef[:, half:].mean(1)

    start = df["timestamp"].map(lambda t: t[0])
    w = pd.DataFrame({
        "user_id": df["user_id"].to_numpy(),
        "workout_type": df["workout_type"].to_numpy(),
        "date": pd.to_datetime(start.to_numpy(), unit="s"),
        "distance_km": (df["speed_mean"] * df["duration_min"] / 60).to_numpy(),
        "speed_kmh": df["speed_mean"].to_numpy(),
        "hr_mean": df["hr_mean"].to_numpy(),
        "hr_max": df["hr_max"].to_numpy(),
        "altitude_gain": df["altitude_gain"].to_numpy(),
        "efficiency": 100 * df["speed_mean"].to_numpy() / df["hr_mean"].to_numpy(),
        "speed_cv": sp.std(1) / np.clip(sp.mean(1), 0.1, None),
        "opening_pace_ratio": sp[:, :cut15].mean(1) / np.clip(sp[:, cut15:].mean(1), 0.1, None),
        "cardiac_drift_pct": 100 * (ef1 - ef2) / np.clip(ef1, 1e-6, None),
    })
    return w


def athlete_features(w: pd.DataFrame, min_workouts: int) -> pd.DataFrame:
    """Aggregate workouts into one robust feature vector per athlete."""
    g = w.groupby("user_id")
    span_weeks = (g["date"].max() - g["date"].min()).dt.days / 7
    a = pd.DataFrame({
        "workouts": g.size(),
        "runs_per_week": g.size() / span_weeks.clip(lower=1),
        "distance_km": g["distance_km"].median(),
        "long_run_km": g["distance_km"].quantile(0.9),
        "speed_kmh": g["speed_kmh"].median(),
        "efficiency": g["efficiency"].median(),
        "rel_intensity": g["hr_mean"].median() / g["hr_max"].quantile(0.95),
        "share_intensive": g["workout_type"].apply(lambda s: (s == "INTENSIVE").mean()),
        "speed_cv": g["speed_cv"].median(),
        "opening_pace_ratio": g["opening_pace_ratio"].median(),
        "cardiac_drift_pct": g["cardiac_drift_pct"].median(),
        "climb_m_per_km": g["altitude_gain"].sum() / g["distance_km"].sum(),
        "share_recovery": g["workout_type"].apply(lambda s: (s == "RECOVERY").mean()),  # descriptive only
    })
    return a[a["workouts"] >= min_workouts].dropna()


def to_matrix(a: pd.DataFrame) -> tuple[np.ndarray, StandardScaler]:
    X = a[list(FEATURES)].copy()
    for c in set(LOG_FEATURES) & set(X.columns):
        X[c] = np.log1p(X[c])
    # Clip extreme outliers (a few broken GPS tracks) so they don't own a cluster.
    X = X.clip(X.quantile(0.01), X.quantile(0.99), axis=1)
    scaler = StandardScaler().fit(X)
    return scaler.transform(X), scaler


# ------------------------------------------------------------- choosing k --

def evaluate_k(X: np.ndarray, ks, seed: int, n_boot: int = 20) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for k in ks:
        km = KMeans(k, n_init=20, random_state=seed).fit(X)
        # Stability: fit on bootstrap samples, compare labels on the full set.
        aris = []
        for _ in range(n_boot):
            idx = rng.choice(len(X), len(X), replace=True)
            kb = KMeans(k, n_init=5, random_state=int(rng.integers(1e9))).fit(X[idx])
            aris.append(adjusted_rand_score(km.labels_, kb.predict(X)))
        rows.append({
            "k": k,
            "inertia": km.inertia_,
            "silhouette": silhouette_score(X, km.labels_),
            "davies_bouldin": davies_bouldin_score(X, km.labels_),
            "stability_ari": float(np.mean(aris)),
            "smallest_cluster": int(np.bincount(km.labels_).min()),
        })
    return pd.DataFrame(rows).set_index("k")


# ----------------------------------------------------------- description --

def name_cluster(z: pd.Series) -> str:
    """Rough human label from the cluster's strongest z-score traits."""
    traits = {
        "runs_per_week": ("frequent", "occasional"),
        "distance_km": ("long-distance", "short-distance"),
        "speed_kmh": ("fast", "slow"),
        "efficiency": ("efficient", "low-efficiency"),
        "rel_intensity": ("high-intensity", "easy-effort"),
        "share_intensive": ("interval-heavy", None),
        "speed_cv": ("variable-pace", "steady-pace"),
        "opening_pace_ratio": ("fast-starter", "negative-splitter"),
        "cardiac_drift_pct": ("high-drift", "durable"),
        "climb_m_per_km": ("hilly", "flat"),
    }
    picked = []
    for feat in z.abs().sort_values(ascending=False).index:
        if feat not in traits or abs(z[feat]) < 0.5:
            continue
        word = traits[feat][0 if z[feat] > 0 else 1]
        if word:
            picked.append(word)
        if len(picked) == 3:
            break
    return ", ".join(picked) or "average"


def plot_k_selection(scores: pd.DataFrame, k: int, path: Path) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.5))
    for ax, col, title in zip(axes, ["inertia", "silhouette", "davies_bouldin", "stability_ari"],
                              ["Inertia (elbow)", "Silhouette (higher = better)",
                               "Davies-Bouldin (lower = better)", "Bootstrap stability (ARI)"]):
        ax.plot(scores.index, scores[col], marker="o", color="#3b6ea8")
        ax.axvline(k, color="#c0504d", ls="--", lw=1)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("k")
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_pca(X: np.ndarray, labels: np.ndarray, names: dict, path: Path) -> pd.DataFrame:
    pca = PCA(2).fit(X)
    P = pca.transform(X)
    fig, ax = plt.subplots(figsize=(8, 6))
    for c in np.unique(labels):
        m = labels == c
        ax.scatter(P[m, 0], P[m, 1], s=14, alpha=0.7, label=f"{c}: {names[c]} (n={m.sum()})")
    # Feature loadings as arrows: which direction means what.
    for i, f in enumerate(FEATURES):
        vx, vy = pca.components_[:, i] * 3
        ax.arrow(0, 0, vx, vy, color="grey", alpha=0.6, head_width=0.06)
        ax.text(vx * 1.1, vy * 1.1, f, fontsize=7, color="dimgrey")
    ev = pca.explained_variance_ratio_
    ax.set_xlabel(f"PC1 ({ev[0]:.0%} of variance)")
    ax.set_ylabel(f"PC2 ({ev[1]:.0%} of variance)")
    ax.legend(fontsize=7, loc="best")
    ax.set_title("Athletes in PCA space, coloured by k-means cluster")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return pd.DataFrame(pca.components_.T, index=list(FEATURES), columns=["PC1", "PC2"])


def plot_heatmap(zprof: pd.DataFrame, names: dict, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 0.6 * len(zprof) + 2))
    im = ax.imshow(zprof.values, cmap="RdBu_r", vmin=-1.5, vmax=1.5, aspect="auto")
    ax.set_xticks(range(zprof.shape[1]), zprof.columns, rotation=40, ha="right", fontsize=8)
    ax.set_yticks(range(len(zprof)), [f"{c}: {names[c]}" for c in zprof.index], fontsize=8)
    for i in range(zprof.shape[0]):
        for j in range(zprof.shape[1]):
            ax.text(j, i, f"{zprof.iat[i, j]:+.1f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, label="cluster mean (z-score)")
    ax.set_title("Cluster profiles: how each group differs from the average athlete")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ------------------------------------------------------------------ main --

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--k", type=int, help="number of clusters (default: best silhouette in 3..8)")
    p.add_argument("--min-workouts", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--drop", nargs="*", default=[], choices=list(FEATURES),
                   help="features to leave out of the clustering, e.g. --drop efficiency long_run_km")
    args = p.parse_args()
    for f in args.drop:
        FEATURES.pop(f)
    OUT.mkdir(exist_ok=True)

    print("Loading workouts ...")
    cols = ["user_id", "workout_type", "duration_min", "timestamp", "heart_rate", "speed",
            "hr_mean", "hr_max", "speed_mean", "altitude_gain"]
    w = workout_features(pd.read_parquet(SRC, columns=cols))
    a = athlete_features(w, args.min_workouts)
    print(f"  {len(w):,} workouts -> {len(a)} athletes with >= {args.min_workouts} workouts")

    # Correlated features count double in k-means; check before trusting the result.
    corr = a[list(FEATURES)].corr(method="spearman")
    high = [(i, j, corr.loc[i, j]) for n, i in enumerate(corr) for j in corr.columns[n + 1:]
            if abs(corr.loc[i, j]) > 0.7]
    print("\nStrongly correlated feature pairs (|rho| > 0.7):")
    for i, j, r in high or [("none", "", 0)]:
        print(f"  {i:<20} {j:<20} {r:+.2f}" if j else "  none")

    X, _ = to_matrix(a)

    print("\nChoosing k ...")
    scores = evaluate_k(X, range(2, 11), args.seed)
    print(scores.round(3).to_string())
    k = args.k or int(scores.loc[3:8, "silhouette"].idxmax())
    print(f"-> using k = {k}")

    km = KMeans(k, n_init=50, random_state=args.seed).fit(X)
    a["cluster"] = km.labels_

    zprof = pd.DataFrame(X, index=a.index, columns=list(FEATURES)).groupby(a["cluster"]).mean()
    names = {c: name_cluster(zprof.loc[c]) for c in zprof.index}
    a["profile"] = a["cluster"].map(names)

    profile = a.groupby("cluster")[list(FEATURES) + ["share_recovery", "workouts"]].median()
    profile.insert(0, "n_athletes", a["cluster"].value_counts().sort_index())
    profile.insert(0, "profile", pd.Series(names))

    print("\nCluster profiles (medians, real units):")
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(profile.round(2).to_string())

    plot_k_selection(scores, k, OUT / "k_selection.png")
    loadings = plot_pca(X, km.labels_, names, OUT / "pca_clusters.png")
    plot_heatmap(zprof, names, OUT / "cluster_heatmap.png")
    a.round(4).to_csv(OUT / "athlete_clusters.csv")
    profile.round(3).to_csv(OUT / "cluster_profiles.csv")
    scores.round(4).to_csv(OUT / "k_selection.csv")
    loadings.round(3).to_csv(OUT / "pca_loadings.csv")
    print(f"\nWrote plots and CSVs to {OUT.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
