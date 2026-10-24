"""
01_pca.py
---------
Stage 1 of the pipeline: the full-instrument reference structure.

Loads the 40 item responses of the 42 respondents (data/alliance_items.csv),
standardises them, runs PCA, retains the first four components and applies
Ward hierarchical clustering to the four-component score matrix, which gives
the four-profile reference segmentation (Sections 2.3 and 3.1). A k-means
solution on the same scores is computed as a cross-check.

Expected runtime: about 1 second.

Usage:
    python src/01_pca.py

Output files (results/):
    Xs.npy            standardised 42 x 40 item matrix
    Y_full.npy        reference principal-component scores (42 x 4)
    profiles_ref.npy  reference profile labels (42,)
    pca_stats.json    variance decomposition and profile sizes
    demo_stats.json   sample composition (coarsened public-release fields)
    items_raw.csv     the 40 item responses, columns Q1 to Q40
    demographics.csv  copy of the coarsened demographics
"""

import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from repo_io import RESULTS_DIR, load_items, load_demographics

DATA_PATH = "data/alliance_items.csv"
OUT_DIR = RESULTS_DIR

# PCA and clustering settings
N_COMPONENTS = 4   # number of PCs to retain for the reference structure
N_PROFILES   = 4   # number of stakeholder profiles to extract

# Reproducibility
np.random.seed(42)


def main():
    OUT_DIR.mkdir(exist_ok=True)

    # -----------------------------------------------------------------
    # 1. Load the item responses (columns Q1..Q40) and demographics
    # -----------------------------------------------------------------
    print("Loading data from", DATA_PATH)
    df, item_cols = load_items()
    assert len(item_cols) == 40, f"Expected 40 item columns, got {len(item_cols)}"
    X_raw = df[item_cols].copy()
    demo = load_demographics()

    print(f"Data shape: {X_raw.shape}  (respondents × items)")
    print(f"Likert range: [{X_raw.values.min()}, {X_raw.values.max()}]")
    assert not X_raw.isna().any().any(), "Unexpected missing values"

    # -----------------------------------------------------------------
    # 2. Standardise and run PCA
    # -----------------------------------------------------------------
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X_raw.values)     # zero mean, unit variance
    pca_full = PCA()
    scores = pca_full.fit_transform(Xs)
    evr = pca_full.explained_variance_ratio_
    cum_evr = np.cumsum(evr)

    print("\nExplained variance by principal component:")
    for i in range(10):
        print(f"  PC{i+1}: {evr[i]*100:6.2f}%   (cumulative {cum_evr[i]*100:6.2f}%)")

    # -----------------------------------------------------------------
    # 3. Retain K principal components as reference structure
    # -----------------------------------------------------------------
    Y_full = scores[:, :N_COMPONENTS]    # n × K reference score matrix

    # -----------------------------------------------------------------
    # 4. Ward hierarchical clustering on PC-score space
    # -----------------------------------------------------------------
    Z = linkage(Y_full, method="ward")
    profiles_ref = fcluster(Z, t=N_PROFILES, criterion="maxclust")

    print(f"\nReference profile sizes ({N_PROFILES} profiles, Ward):")
    for p, n in sorted(pd.Series(profiles_ref).value_counts().to_dict().items()):
        print(f"  Profile {p}:  {n} respondents")

    # Robustness cross-check: k-means on the same score matrix
    km = KMeans(n_clusters=N_PROFILES, n_init=30, random_state=42).fit(Y_full)
    ari_ward_km = adjusted_rand_score(profiles_ref, km.labels_)
    print(f"\nARI(Ward, k-means) = {ari_ward_km:.4f}  "
          f"{'(identical)' if ari_ward_km == 1.0 else '(differ)'}")

    # -----------------------------------------------------------------
    # 5. Save artefacts
    # -----------------------------------------------------------------
    np.save(OUT_DIR / "Xs.npy", Xs)
    np.save(OUT_DIR / "Y_full.npy", Y_full)
    np.save(OUT_DIR / "profiles_ref.npy", profiles_ref)

    # PCA and profile statistics
    pca_stats = {
        "evr_top10": evr[:10].tolist(),
        "cum_top10": cum_evr[:10].tolist(),
        "K_components": N_COMPONENTS,
        "variance_explained_k": float(cum_evr[N_COMPONENTS - 1]),
        "K_profiles": N_PROFILES,
        "profile_sizes_ward":
            pd.Series(profiles_ref).value_counts().sort_index().to_dict(),
        "profile_sizes_kmeans":
            pd.Series(km.labels_).value_counts().sort_index().to_dict(),
        "ari_ward_km": float(ari_ward_km),
    }
    with open(OUT_DIR / "pca_stats.json", "w") as f:
        json.dump(pca_stats, f, indent=2, default=str)

    # Sample composition. Demographics are released in coarsened form (age in
    # bands, rare organisation types pooled, pilots coded), and the summary
    # reflects that granularity.
    demo_stats = {
        "n_total": int(len(X_raw)),
        "n_items": int(X_raw.shape[1]),
        "age_band": demo["age_band"].value_counts().to_dict(),
        "gender": demo["gender"].value_counts().to_dict(),
        "country": demo["country"].value_counts().to_dict(),
        "pilot": demo["pilot"].value_counts().to_dict(),
        "org_type": demo["org_type"].value_counts().to_dict(),
    }
    with open(OUT_DIR / "demo_stats.json", "w") as f:
        json.dump(demo_stats, f, indent=2, default=str)

    # The item matrix and the demographics, for the later stages
    X_raw.to_csv(OUT_DIR / "items_raw.csv", index=False)
    demo.to_csv(OUT_DIR / "demographics.csv", index=False)

    print(f"\nWrote reference artefacts to {OUT_DIR}/")
    print("Stage 1 complete: proceed to 02_tabu_search.py")


if __name__ == "__main__":
    main()
