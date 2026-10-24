"""
03_baselines_and_profiles.py
----------------------------
Stage 3 of the pipeline: baselines and profile characterisation.

1. Scores two baseline short-form selection strategies against the Tabu
   Search result (Section 3.4):
       (a) random subsets: 200 replications at the consensus size and
           100 replications at each size of the frontier;
       (b) top-loading items, ranked by summed |loading| on the first four
           principal components.
   The fidelity-size frontier (Figure 7, Table B1) is saved to frontier.csv.

2. Characterises the four stakeholder profiles by their mean responses on the
   full 40-item instrument (Table A1) and on the seven-item consensus short
   form (Table 3, Figure 8).

Expected runtime: a few seconds.

Usage:
    python src/03_baselines_and_profiles.py

Output files (results/):
    baselines.json
    frontier.csv
    profile_sf_means.csv
    profile_all_means.csv
"""

import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial import procrustes
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from repo_io import RESULTS_DIR

OUT_DIR = RESULTS_DIR

N_COMPONENTS = 4
N_PROFILES = 4

# Number of random-subset replications per size in the frontier analysis
N_RANDOM_REPLICATIONS = 100

# Subset sizes for which to build the frontier
FRONTIER_SIZES = [5, 7, 10, 15, 20, 30, 40]

# Data-driven profile names, keyed by cluster label as produced by stage 1
# (the labels 1..4 are SciPy's Ward cluster IDs from the 4-PC space).
# Sizes from stage 1: label 1 -> 2 members, label 2 -> 7, label 3 -> 4, label 4 -> 29.
# The rationale for each name is given in Section 3.1 of the article.
PROFILE_NAMES = {
    4: "Pragmatic Optimists",       # 29 members, moderate positive on all
    2: "Committed Enthusiasts",     # 7 members, strong positive, reject concerns
    3: "Operationally Cautious",    # 4 members, positive on tech but concerned
    1: "Risk-Averse Sceptics",      # 2 members, negative on opportunities, maximal concern
}


def score_proc_ari(Xs, Y_full, profiles_ref, sel):
    """Score a subset by Procrustes similarity and ARI."""
    n, p = Xs.shape
    X_sub = Xs[:, sel]
    k_use = min(N_COMPONENTS, len(sel))
    Y_sub = PCA(n_components=k_use).fit_transform(X_sub)
    if Y_sub.shape[1] < N_COMPONENTS:
        Y_sub = np.hstack([Y_sub, np.zeros((n, N_COMPONENTS - Y_sub.shape[1]))])
    try:
        _, _, disp = procrustes(Y_full, Y_sub)
        proc = max(0.0, 1 - disp)
    except Exception:
        proc = 0.0
    try:
        Z = linkage(Y_sub, method="ward")
        labels_sub = fcluster(Z, t=N_PROFILES, criterion="maxclust")
        ari = adjusted_rand_score(profiles_ref, labels_sub)
    except Exception:
        ari = 0.0
    return proc, ari


def main():
    # -----------------------------------------------------------------
    # Load the artefacts of stages 1 and 2
    # -----------------------------------------------------------------
    Xs = np.load(OUT_DIR / "Xs.npy")
    Y_full = np.load(OUT_DIR / "Y_full.npy")
    profiles_ref = np.load(OUT_DIR / "profiles_ref.npy")
    all_best_matrix = np.load(OUT_DIR / "all_best_subsets.npy")
    with open(OUT_DIR / "tabu_results.json") as f:
        tabu = json.load(f)

    X_raw = pd.read_csv(OUT_DIR / "items_raw.csv")
    n, p = Xs.shape

    # Consensus short form (Tabu Search best-of-30)
    SF = tabu["best_run_items"]
    SF_idx = [int(q[1:]) - 1 for q in SF]
    print(f"Consensus short form: {SF}")

    # -----------------------------------------------------------------
    # Fidelity of the short form and of the full instrument
    # -----------------------------------------------------------------
    proc_sf, ari_sf = score_proc_ari(Xs, Y_full, profiles_ref, SF_idx)
    proc_full, ari_full = score_proc_ari(Xs, Y_full, profiles_ref, list(range(p)))
    print(f"Short form: Procrustes={proc_sf:.4f}  ARI={ari_sf:.4f}")
    print(f"Full:       Procrustes={proc_full:.4f}  ARI={ari_full:.4f}")

    # -----------------------------------------------------------------
    # Baseline 1: random subsets at size 7 (n=200)
    # -----------------------------------------------------------------
    np.random.seed(42)
    sf_size = len(SF_idx)
    rand_procs, rand_aris = [], []
    for _ in range(200):
        sel = sorted(np.random.choice(p, sf_size, replace=False).tolist())
        pr, ar = score_proc_ari(Xs, Y_full, profiles_ref, sel)
        rand_procs.append(pr)
        rand_aris.append(ar)
    rand_procs = np.array(rand_procs)
    rand_aris = np.array(rand_aris)
    print(f"\nRandom subsets (size {sf_size}, n=200):")
    print(f"  Procrustes: mean={rand_procs.mean():.3f}  std={rand_procs.std():.3f}")
    print(f"  ARI:        mean={rand_aris.mean():.3f}  std={rand_aris.std():.3f}")
    print(f"  ARI=1.0 achieved in {100*(rand_aris==1.0).mean():.1f}% of random subsets")

    # -----------------------------------------------------------------
    # Baseline 2: top-loading items (ranked by sum of |loading| on PC1..PC4)
    # -----------------------------------------------------------------
    pca_ref = PCA(n_components=N_COMPONENTS).fit(Xs)
    loadings = pca_ref.components_.T * np.sqrt(pca_ref.explained_variance_)
    loading_magnitude = np.abs(loadings).sum(axis=1)
    top_idx = np.argsort(-loading_magnitude)
    tl_sel = top_idx[:sf_size].tolist()
    tl_labels = [f"Q{j+1}" for j in tl_sel]
    pr_tl, ar_tl = score_proc_ari(Xs, Y_full, profiles_ref, tl_sel)
    print(f"\nTop-loading baseline (size {sf_size}):")
    print(f"  Items: {tl_labels}")
    print(f"  Procrustes={pr_tl:.3f}  ARI={ar_tl:.3f}")

    # -----------------------------------------------------------------
    # Frontier: fidelity vs subset size, for random / top-loading / Tabu
    # -----------------------------------------------------------------
    print("\nBuilding fidelity-size frontier...")
    frontier_rows = []
    tabu_sizes = np.array([int(s.sum()) for s in all_best_matrix])
    for sz in FRONTIER_SIZES:
        # Random mean
        rand_scores = []
        for _ in range(N_RANDOM_REPLICATIONS):
            sel = sorted(np.random.choice(p, sz, replace=False).tolist())
            pr, ar = score_proc_ari(Xs, Y_full, profiles_ref, sel)
            rand_scores.append((pr, ar))
        rs = np.array(rand_scores)

        # Top-loading at this size
        tl_sel_sz = top_idx[:sz].tolist()
        pr_t, ar_t = score_proc_ari(Xs, Y_full, profiles_ref, tl_sel_sz)

        # Tabu's best solution observed at this size (if any)
        matches = np.where(tabu_sizes == sz)[0]
        if len(matches) > 0:
            tab_procs = [score_proc_ari(
                Xs, Y_full, profiles_ref,
                np.where(all_best_matrix[i] == 1)[0].tolist())[0]
                for i in matches]
            tab_aris = [score_proc_ari(
                Xs, Y_full, profiles_ref,
                np.where(all_best_matrix[i] == 1)[0].tolist())[1]
                for i in matches]
            best_tp, best_ta = max(tab_procs), max(tab_aris)
        else:
            best_tp = best_ta = None

        frontier_rows.append({
            "size": sz,
            "random_proc": rs[:, 0].mean(),
            "random_ari":  rs[:, 1].mean(),
            "toploading_proc": pr_t, "toploading_ari": ar_t,
            "tabu_best_proc": best_tp, "tabu_best_ari": best_ta,
        })
        print(f"  size={sz:2d}  "
              f"random proc={rs[:,0].mean():.3f} ari={rs[:,1].mean():.3f}  "
              f"top-load proc={pr_t:.3f} ari={ar_t:.3f}  "
              f"tabu proc={best_tp} ari={best_ta}")

    frontier = pd.DataFrame(frontier_rows)
    frontier.to_csv(OUT_DIR / "frontier.csv", index=False)

    # -----------------------------------------------------------------
    # Profile characterisation
    # -----------------------------------------------------------------
    X_raw["profile"] = profiles_ref
    X_raw["profile_name"] = X_raw["profile"].map(PROFILE_NAMES)
    profile_sizes = X_raw.groupby("profile_name").size().to_dict()
    print(f"\nProfile sizes: {profile_sizes}")

    profile_ordering = ["Pragmatic Optimists", "Committed Enthusiasts",
                        "Operationally Cautious", "Risk-Averse Sceptics"]

    prof_sf = X_raw.groupby("profile_name")[SF].mean().round(2).loc[profile_ordering]
    prof_all = X_raw.groupby("profile_name")[[f"Q{i+1}" for i in range(p)]].mean().round(2).loc[profile_ordering]
    prof_sf.to_csv(OUT_DIR / "profile_sf_means.csv")
    prof_all.to_csv(OUT_DIR / "profile_all_means.csv")

    print("\nShort-form item means by profile:")
    print(prof_sf.to_string())

    # -----------------------------------------------------------------
    # Save baselines summary
    # -----------------------------------------------------------------
    baselines = {
        "short_form_items": SF,
        "short_form_size": len(SF),
        "short_form_proc": float(proc_sf),
        "short_form_ari":  float(ari_sf),
        "full_proc": float(proc_full),
        "full_ari":  float(ari_full),
        "random_proc_mean": float(rand_procs.mean()),
        "random_proc_std":  float(rand_procs.std()),
        "random_ari_mean":  float(rand_aris.mean()),
        "random_ari_std":   float(rand_aris.std()),
        "random_ari_1_fraction": float((rand_aris == 1.0).mean()),
        "toploading_items": tl_labels,
        "toploading_proc": float(pr_tl),
        "toploading_ari":  float(ar_tl),
        "profile_sizes": profile_sizes,
    }
    with open(OUT_DIR / "baselines.json", "w") as f:
        json.dump(baselines, f, indent=2)

    print(f"\nSaved baselines, frontier and profile means to {OUT_DIR}/")
    print("Stage 3 complete.")


if __name__ == "__main__":
    main()
