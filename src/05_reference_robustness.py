"""
05_reference_robustness.py
--------------------------
Stage 5 of the pipeline (Section 3.7): reliability of the full-instrument
PCA + Ward reference segmentation that the Tabu Search is asked to reproduce,
assessed at a sample size (n = 42) close to the number of items (p = 40).

Because the short-form objective is defined relative to the reference
segmentation, the reference itself must be shown to be reasonably reliable
under an n (=42) that is close to p (=40). This script reports:

  1. Factorability diagnostics for the standardised item matrix
       - Bartlett's test of sphericity
       - Kaiser-Meyer-Olkin (KMO) overall measure of sampling adequacy
     (both interpreted with the explicit caveat that n is close to p).

  2. Internal cluster-validity indices for the 4-PC Ward solution across
     k = 2..6 clusters (silhouette, Calinski-Harabasz, Davies-Bouldin),
     to show whether k = 4 is defensible on internal grounds (Figure 12).

  3. Sensitivity of the reference partition to the number of retained
     principal components (K = 3, 4, 5, 6): Ward(k=4) labels are recomputed
     on each K-dimensional score matrix and compared (ARI) to the K = 4
     reference used in the paper.

  4. Sensitivity to the number of profiles: the Ward tree is cut at
     k = 3, 4 and 5, and each partition is compared (ARI) with the
     four-profile reference.

  5. Leave-one-respondent-out (LORO) and leave-one-pilot-case-out (LOPO)
     stability: standardisation, PCA and Ward clustering are refit on each
     reduced sample and the induced partition is compared (ARI) with the
     reference restricted to the retained respondents (Figure 13).

  6. A focused examination of the two-member "Risk-Averse Sceptics" cluster
     (pilot-case membership, pairwise similarity, near-duplication check).

  7. The reference re-derived on the 23 distinct response vectors only
     (first record of each group of identical vectors), compared (ARI) with
     the published partition restricted to the same records, with the
     silhouette width by k on that reduced set.

Outputs:
    results/reference_robustness.json
    results/cluster_validity.csv
    results/loro_stability.csv   (one ARI per held-out respondent)
    results/lopo_stability.csv   (one ARI per held-out pilot)

Usage:
    python src/05_reference_robustness.py
"""

import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.stats import chi2
from sklearn.decomposition import PCA
from sklearn.metrics import (adjusted_rand_score, calinski_harabasz_score,
                             davies_bouldin_score, silhouette_score)
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from repo_io import RESULTS_DIR, load_items

OUT_DIR = RESULTS_DIR

N_COMPONENTS = 4
N_PROFILES = 4
PROFILE_NAMES = {4: "Pragmatic Optimists", 2: "Committed Enthusiasts",
                 3: "Operationally Cautious", 1: "Risk-Averse Sceptics"}


def bartlett_sphericity(Xs):
    """Bartlett's test of sphericity on the correlation matrix of Xs."""
    n, p = Xs.shape
    R = np.corrcoef(Xs, rowvar=False)
    sign, logdet = np.linalg.slogdet(R)
    # A near-singular R (n close to p) makes logdet very negative and chi2 very large.
    stat = -(n - 1 - (2 * p + 5) / 6) * logdet
    dof = p * (p - 1) / 2
    pval = chi2.sf(stat, dof)
    return float(stat), float(dof), float(pval), float(logdet)


def kmo_overall(Xs):
    """Kaiser-Meyer-Olkin overall measure of sampling adequacy.

    Uses the pseudo-inverse of the correlation matrix because with n close to p the
    correlation matrix is close to singular; the value is therefore reported
    as indicative rather than exact.
    """
    R = np.corrcoef(Xs, rowvar=False)
    R_inv = np.linalg.pinv(R)
    d = np.sqrt(np.outer(np.diag(R_inv), np.diag(R_inv)))
    partial = -R_inv / d                      # anti-image / partial corr
    np.fill_diagonal(partial, 0.0)
    R_off = R.copy()
    np.fill_diagonal(R_off, 0.0)
    sum_r2 = np.sum(R_off ** 2)
    sum_p2 = np.sum(partial ** 2)
    return float(sum_r2 / (sum_r2 + sum_p2))


def ref_labels(Xs, K, k):
    """PCA(K) then Ward cut at k clusters -> integer labels."""
    Y = PCA(n_components=K).fit_transform(Xs)
    Z = linkage(Y, method="ward")
    return fcluster(Z, t=k, criterion="maxclust"), Y


def main():
    OUT_DIR.mkdir(exist_ok=True)

    # ----- Load the item responses and pilot codes -----
    df, item_cols = load_items()
    X_raw = df[item_cols].values
    pilots = df["pilot"].values
    n, p = X_raw.shape
    Xs = StandardScaler().fit_transform(X_raw)

    # Reference labels used in the article (K=4, k=4)
    ref, Y_full = ref_labels(Xs, N_COMPONENTS, N_PROFILES)

    results = {"n": int(n), "p": int(p)}

    # ----- 1. Factorability diagnostics -----
    stat, dof, pval, logdet = bartlett_sphericity(Xs)
    try:
        kmo = kmo_overall(Xs)
    except Exception as e:
        kmo = None
        print("KMO computation failed:", e)
    corr_rank = int(np.linalg.matrix_rank(np.corrcoef(Xs, rowvar=False)))
    results["factorability"] = {
        "bartlett_chi2": stat, "bartlett_dof": dof, "bartlett_p": pval,
        "corr_logdet": logdet, "kmo_overall": kmo,
        "correlation_matrix_rank": corr_rank,
        "note": (f"The 40x40 item correlation matrix has rank {corr_rank} "
                 "(n close to p and repeated response vectors), so it is "
                 "singular: the Bartlett statistic rests on a numerically "
                 "meaningless log-determinant and is not informative, and the "
                 "KMO value (computed with a pseudo-inverse) is indicative "
                 "only. PCA is used here for dimension reduction feeding "
                 "clustering, not as a measurement/factor model."),
    }
    print(f"Correlation matrix rank = {corr_rank} of {p}")
    print(f"Bartlett chi2={stat:.1f} (df={dof:.0f}), p={pval:.2e} (not informative: singular matrix)")
    print(f"KMO overall = {kmo if kmo is None else round(kmo, 3)} (pseudo-inverse, indicative)")

    # ----- 2. Internal cluster validity across k -----
    validity_rows = []
    for k in range(2, 7):
        lab, _ = ref_labels(Xs, N_COMPONENTS, k)
        if len(np.unique(lab)) < 2:
            continue
        sil = silhouette_score(Y_full, lab)
        ch = calinski_harabasz_score(Y_full, lab)
        db = davies_bouldin_score(Y_full, lab)
        sizes = pd.Series(lab).value_counts().sort_index().tolist()
        validity_rows.append({"k": k, "silhouette": sil,
                              "calinski_harabasz": ch, "davies_bouldin": db,
                              "min_cluster_size": int(min(sizes)),
                              "sizes": sizes})
        print(f"k={k}: silhouette={sil:.3f}  CH={ch:.1f}  DB={db:.3f}  "
              f"sizes={sizes}")
    validity_df = pd.DataFrame(validity_rows)
    validity_df.drop(columns=["sizes"]).to_csv(
        OUT_DIR / "cluster_validity.csv", index=False)
    results["cluster_validity"] = validity_rows

    # ----- 3. Sensitivity to number of retained components -----
    comp_rows = []
    for K in [3, 4, 5, 6]:
        lab, _ = ref_labels(Xs, K, N_PROFILES)
        ari = adjusted_rand_score(ref, lab)
        sizes = pd.Series(lab).value_counts().sort_index().tolist()
        comp_rows.append({"K_components": K, "ari_vs_reference": float(ari),
                          "sizes": sizes})
        print(f"K={K} components: ARI vs paper reference = {ari:.3f}, "
              f"sizes={sizes}")
    results["component_sensitivity"] = comp_rows

    # ----- 4. Sensitivity to number of profiles -----
    prof_rows = []
    for k in [3, 4, 5]:
        lab, _ = ref_labels(Xs, N_COMPONENTS, k)
        ari = adjusted_rand_score(ref, lab)
        sizes = pd.Series(lab).value_counts().sort_index().tolist()
        prof_rows.append({"k_profiles": k, "ari_vs_reference_k4": float(ari),
                          "sizes": sizes})
    results["profile_count_sensitivity"] = prof_rows

    # ----- 5a. Leave-one-respondent-out stability -----
    loro_aris = []
    for i in range(n):
        keep = np.arange(n) != i
        Xs_i = StandardScaler().fit_transform(X_raw[keep])
        lab_i, _ = ref_labels(Xs_i, N_COMPONENTS, N_PROFILES)
        ari = adjusted_rand_score(ref[keep], lab_i)
        loro_aris.append(ari)
    loro_aris = np.array(loro_aris)
    print(f"\nLORO: mean ARI={loro_aris.mean():.3f} "
          f"(min {loro_aris.min():.3f}), "
          f"ARI=1.0 in {int((loro_aris == 1).sum())}/{n}")

    # ----- 5b. Leave-one-pilot-case-out stability -----
    lopo_rows = []
    for pilot in pd.unique(pilots):
        keep = pilots != pilot
        Xs_p = StandardScaler().fit_transform(X_raw[keep])
        lab_p, _ = ref_labels(Xs_p, N_COMPONENTS, N_PROFILES)
        ari = adjusted_rand_score(ref[keep], lab_p)
        lopo_rows.append({"pilot_removed": str(pilot),
                          "n_removed": int((~keep).sum()),
                          "ari_vs_reference": float(ari),
                          "n_distinct_clusters": int(len(np.unique(lab_p)))})
        print(f"LOPO drop '{pilot}': ARI={ari:.3f}")
    lopo_df = pd.DataFrame(lopo_rows)

    pd.DataFrame({"loro_ari": loro_aris}).to_csv(OUT_DIR / "loro_stability.csv",
                                                 index=False)
    lopo_df.to_csv(OUT_DIR / "lopo_stability.csv", index=False)
    results["loro"] = {
        "mean_ari": float(loro_aris.mean()), "min_ari": float(loro_aris.min()),
        "std_ari": float(loro_aris.std()),
        "fraction_ari_1": float((loro_aris == 1).mean()),
        "n_below_0_9": int((loro_aris < 0.9).sum()),
    }
    results["lopo"] = lopo_rows

    # ----- 6. Risk-Averse Sceptics examination -----
    ras_label = [lab for lab, name in PROFILE_NAMES.items()
                 if name == "Risk-Averse Sceptics"][0]
    ras_idx = np.where(ref == ras_label)[0]
    ras_pilots = [str(pilots[i]) for i in ras_idx]
    # pairwise Pearson correlation between the two raw response vectors
    if len(ras_idx) == 2:
        a, b = X_raw[ras_idx[0]], X_raw[ras_idx[1]]
        pair_r = float(np.corrcoef(a, b)[0, 1])
        exact_agree = int((a == b).sum())
    else:
        pair_r, exact_agree = None, None
    # median within-sample pairwise correlation for context
    C = np.corrcoef(X_raw)
    off = C[np.triu_indices(n, 1)]
    results["risk_averse_sceptics"] = {
        "members_row_index": ras_idx.tolist(),
        "pilot_cases": ras_pilots,
        "same_pilot": len(set(ras_pilots)) == 1,
        "pairwise_pearson_r": pair_r,
        "n_items_identical_of_40": exact_agree,
        "sample_median_pairwise_r": float(np.median(off)),
        "sample_p95_pairwise_r": float(np.percentile(off, 95)),
    }
    print(f"\nRisk-Averse Sceptics: rows {ras_idx.tolist()}, "
          f"pilots {ras_pilots}, pairwise r={pair_r}")

    # ----- 7. Reference on the distinct response vectors only -----
    seen, reps = set(), []
    for i in range(n):
        key = tuple(X_raw[i])
        if key not in seen:
            seen.add(key)
            reps.append(i)
    reps = np.array(reps)
    Xs_d = StandardScaler().fit_transform(X_raw[reps])
    lab_d, Y_d = ref_labels(Xs_d, N_COMPONENTS, N_PROFILES)
    ari_d = adjusted_rand_score(ref[reps], lab_d)
    Z_d = linkage(Y_d, method="ward")
    k_gap = len(reps) - (int(np.argmax(np.diff(Z_d[:, 2]))) + 1)
    sil_d = {k: float(silhouette_score(Y_d, fcluster(Z_d, t=k,
                                                     criterion="maxclust")))
             for k in range(2, 7)}
    k_sil = max(sil_d, key=sil_d.get)
    results["distinct_vectors_reference"] = {
        "n_distinct": int(len(reps)),
        "kept_rows": reps.tolist(),
        "reference_sizes": pd.Series(lab_d).value_counts().sort_index().tolist(),
        "ari_vs_published_restricted": float(ari_d),
        "silhouette_by_k": {str(k): v for k, v in sil_d.items()},
        "silhouette_best_k": int(k_sil),
        "largest_merge_gap_k": int(k_gap),
    }
    print(f"\nDistinct vectors only: n={len(reps)}, ARI vs published "
          f"(restricted) = {ari_d:.3f}; silhouette-best k = {k_sil} "
          f"({sil_d[k_sil]:.3f})")

    with open(OUT_DIR / "reference_robustness.json", "w") as f:
        json.dump(results, f, indent=2)

    print("Reference-robustness analysis complete.")


if __name__ == "__main__":
    main()
