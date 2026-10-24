"""
06_sceptics_sensitivity.py
--------------------------
Stage 6 of the pipeline (Section 3.7): the standing of the two-member
"Risk-Averse Sceptics" cluster, the presence of content-identical response
vectors in the sample, and the robustness of the recommended short form to
how these are handled.

Findings established in stage 05 and here:
  * The two respondents in the sceptics cluster are identical on all 40 items
    and share a pilot case.
  * More broadly, the 42 respondents reduce to 23 distinct response vectors;
    several identical groups span different pilot cases. Whatever the origin
    (duplicate submissions retained under an id-based de-duplication, or an
    artefact of data assembly), a conservative analysis restricts attention
    to the distinct patterns.

The recommended short form (Q1, Q4, Q8, Q9, Q21, Q32, Q38) is re-derived
against four alternative reference structures and compared by set overlap:
  (1) full sample, three profiles (n = 42, k = 3);
  (2) sceptics duplicate consolidated to one respondent (n = 41, k = 4);
  (3) distinct response patterns only (n = 23, k = 4/k = 3).

Short Tabu Search (6 restarts x 50 iterations) is used for each; every reference
still yields ARI = 1.000 short forms, so the comparison isolates item
identity rather than recoverability.

Outputs:
    results/sceptics_sensitivity.json

Usage:
    python src/06_sceptics_sensitivity.py
"""

import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from repo_io import RESULTS_DIR, load_items
from sf_common import multi_start_tabu, pca_ward_labels

OUT_DIR = RESULTS_DIR
CONSENSUS = {1, 4, 8, 9, 21, 32, 38}     # 1-based item numbers
K = 4


def q(indices):
    return [f"Q{i + 1}" for i in sorted(indices)]


def overlap(sel_idx):
    sel_1 = {i + 1 for i in sel_idx}
    common = sorted(sel_1 & CONSENSUS)
    return len(common), [f"Q{c}" for c in common]


def identical_groups(X):
    """Connected components of the identical-row graph."""
    n = len(X)
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for i in range(n):
        for j in range(i + 1, n):
            if np.array_equal(X[i], X[j]):
                ra, rb = find(i), find(j)
                if ra != rb:
                    parent[ra] = rb
    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return list(groups.values())


def run_reference(Xs, K, k, tag, n_runs=6):
    Y, ref = pca_ward_labels(Xs, K, k)
    sizes = pd.Series(ref).value_counts().sort_index().tolist()
    vec, met, _ = multi_start_tabu(Xs, Y, ref, n_runs=n_runs, seed_base=3100,
                                   K=K, k=k, alpha=0.3, w=0.5,
                                   min_items=5, tenure=10, n_iter=50)
    sel = np.where(vec == 1)[0].tolist()
    n_ov, ov_items = overlap(sel)
    print(f"[{tag}] k={k} n={Xs.shape[0]} sizes={sizes} -> {q(sel)} "
          f"(|S|={met['n']}, ARI={met['ari']:.3f}, proc={met['proc']:.3f}); "
          f"overlap={n_ov}/7 {ov_items}")
    return {"tag": tag, "n": int(Xs.shape[0]), "k": k,
            "reference_sizes": sizes, "min_cluster_size": int(min(sizes)),
            "short_form": q(sel), "size": met["n"], "ari": met["ari"],
            "procrustes": met["proc"], "overlap_count": n_ov,
            "overlap_items": ov_items}


def main():
    df, item_cols = load_items()
    X = df[item_cols].values
    n, p = X.shape

    groups = identical_groups(X)
    distinct = len(groups)
    dup_groups = [g for g in groups if len(g) > 1]

    results = {
        "n_respondents": int(n),
        "n_distinct_patterns": int(distinct),
        "n_rows_in_duplicate_groups": int(sum(len(g) for g in dup_groups)),
        "identical_groups": [[int(i) for i in g] for g in dup_groups],
        "consensus": q([c - 1 for c in CONSENSUS]),
        "analyses": [],
    }
    print(f"Distinct response patterns: {distinct}/{n}; "
          f"{len(dup_groups)} identical groups")

    # (0) The reference of the article, for anchoring
    Xs_full = StandardScaler().fit_transform(X)
    results["analyses"].append(run_reference(Xs_full, K, 4, "full_n42_k4"))
    # (1) Full sample, three profiles
    results["analyses"].append(run_reference(Xs_full, K, 3, "full_n42_k3"))

    # (2) Consolidate ONLY the sceptics duplicate (rows 6 & 8): drop row 8
    sceptics_group = next((g for g in dup_groups if set(g) == {6, 8}), None)
    if sceptics_group is not None:
        keep = np.ones(n, dtype=bool)
        keep[8] = False
        Xs41 = StandardScaler().fit_transform(X[keep])
        results["analyses"].append(run_reference(Xs41, K, 4, "drop_dup_sceptic_n41_k4"))

    # (3) Distinct patterns only: keep first row of each identical group
    keep_rows = sorted(g[0] for g in groups)
    Xs_dist = StandardScaler().fit_transform(X[keep_rows])
    results["distinct_kept_rows"] = keep_rows
    results["analyses"].append(run_reference(Xs_dist, K, 4, "distinct_n23_k4"))
    results["analyses"].append(run_reference(Xs_dist, K, 3, "distinct_n23_k3"))

    ov = [a["overlap_count"] for a in results["analyses"]]
    core = set.intersection(*[set(a["overlap_items"]) for a in results["analyses"]])
    results["overlap_summary"] = {
        "min": int(min(ov)), "max": int(max(ov)), "mean": float(np.mean(ov)),
        "items_common_to_all_alternatives": sorted(core)}
    print(f"\nOverlap with 7-item consensus across {len(ov)} alternative "
          f"references: min={min(ov)}, max={max(ov)}, mean={np.mean(ov):.1f} of 7")
    print(f"Items retained under every alternative: {sorted(core)}")

    with open(OUT_DIR / "sceptics_sensitivity.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Sceptics-sensitivity analysis complete.")


if __name__ == "__main__":
    main()
