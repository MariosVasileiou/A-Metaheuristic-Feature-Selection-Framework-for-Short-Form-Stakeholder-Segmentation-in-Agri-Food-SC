"""
04_subsampling_stability.py
---------------------------
Stage 4 of the pipeline: repeated-subsampling stability analysis of the
consensus short form (Section 3.6 and Algorithm 2 of the article).

Tests whether the framework reliably produces high-fidelity short forms across
random subsamples of the 42 respondents, and how stable the exact identity of
the selected items is to which respondents are in the sample. Subsamples are
drawn WITHOUT replacement, so the procedure is repeated subsampling rather than
a bootstrap.

Procedure:
  1. Draw 200 random subsamples of 35 of the 42 respondents (no replacement).
  2. For each subsample, run the full pipeline:
       standardise -> PCA(4) -> Ward(4) -> single Tabu Search run (50 iters).
  3. Record the selected items, fidelity statistics, and overlap with the
     consensus 7-item short form (Q1, Q4, Q8, Q9, Q21, Q32, Q38).
  4. Aggregate: per-item subsample selection frequency, per-subsample fidelity
     statistics, distribution of overlap with the consensus.

Figures 9 and 10 of the paper are drawn from these results by make_figures.py.

Expected runtime: about 8 to 13 minutes on one core (200 subsamples).

Usage:
    python src/04_subsampling_stability.py

Output files:
    results/subsampling_results.json           aggregated summary statistics
    results/subsampling_per_run.csv            per-subsample details (200 rows)
    results/subsampling_item_frequencies.csv   per-item subsample vs main-analysis frequencies
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial import procrustes
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from repo_io import RESULTS_DIR, load_items

# Settings (must match stages 1 to 3)
DATA_PATH = "data/alliance_items.csv"
OUT_DIR = RESULTS_DIR

N_SUBSAMPLES = 200
SUBSAMPLE_SIZE = 35       # of the 42 respondents
TABU_ITERS_PER_SUBSAMPLE = 50
ALPHA = 0.3
MIN_ITEMS = 5
TABU_TENURE = 10
N_COMPONENTS = 4
N_PROFILES = 4
SEED_BASE = 7000          # subsample seeds: SEED_BASE..SEED_BASE+N_SUBSAMPLES-1

# Consensus short form from stage 2
CONSENSUS_ITEMS = ["Q1", "Q4", "Q8", "Q9", "Q21", "Q32", "Q38"]


def load_data():
    """Load the raw Likert response matrix."""
    df, item_cols = load_items()
    return df[item_cols].values   # (42, 40)


def run_one_subsample(seed, X_raw):
    """
    One subsample: random subsample without replacement, then the full
    pipeline and one Tabu Search run.

    Returns dict with keys: seed, idx, items, size, obj, proc, ari,
    fidelity, overlap_consensus. Returns None if the Ward cut produces
    fewer than 4 distinct profiles (rare degenerate case).
    """
    rng = np.random.default_rng(seed)
    n_full, p = X_raw.shape

    idx = rng.choice(n_full, size=SUBSAMPLE_SIZE, replace=False)
    Xb = X_raw[idx]
    Xs = StandardScaler().fit_transform(Xb)
    Y_full = PCA(n_components=N_COMPONENTS).fit_transform(Xs)
    Z = linkage(Y_full, method="ward")
    profiles = fcluster(Z, t=N_PROFILES, criterion="maxclust")
    if len(np.unique(profiles)) < N_PROFILES:
        return None

    n_b = SUBSAMPLE_SIZE
    consensus_idx = set(int(q[1:]) - 1 for q in CONSENSUS_ITEMS)
    cache = {}

    def score(s):
        sel = tuple(np.where(s == 1)[0])
        if len(sel) < MIN_ITEMS:
            return -np.inf, None
        if sel in cache:
            return cache[sel]
        Xsub = Xs[:, list(sel)]
        Y_ = PCA(n_components=min(N_COMPONENTS, len(sel))).fit_transform(Xsub)
        if Y_.shape[1] < N_COMPONENTS:
            pad = np.zeros((n_b, N_COMPONENTS - Y_.shape[1]))
            Y_ = np.hstack([Y_, pad])
        try:
            _, _, d = procrustes(Y_full, Y_)
            ps = max(0.0, 1 - d)
        except Exception:
            ps = 0.0
        try:
            Zs = linkage(Y_, method="ward")
            lab = fcluster(Zs, t=N_PROFILES, criterion="maxclust")
            ari = adjusted_rand_score(profiles, lab)
        except Exception:
            ari = 0.0
        fid = 0.5 * ps + 0.5 * max(0.0, ari)
        obj = fid - ALPHA * len(sel) / p
        met = {"obj": obj, "proc": ps, "ari": ari, "fid": fid, "n": len(sel)}
        cache[sel] = (obj, met)
        return obj, met

    # Tabu Search (single restart from random init)
    s = (rng.random(p) < 0.5).astype(int)
    while s.sum() < MIN_ITEMS:
        s[rng.integers(p)] = 1
    obj_cur, met_cur = score(s)
    s_best, obj_best, met_best = s.copy(), obj_cur, dict(met_cur)
    tabu = {}
    for t in range(TABU_ITERS_PER_SUBSAMPLE):
        bj = bs = bm = None
        bo = -np.inf
        for j in range(p):
            is_t = j in tabu and tabu[j] > t
            st = s.copy()
            st[j] = 1 - st[j]
            if st.sum() < MIN_ITEMS:
                continue
            o, m = score(st)
            if is_t and o <= obj_best:
                continue
            if o > bo:
                bo, bj, bs, bm = o, j, st, m
        if bs is None:
            break
        s, obj_cur, met_cur = bs, bo, bm
        tabu[bj] = t + TABU_TENURE
        tabu = {k: v for k, v in tabu.items() if v > t}
        if obj_cur > obj_best:
            s_best, obj_best, met_best = s.copy(), obj_cur, dict(met_cur)

    selected = set(np.where(s_best == 1)[0].tolist())
    overlap = len(selected & consensus_idx)
    return {
        "seed": int(seed),
        "idx": idx.tolist(),
        "items": [f"Q{j+1}" for j in sorted(selected)],
        "size": int(met_best["n"]),
        "obj": float(obj_best),
        "proc": float(met_best["proc"]),
        "ari": float(met_best["ari"]),
        "fidelity": float(met_best["fid"]),
        "overlap_consensus": overlap,
        "selection_vector": s_best.tolist(),
    }


def main():
    OUT_DIR.mkdir(exist_ok=True)
    print(f"Loading data from {DATA_PATH}")
    X_raw = load_data()
    print(f"  Shape: {X_raw.shape}")

    print(f"\nRunning {N_SUBSAMPLES} subsamples (size {SUBSAMPLE_SIZE}, without replacement)...")
    t0 = time.time()
    results = []
    fails = 0
    for i in range(N_SUBSAMPLES):
        seed = SEED_BASE + i
        r = run_one_subsample(seed, X_raw)
        if r is None:
            fails += 1
            continue
        results.append(r)
        if (i + 1) % 25 == 0:
            elapsed = time.time() - t0
            print(f"  {i+1}/{N_SUBSAMPLES}  ({elapsed:.0f}s elapsed, "
                  f"{fails} degenerate)")

    total_time = time.time() - t0
    print(f"\nDone in {total_time:.0f}s. {len(results)} successful, "
          f"{fails} degenerate.")

    # ----- Aggregate -----
    sizes = np.array([r["size"] for r in results])
    procs = np.array([r["proc"] for r in results])
    aris = np.array([r["ari"] for r in results])
    overlaps = np.array([r["overlap_consensus"] for r in results])

    # Per-item subsample selection frequency
    p = X_raw.shape[1]
    item_counts = np.zeros(p, dtype=int)
    for r in results:
        item_counts[np.array(r["selection_vector"]) == 1] += 1
    item_freq = pd.DataFrame({
        "item": [f"Q{j+1}" for j in range(p)],
        "subsample_freq": item_counts,
        "subsample_pct": 100 * item_counts / len(results),
    }).sort_values("subsample_freq", ascending=False)

    # Merge with original frequencies for comparison
    orig = pd.read_csv(OUT_DIR / "item_frequencies.csv")
    combined = item_freq.merge(orig[["item", "pct_best_runs"]], on="item")
    combined.columns = ["item", "subsample_freq", "subsample_pct", "original_pct"]
    combined = combined.sort_values("original_pct", ascending=False)
    combined.to_csv(OUT_DIR / "subsampling_item_frequencies.csv", index=False)

    # Per-subsample details (drop the 40-element selection vector to keep the
    # CSV readable; re-derivable from item_freq if needed)
    per_run_df = pd.DataFrame([{k: v for k, v in r.items()
                                if k != "selection_vector"}
                               for r in results])
    per_run_df.to_csv(OUT_DIR / "subsampling_per_run.csv", index=False)

    # Summary JSON
    summary = {
        "n_subsamples": len(results),
        "subsample_size": SUBSAMPLE_SIZE,
        "tabu_iters_per_subsample": TABU_ITERS_PER_SUBSAMPLE,
        "consensus_items": CONSENSUS_ITEMS,
        "size": {
            "mean": float(sizes.mean()),
            "std": float(sizes.std()),
            "min": int(sizes.min()),
            "max": int(sizes.max()),
            "median": float(np.median(sizes)),
        },
        "procrustes": {"mean": float(procs.mean()), "std": float(procs.std())},
        "ari": {"mean": float(aris.mean()), "std": float(aris.std()),
                "fraction_equal_1": float((aris == 1.0).mean())},
        "overlap_with_consensus": {
            "mean": float(overlaps.mean()),
            "std": float(overlaps.std()),
            "distribution": {int(k): int(v)
                             for k, v in pd.Series(overlaps).value_counts()
                             .sort_index().items()},
        },
    }
    with open(OUT_DIR / "subsampling_results.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nProcrustes: {procs.mean():.4f} ± {procs.std():.4f}")
    print(f"ARI:        {aris.mean():.4f} ± {aris.std():.4f}  "
          f"(== 1.0 in {(aris==1.0).sum()}/{len(results)})")
    print(f"Overlap with consensus 7-item: "
          f"{overlaps.mean():.2f} ± {overlaps.std():.2f} of 7")
    print("\nTop 15 items by subsample selection frequency:")
    print(combined.head(15).to_string(index=False))

    print("Repeated-subsampling analysis complete.")


if __name__ == "__main__":
    main()
