"""
02_tabu_search.py
-----------------
Stage 2 of the pipeline: Tabu Search item selection (Sections 2.4 and 2.5).

Runs 30 independent Tabu Search runs over the space of item subsets. Each
candidate subset is scored by a composite fidelity objective:

    F(s) = 0.5 · Procrustes(Y_full, Y_s)  +  0.5 · ARI(labels_full, labels_s)
           - alpha · |s| / p                      # sparsity penalty

where
    Y_full  = reference PC scores from the full 40-item instrument
    Y_s     = PC scores when PCA is re-run on the subset s
    labels_* = Ward-4 cluster labels on the respective PC spaces

Search strategy: single-bit-flip local search with a tabu list of tenure 10
and global-best aspiration (tabu moves are admissible if they strictly
improve the best-so-far objective).

Expected runtime: about 2.5 to 4 minutes on one core (30 runs x 100
iterations, about 120,000 candidate evaluations).

Usage:
    python 02_tabu_search.py

Output files (results/):
    tabu_results.json        aggregated summary + per-run entries
    item_frequencies.csv     per-item selection frequency
    histories.csv            per-iteration trajectories for all 30 runs
    all_best_subsets.npy     30 × 40 binary matrix of best solutions
    item_freq_best_pct.npy   40-vector of best-of-run selection %
    item_freq_all_pct.npy    40-vector of all-iteration selection %
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from repo_io import RESULTS_DIR

OUT_DIR = RESULTS_DIR

# Tabu Search parameters (see Section 2.5 and Table 1 of the paper)
N_RUNS       = 30   # independent restarts from random initialisations
N_ITER       = 100  # iterations per run (last improvement at median iteration 64, Figure 5)
ALPHA        = 0.3  # sparsity penalty coefficient
TABU_TENURE  = 10   # number of iterations a flipped bit stays tabu
MIN_ITEMS    = 5    # smallest admissible subset size (needed for 4-PC PCA)
CACHE_RESET_EVERY = 10   # reset scoring cache every N runs to bound memory

# Loaded from step 1 output
_Xs = None
_Y_full = None
_profiles_ref = None
_n = None
_p = None
_K = None   # number of reference components
_K_prof = 4  # number of reference profiles


def score_subset(s, cache, alpha=ALPHA, min_items=MIN_ITEMS):
    """
    Compute the Tabu Search objective for a candidate subset.

    Parameters
    ----------
    s : np.ndarray of {0,1}, shape (p,)
        Binary selection vector: s[j]=1 means item j is in the subset.
    cache : dict
        Shared memoisation table mapping selection-tuple -> (objective, metrics).

    Returns
    -------
    obj : float         the objective F(s); -inf if constraint violated
    metrics : dict      breakdown: n_sel, proc_sim, ari, fidelity, obj
    """
    sel = tuple(np.where(s == 1)[0])
    n_sel = len(sel)
    if n_sel < min_items:
        return -np.inf, None
    if sel in cache:
        return cache[sel]

    # Sub-matrix of standardised items
    X_sub = _Xs[:, list(sel)]
    k_use = min(_K, n_sel)

    # PCA on reduced item matrix
    Y_sub = PCA(n_components=k_use).fit_transform(X_sub)
    # Pad with zeros if fewer components were produced than the reference
    if Y_sub.shape[1] < _K:
        Y_sub = np.hstack([Y_sub, np.zeros((_n, _K - Y_sub.shape[1]))])

    # Procrustes similarity on the PC score geometry
    try:
        _, _, disparity = procrustes(_Y_full, Y_sub)
        proc_sim = max(0.0, 1.0 - disparity)
    except Exception:
        proc_sim = 0.0

    # Ward clustering on the subset's PC space, then ARI against reference labels
    try:
        Z = linkage(Y_sub, method="ward")
        labels_sub = fcluster(Z, t=_K_prof, criterion="maxclust")
        ari = adjusted_rand_score(_profiles_ref, labels_sub)
    except Exception:
        ari = 0.0

    fidelity = 0.5 * proc_sim + 0.5 * max(0.0, ari)
    obj = fidelity - alpha * n_sel / _p
    metrics = {"n_sel": int(n_sel), "proc_sim": float(proc_sim),
               "ari": float(ari), "fidelity": float(fidelity), "obj": float(obj)}
    cache[sel] = (obj, metrics)
    return obj, metrics


def tabu_search_run(rng, cache, n_iter=N_ITER, tabu_tenure=TABU_TENURE,
                    alpha=ALPHA, min_items=MIN_ITEMS):
    """
    Execute one Tabu Search run from a random initial subset.

    Returns a dict containing:
      s_best         binary best-of-run solution
      obj_best       objective at s_best
      metrics_best   full metric breakdown at s_best
      history        list of dicts, one per iteration
      iter_selections  (n_iter, p) boolean matrix of trajectories
    """
    # Random Bernoulli(0.5) initialisation, resampled if too small
    while True:
        s = (rng.random(_p) < 0.5).astype(int)
        if s.sum() >= min_items:
            break

    obj_cur, met_cur = score_subset(s, cache, alpha, min_items)
    s_best, obj_best, met_best = s.copy(), obj_cur, dict(met_cur)
    tabu = {}                # j -> iteration at which j becomes free again
    history, iter_sel = [], []

    for t in range(n_iter):
        # Evaluate every single-bit-flip neighbour of the current s
        best_j = best_s = best_m = None
        best_o = -np.inf
        for j in range(_p):
            is_tabu = j in tabu and tabu[j] > t
            s_try = s.copy()
            s_try[j] = 1 - s_try[j]
            if s_try.sum() < min_items:
                continue
            o, m = score_subset(s_try, cache, alpha, min_items)
            # Aspiration criterion: tabu moves allowed only if they beat best-so-far
            if is_tabu and o <= obj_best:
                continue
            if o > best_o:
                best_o, best_j, best_s, best_m = o, j, s_try, m

        if best_s is None:       # no admissible move
            break

        # Accept the move (even if it worsens the current objective;
        # "best admissible" not "best improving")
        s, obj_cur, met_cur = best_s, best_o, best_m

        # Add the flipped item to the tabu list
        tabu[best_j] = t + tabu_tenure
        tabu = {k: v for k, v in tabu.items() if v > t}  # purge expired

        # Update global best
        if obj_cur > obj_best:
            s_best, obj_best, met_best = s.copy(), obj_cur, dict(met_cur)

        history.append({
            "iter": t, "obj_cur": obj_cur, "obj_best": obj_best,
            "n_cur": int(s.sum()),
            "proc_cur": met_cur["proc_sim"], "ari_cur": met_cur["ari"],
        })
        iter_sel.append(s.copy())

    return {
        "s_best": s_best,
        "obj_best": obj_best,
        "metrics_best": met_best,
        "history": history,
        "iter_selections": np.array(iter_sel),
    }


def main():
    global _Xs, _Y_full, _profiles_ref, _n, _p, _K

    # Load the reference artefacts of stage 1
    _Xs = np.load(OUT_DIR / "Xs.npy")
    _Y_full = np.load(OUT_DIR / "Y_full.npy")
    _profiles_ref = np.load(OUT_DIR / "profiles_ref.npy")
    _n, _p = _Xs.shape
    _K = _Y_full.shape[1]
    print(f"Loaded reference: n={_n}, p={_p}, K={_K}, profiles={_K_prof}")

    print(f"Running {N_RUNS} Tabu Search runs × {N_ITER} iters, alpha={ALPHA}")
    t0 = time.time()
    runs = []
    cache = {}

    for r in range(N_RUNS):
        # Reset cache periodically to keep memory bounded
        if r > 0 and r % CACHE_RESET_EVERY == 0:
            cache = {}

        rng = np.random.default_rng(seed=1000 + r)   # fixed seeds for repro
        tr = time.time()
        res = tabu_search_run(rng, cache)
        res["run_id"] = r
        res["time_s"] = time.time() - tr
        runs.append(res)
        print(f"  Run {r+1:2d}/{N_RUNS} | "
              f"obj={res['obj_best']:.4f} | |S|={int(res['s_best'].sum()):2d} | "
              f"proc={res['metrics_best']['proc_sim']:.3f} | "
              f"ari={res['metrics_best']['ari']:.3f} | "
              f"t={res['time_s']:.1f}s")

    total_time = time.time() - t0
    print(f"\nTotal wall time: {total_time:.1f}s ({total_time/N_RUNS:.1f}s per run)")

    # -----------------------------------------------------------------
    # Aggregate statistics
    # -----------------------------------------------------------------
    all_best = [r["s_best"] for r in runs]
    best_objs = np.array([r["obj_best"] for r in runs])
    best_sizes = np.array([int(s.sum()) for s in all_best])
    best_procs = np.array([r["metrics_best"]["proc_sim"] for r in runs])
    best_aris  = np.array([r["metrics_best"]["ari"]      for r in runs])
    best_fids  = np.array([r["metrics_best"]["fidelity"] for r in runs])

    print(f"\nObjective: mean={best_objs.mean():.4f}  std={best_objs.std():.4f}  "
          f"range=[{best_objs.min():.4f}, {best_objs.max():.4f}]")
    print(f"|S|: mean={best_sizes.mean():.2f}  median={np.median(best_sizes):.1f}  "
          f"range=[{best_sizes.min()}, {best_sizes.max()}]")
    print(f"Procrustes: mean={best_procs.mean():.4f}  std={best_procs.std():.4f}")
    print(f"ARI:        mean={best_aris.mean():.4f}  std={best_aris.std():.4f}")

    # Item selection frequencies
    best_matrix = np.vstack(all_best)                            # 30 × 40
    item_freq_best = best_matrix.sum(axis=0)                     # count per item
    item_freq_best_pct = 100.0 * item_freq_best / N_RUNS

    all_iters = np.vstack([r["iter_selections"] for r in runs])  # long × 40
    item_freq_all_pct = 100.0 * all_iters.sum(axis=0) / all_iters.shape[0]

    freq_df = pd.DataFrame({
        "item": [f"Q{j+1}" for j in range(_p)],
        "freq_best_runs": item_freq_best.astype(int),
        "pct_best_runs": item_freq_best_pct,
        "pct_all_iters": item_freq_all_pct,
    }).sort_values("freq_best_runs", ascending=False)

    print("\nTop 15 items by best-of-run selection frequency:")
    print(freq_df.head(15).to_string(index=False))

    # Consensus subsets
    unanimous = [f"Q{j+1}" for j in range(_p) if item_freq_best[j] == N_RUNS]
    majority  = [f"Q{j+1}" for j in range(_p) if item_freq_best[j] >= N_RUNS / 2]
    print(f"\nUnanimous (30/30): {len(unanimous)}  -> {unanimous}")
    print(f"Majority  (>=15):  {len(majority)}  -> {majority}")

    # Best overall solution
    best_run_idx = int(np.argmax(best_objs))
    s_final = all_best[best_run_idx]
    final_labels = [f"Q{j+1}" for j in np.where(s_final == 1)[0]]
    tied = int((best_objs == best_objs.max()).sum())
    print(f"\nBest-of-30 solution (run {best_run_idx}, "
          f"tied by {tied} runs): {final_labels}")
    print(f"  Objective = {best_objs[best_run_idx]:.4f}, "
          f"Procrustes = {best_procs[best_run_idx]:.3f}, "
          f"ARI = {best_aris[best_run_idx]:.3f}")

    # -----------------------------------------------------------------
    # Persist results
    # -----------------------------------------------------------------
    np.save(OUT_DIR / "all_best_subsets.npy", best_matrix)
    np.save(OUT_DIR / "item_freq_best_pct.npy", item_freq_best_pct)
    np.save(OUT_DIR / "item_freq_all_pct.npy", item_freq_all_pct)
    freq_df.to_csv(OUT_DIR / "item_frequencies.csv", index=False)

    # Iteration histories
    hist_records = []
    for r_idx, run in enumerate(runs):
        for h in run["history"]:
            hist_records.append({"run": r_idx, **h})
    pd.DataFrame(hist_records).to_csv(OUT_DIR / "histories.csv", index=False)

    # JSON summary
    summary = {
        "config": {
            "N_RUNS": N_RUNS, "N_ITER": N_ITER, "ALPHA": ALPHA,
            "TABU_TENURE": TABU_TENURE, "MIN_ITEMS": MIN_ITEMS,
            "K_COMPONENTS": _K, "K_PROFILES": _K_prof,
        },
        "aggregate": {
            "obj_mean": float(best_objs.mean()),
            "obj_std":  float(best_objs.std()),
            "obj_min":  float(best_objs.min()),
            "obj_max":  float(best_objs.max()),
            "size_mean":  float(best_sizes.mean()),
            "size_median": float(np.median(best_sizes)),
            "size_min":  int(best_sizes.min()),
            "size_max":  int(best_sizes.max()),
            "proc_mean": float(best_procs.mean()),
            "proc_std":  float(best_procs.std()),
            "ari_mean":  float(best_aris.mean()),
            "ari_std":   float(best_aris.std()),
            "fid_mean":  float(best_fids.mean()),
            "fid_std":   float(best_fids.std()),
            "total_time_s": float(total_time),
            "avg_time_per_run_s": float(total_time / N_RUNS),
        },
        "unanimous_items": unanimous,
        "majority_items": majority,
        "best_run_id": best_run_idx,
        "best_run_items": final_labels,
        "best_run_obj":  float(best_objs[best_run_idx]),
        "best_run_proc": float(best_procs[best_run_idx]),
        "best_run_ari":  float(best_aris[best_run_idx]),
        "best_run_size": int(best_sizes[best_run_idx]),
        "n_runs_at_best_obj": tied,
        "per_run": [{
            "run_id": r,
            "obj_best": float(best_objs[r]),
            "size":     int(best_sizes[r]),
            "proc":     float(best_procs[r]),
            "ari":      float(best_aris[r]),
            "fidelity": float(best_fids[r]),
            "items":    [f"Q{j+1}" for j in np.where(all_best[r] == 1)[0]],
            "time_s":   runs[r]["time_s"],
        } for r in range(N_RUNS)],
    }
    with open(OUT_DIR / "tabu_results.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved results to {OUT_DIR}/")
    print("Stage 2 complete: proceed to 03_baselines_and_profiles.py")


if __name__ == "__main__":
    main()
