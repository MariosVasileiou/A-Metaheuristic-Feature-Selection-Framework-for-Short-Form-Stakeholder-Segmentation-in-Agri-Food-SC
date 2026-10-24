"""
sf_common.py
------------
Shared helpers: a single, parameterised implementation of the composite-fidelity
scorer and the Tabu Search, so that the sensitivity analyses (penalty, weighting,
alternative references) reuse exactly the same machinery as the main pipeline.

The objective reproduces Eqs. (1)-(2) of the paper, generalised with a
tunable Procrustes/ARI weight w (default 0.5 = the paper's 50/50 split):

    fidelity(s) = w * Procrustes(Y_full, Y_s) + (1 - w) * max(0, ARI)
    F(s)        = fidelity(s) - alpha * |s| / p
"""

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial import procrustes
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score


def pca_ward_labels(Xs, K, k):
    """PCA(K) scores + Ward cut at k clusters."""
    Y = PCA(n_components=K).fit_transform(Xs)
    Z = linkage(Y, method="ward")
    lab = fcluster(Z, t=k, criterion="maxclust")
    return Y, lab


def score_subset(sel, Xs, Y_full, profiles_ref, K=4, k=4, alpha=0.3,
                 w=0.5, min_items=5):
    """Composite-fidelity objective for a list/array of selected item indices.

    Returns (objective, metrics_dict). Objective is -inf if |sel| < min_items.
    """
    sel = list(sel)
    n = Xs.shape[0]
    p = Xs.shape[1]
    if len(sel) < min_items:
        return -np.inf, None
    Xsub = Xs[:, sel]
    k_use = min(K, len(sel))
    Ys = PCA(n_components=k_use).fit_transform(Xsub)
    if Ys.shape[1] < K:
        Ys = np.hstack([Ys, np.zeros((n, K - Ys.shape[1]))])
    try:
        _, _, disp = procrustes(Y_full, Ys)
        proc = max(0.0, 1.0 - disp)
    except Exception:
        proc = 0.0
    try:
        _, lab = pca_ward_labels(Xsub, k_use, k)
        ari = adjusted_rand_score(profiles_ref, lab)
    except Exception:
        ari = 0.0
    fidelity = w * proc + (1.0 - w) * max(0.0, ari)
    obj = fidelity - alpha * len(sel) / p
    return obj, {"n": len(sel), "proc": float(proc), "ari": float(ari),
                 "fidelity": float(fidelity), "obj": float(obj)}


def tabu_search(Xs, Y_full, profiles_ref, K=4, k=4, alpha=0.3, w=0.5,
                min_items=5, tenure=10, n_iter=100, rng=None, cache=None):
    """Single Tabu Search run. Returns (best_selection_vector, metrics)."""
    if rng is None:
        rng = np.random.default_rng()
    if cache is None:
        cache = {}
    p = Xs.shape[1]

    def sc(vec):
        key = tuple(np.where(vec == 1)[0])
        if key in cache:
            return cache[key]
        obj, met = score_subset(list(key), Xs, Y_full, profiles_ref,
                                K, k, alpha, w, min_items)
        cache[key] = (obj, met)
        return obj, met

    s = (rng.random(p) < 0.5).astype(int)
    while s.sum() < min_items:
        s[rng.integers(p)] = 1
    obj_cur, met_cur = sc(s)
    s_best, obj_best, met_best = s.copy(), obj_cur, dict(met_cur)
    tabu = {}
    for t in range(n_iter):
        bj = bs = bm = None
        bo = -np.inf
        for j in range(p):
            is_t = j in tabu and tabu[j] > t
            st = s.copy()
            st[j] = 1 - st[j]
            if st.sum() < min_items:
                continue
            o, m = sc(st)
            if is_t and o <= obj_best:
                continue
            if o > bo:
                bo, bj, bs, bm = o, j, st, m
        if bs is None:
            break
        s, obj_cur, met_cur = bs, bo, bm
        tabu[bj] = t + tenure
        tabu = {kk: vv for kk, vv in tabu.items() if vv > t}
        if obj_cur > obj_best:
            s_best, obj_best, met_best = s.copy(), obj_cur, dict(met_cur)
    return s_best, met_best


def multi_start_tabu(Xs, Y_full, profiles_ref, n_runs=10, seed_base=2000,
                     **kw):
    """Several Tabu Search restarts; returns best-of-run selection + metrics
    and the list of all best-of-run item sets."""
    best_vec, best_met, best_obj = None, None, -np.inf
    all_sets = []
    cache = {}
    for r in range(n_runs):
        rng = np.random.default_rng(seed_base + r)
        vec, met = tabu_search(Xs, Y_full, profiles_ref, rng=rng,
                               cache=cache, **kw)
        all_sets.append(sorted(int(j) for j in np.where(vec == 1)[0]))
        if met["obj"] > best_obj:
            best_obj, best_vec, best_met = met["obj"], vec.copy(), dict(met)
    return best_vec, best_met, all_sets
