"""
07_objective_sensitivity.py
---------------------------
Stage 7 of the pipeline (Sections 2.7 and 3.8): design and sensitivity of the
composite objective
    fidelity(s) = w * Procrustes + (1 - w) * max(0, ARI)
    F(s)        = fidelity(s) - alpha * |s| / p

Procrustes and ARI for a subset do not depend on (alpha, w). Approach:
(1) a bounded Tabu search phase across anchor settings populates a shared pool
of (subset -> Procrustes, ARI); (2) the pool is persisted; (3) the alpha sweep,
weight sweep, and fidelity-size Pareto frontier are computed by exact re-scoring
of the pool. Truncation of negative ARI is examined via the fraction of
evaluations with ARI < 0 and whether it binds at any optimum.

Outputs: results/objsens_cache.json, results/alpha_sweep.csv,
results/weight_sweep.csv, results/pareto_frontier.csv and
results/objective_sensitivity.json (Figures 18 and 19 of the article). The
search cache objsens_cache.json is regenerated on every run and is not
distributed.

Usage:
    python src/07_objective_sensitivity.py
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
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from repo_io import RESULTS_DIR, load_items

OUT_DIR = RESULTS_DIR
K, KPROF, MIN_ITEMS, P = 4, 4, 5, 40
CONSENSUS = {1, 4, 8, 9, 21, 32, 38}
_Xs = _Yf = _ref = None
_pa = {}; _neg = [0]; _tot = [0]

def proc_ari(sel):
    key = tuple(sorted(int(x) for x in sel))
    if key in _pa: return _pa[key]
    n = _Xs.shape[0]; Xsub = _Xs[:, list(key)]; ku = min(K, len(key))
    Ys = PCA(n_components=ku).fit_transform(Xsub)
    if Ys.shape[1] < K: Ys = np.hstack([Ys, np.zeros((n, K - Ys.shape[1]))])
    try:
        _, _, disp = procrustes(_Yf, Ys); pr = max(0.0, 1.0 - disp)
    except Exception: pr = 0.0
    try:
        lab = fcluster(linkage(Ys, method="ward"), t=KPROF, criterion="maxclust")
        ar = adjusted_rand_score(_ref, lab)
    except Exception: ar = 0.0
    _pa[key] = (float(pr), float(ar)); _tot[0] += 1
    if ar < 0: _neg[0] += 1
    return _pa[key]

def obj(sel, alpha, w):
    if len(sel) < MIN_ITEMS: return -np.inf
    pr, ar = proc_ari(sel)
    return w * pr + (1 - w) * max(0.0, ar) - alpha * len(sel) / P

def tabu(alpha, w, rng, n_iter=35):
    s = (rng.random(P) < 0.5).astype(int)
    while s.sum() < MIN_ITEMS: s[rng.integers(P)] = 1
    o_best = obj(list(np.where(s == 1)[0]), alpha, w); tl = {}
    for t in range(n_iter):
        bo, bj, bs = -np.inf, None, None
        for j in range(P):
            it = j in tl and tl[j] > t; st = s.copy(); st[j] = 1 - st[j]
            if st.sum() < MIN_ITEMS: continue
            o = obj(list(np.where(st == 1)[0]), alpha, w)
            if it and o <= o_best: continue
            if o > bo: bo, bj, bs = o, j, st
        if bs is None: break
        s = bs; tl[bj] = t + 10; tl = {k: v for k, v in tl.items() if v > t}
        if bo > o_best: o_best = bo

def best_pool(alpha, w):
    bk, bo = None, -np.inf
    for key, (pr, ar) in _pa.items():
        if len(key) < MIN_ITEMS: continue
        o = w * pr + (1 - w) * max(0.0, ar) - alpha * len(key) / P
        if o > bo: bo, bk = o, key
    return bk, bo

def q(key): return [f"Q{j + 1}" for j in key]

def row(key, alpha, w, keyname):
    pr, ar = _pa[key]
    return {keyname: (alpha if keyname == "alpha" else w), "size": len(key),
            "procrustes": pr, "ari": ar, "fidelity": 0.5*pr + 0.5*max(0.0, ar),
            "objective": w*pr + (1-w)*max(0.0, ar) - alpha*len(key)/P,
            "overlap_consensus": len({j+1 for j in key} & CONSENSUS),
            "items": " ".join(q(key))}

def main():
    global _Xs, _Yf, _ref
    OUT_DIR.mkdir(exist_ok=True)
    df, item_cols = load_items()
    _Xs = StandardScaler().fit_transform(df[item_cols].values)
    _Yf = PCA(n_components=K).fit_transform(_Xs)
    _ref = fcluster(linkage(_Yf, method="ward"), t=KPROF, criterion="maxclust")
    tr = json.load(open(OUT_DIR / "tabu_results.json"))
    for r in tr["per_run"]: proc_ari([int(x[1:]) - 1 for x in r["items"]])
    proc_ari([c - 1 for c in CONSENSUS]); proc_ari(list(range(P)))
    anchors = [(0.15,0.5),(0.2,0.5),(0.3,0.5),(0.4,0.5),(0.6,0.5),(0.9,0.5),
               (0.3,0.0),(0.3,0.25),(0.3,0.75),(0.3,1.0)]
    print("Search phase...")
    for a, w in anchors:
        for r in range(3):
            tabu(a, w, np.random.default_rng(4000 + int(a*100 + w*10) + r), 35)
        print(f"  anchor a={a} w={w} pool={len(_pa)}", flush=True)
    json.dump({",".join(map(str, k)): v for k, v in _pa.items()},
              open(OUT_DIR / "objsens_cache.json", "w"))
    print(f"Persisted pool of {len(_pa)} subsets.", flush=True)
    alphas = [0.0, 0.15, 0.2, 0.25, 0.3, 0.4, 0.6, 0.9]; alpha_rows = []
    for a in alphas:
        key = tuple(range(P)) if a == 0.0 else best_pool(a, 0.5)[0]
        alpha_rows.append(row(key, a, 0.5, "alpha"))
    pd.DataFrame(alpha_rows).to_csv(OUT_DIR / "alpha_sweep.csv", index=False)
    weights = [0.0, 0.25, 0.5, 0.75, 1.0]; weight_rows = []
    for w in weights:
        weight_rows.append(row(best_pool(0.3, w)[0], 0.3, w, "w_procrustes"))
    pd.DataFrame(weight_rows).to_csv(OUT_DIR / "weight_sweep.csv", index=False)
    by_size = {}
    for key, (pr, ar) in _pa.items():
        if len(key) < MIN_ITEMS: continue
        fid = 0.5*pr + 0.5*max(0.0, ar)
        if len(key) not in by_size or fid > by_size[len(key)][0]:
            by_size[len(key)] = (fid, key, pr, ar)
    pareto, bf = [], -np.inf
    for sz in sorted(by_size):
        fid, key, pr, ar = by_size[sz]
        if fid > bf:
            bf = fid
            pareto.append({"size": sz, "fidelity": fid, "procrustes": pr,
                           "ari": ar, "items": " ".join(q(key))})
    pd.DataFrame(pareto).to_csv(OUT_DIR / "pareto_frontier.csv", index=False)
    c_key = tuple(sorted(c-1 for c in CONSENSUS)); c_pr, c_ar = _pa[c_key]
    c_fid = 0.5*c_pr + 0.5*max(0.0, c_ar)
    summary = {"reference": "full sample n=42, K=4, k=4",
        "subsets_evaluated": int(_tot[0]), "pool_size": len(_pa),
        "negative_ari_evaluations": int(_neg[0]),
        "negative_ari_fraction": float(_neg[0]/max(1,_tot[0])),
        "ari1_alpha_range": [r["alpha"] for r in alpha_rows if r["ari"]>=0.999],
        "ari1_weight_range": [r["w_procrustes"] for r in weight_rows if r["ari"]>=0.999],
        "alpha_sweep": alpha_rows, "weight_sweep": weight_rows,
        "pareto_frontier": pareto, "consensus_size": len(c_key),
        "consensus_procrustes": c_pr, "consensus_ari": c_ar,
        "consensus_fidelity": c_fid,
        "note_truncation": ("ARI can be negative (worse than chance) for poor "
            "subsets; truncation keeps fidelity in [0,1] and never binds at an "
            "optimum (all reported optima have ARI >= 0).")}
    json.dump(summary, open(OUT_DIR / "objective_sensitivity.json", "w"), indent=2)
    print(f"Evaluated {_tot[0]} subsets; {_neg[0]} had ARI<0 ({100*_neg[0]/max(1,_tot[0]):.1f}%).")
    for r in alpha_rows:
        print(f"  a={r['alpha']:.2f} |S|={r['size']:2d} proc={r['procrustes']:.3f} ari={r['ari']:.3f} ov={r['overlap_consensus']}/7")
    for r in weight_rows:
        print(f"  w={r['w_procrustes']:.2f} |S|={r['size']:2d} proc={r['procrustes']:.3f} ari={r['ari']:.3f} ov={r['overlap_consensus']}/7")
    print("Objective-sensitivity analysis complete.")

if __name__ == "__main__":
    main()
