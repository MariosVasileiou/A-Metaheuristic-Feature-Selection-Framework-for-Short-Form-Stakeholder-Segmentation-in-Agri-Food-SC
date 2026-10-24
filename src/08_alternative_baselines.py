"""
08_alternative_baselines.py
---------------------------
Stage 8 of the pipeline (Section 3.7 and Appendix E): comparison of Tabu
Search with eight alternative item-selection procedures (wrapper, filter and
embedded methods), each scored on the same composite fidelity objective as
the main pipeline.

Wrapper baselines (optimise the same composite fidelity objective):
    - greedy forward selection
    - backward elimination
    - sequential floating forward selection (SFFS)
    - simulated annealing
    - genetic algorithm

Filter/embedded baselines (rank items by an external criterion, then take the
top-|S|, then score on the same fidelity objective):
    - sparse PCA loadings
    - L1-penalised multinomial logistic regression on the reference labels
    - random-forest importance on the reference labels

Every method is evaluated at the matched size |S| = 7 (the consensus size).
Simulated annealing and the genetic algorithm are therefore run in
size-constrained form (swap moves and offspring repair respectively). Their
unconstrained best-found solutions are additionally recorded in the best_size /
best_items columns for the variable-cardinality comparison. Scoring uses the
objective of the article (Procrustes + ARI, alpha = 0.3, w = 0.5). Results are
written incrementally, so partial progress is never lost.

Sequential floating selection stops as soon as a subset recurs; without this
rule the conditional backward step undoes the forward step on these data and
the search cycles indefinitely.

In the size-constrained genetic algorithm used for the matched-size comparison,
crossover draws seven items from the union of the two parents and mutation
swaps one retained item for one excluded item (probability 0.05 per child);
the unconstrained variant uses single-point crossover and bit-flip mutation.

Outputs:
    results/alternative_baselines.json
    results/alternative_baselines.csv   (Figure 11; Tables E2 and E3)

Usage:
    python src/08_alternative_baselines.py
"""

import json
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.decomposition import SparsePCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from repo_io import RESULTS_DIR, load_items
from sf_common import pca_ward_labels, score_subset

OUT_DIR = RESULTS_DIR
K, KPROF, MIN_ITEMS, P = 4, 4, 5, 40
TARGET = 7
CONSENSUS = {1, 4, 8, 9, 21, 32, 38}

_Xs = _Yf = _ref = None
_results = {}


def sc(sel):
    _, m = score_subset(sel, _Xs, _Yf, _ref, K=K, k=KPROF, alpha=0.3, w=0.5,
                        min_items=MIN_ITEMS)
    return m


def q(sel):
    return [f"Q{j + 1}" for j in sorted(sel)]


def record(name, sel, tag="wrapper", best_sel=None):
    m = sc(sel)
    entry = {"method": name, "type": tag, "size": len(sel),
             "procrustes": round(m["proc"], 4), "ari": round(m["ari"], 4),
             "fidelity": round(m["fidelity"], 4),
             "overlap_consensus": len({j + 1 for j in sel} & CONSENSUS),
             "items": " ".join(q(sel)),
             "objective": round(m["obj"], 4)}
    if best_sel is not None and len(best_sel) != len(sel):
        mb = sc(best_sel)
        entry["best_size"] = len(best_sel)
        entry["best_ari"] = round(mb["ari"], 4)
        entry["best_procrustes"] = round(mb["proc"], 4)
        entry["best_items"] = " ".join(q(best_sel))
    _results[name] = entry
    json.dump(list(_results.values()),
              open(OUT_DIR / "alternative_baselines.json", "w"), indent=2)
    extra = ""
    if "best_size" in entry:
        extra = (f" | best|S|={entry['best_size']} "
                 f"ARI={entry['best_ari']:.3f}")
    print(f"  {name:36s} |S|={len(sel):2d} proc={m['proc']:.3f} "
          f"ari={m['ari']:.3f} ov={entry['overlap_consensus']}/7{extra}",
          flush=True)


# ---------- wrapper baselines ----------
def greedy_forward(target):
    """Greedy forward selection on the composite objective.

    The objective is undefined below MIN_ITEMS items, so all candidates of a
    smaller subset receive the same constant score and the first
    MIN_ITEMS - 1 items enter in item order (Q1 to Q4). From then on, each
    step adds the item that gives the largest objective.
    """
    sel = []
    remaining = set(range(P))
    while len(sel) < target:
        best_j, best_val = None, -np.inf
        for j in remaining:
            trial = sel + [j]
            if len(trial) < MIN_ITEMS:
                # objective undefined: constant score, so ties go to item order
                val = -len(trial)
            else:
                val = sc(trial)["obj"]
            if val > best_val:
                best_val, best_j = val, j
        sel.append(best_j)
        remaining.discard(best_j)
    return sel


def backward_elim(target):
    sel = list(range(P))
    while len(sel) > target:
        best_j, best_val = None, -np.inf
        for j in sel:
            trial = [x for x in sel if x != j]
            if len(trial) < MIN_ITEMS:
                continue
            val = sc(trial)["obj"]
            if val > best_val:
                best_val, best_j = val, j
        sel.remove(best_j)
    return sel


def sffs(target, max_size=14):
    """Sequential floating forward selection on the composite objective.

    Each step adds the best item and then removes items while a removal
    improves the objective. Because the objective contains the size penalty,
    the removal can undo the addition and the search can return to a subset it
    has already visited; the search therefore stops as soon as a subset recurs
    (without this rule it cycles indefinitely on these data). The best subset
    found at each size is kept. As in greedy forward selection, the first
    MIN_ITEMS - 1 items enter in item order, because the objective is undefined
    below MIN_ITEMS items.
    """
    sel = []
    remaining = set(range(P))
    best_by_size = {}
    visited = set()
    while len(sel) < max_size:
        # forward step
        best_j, best_val = None, -np.inf
        for j in remaining:
            trial = sel + [j]
            val = sc(trial)["obj"] if len(trial) >= MIN_ITEMS else -len(trial)
            if val > best_val:
                best_val, best_j = val, j
        sel.append(best_j)
        remaining.discard(best_j)
        # conditional backward steps
        improved = True
        while improved and len(sel) > MIN_ITEMS:
            improved = False
            cur = sc(sel)["obj"]
            best_r, best_rv = None, cur
            for j in sel:
                trial = [x for x in sel if x != j]
                if len(trial) < MIN_ITEMS:
                    continue
                v = sc(trial)["obj"]
                if v > best_rv:
                    best_rv, best_r = v, j
            if best_r is not None:
                sel.remove(best_r)
                remaining.add(best_r)
                improved = True
        if len(sel) >= MIN_ITEMS:
            o = sc(sel)["obj"]
            if len(sel) not in best_by_size or o > best_by_size[len(sel)][0]:
                best_by_size[len(sel)] = (o, list(sel))
        state = tuple(sorted(sel))
        if state in visited:
            break
        visited.add(state)
    # return size-target subset and the best-objective subset overall
    target_sel = best_by_size.get(target, (None, sel))[1]
    best_overall = max(best_by_size.values(), key=lambda t: t[0])[1]
    return target_sel, best_overall


def simulated_annealing(rng, iters=1500, T0=0.05, cool=0.995):
    s = (rng.random(P) < 0.25).astype(int)
    while s.sum() < MIN_ITEMS:
        s[rng.integers(P)] = 1
    cur = sc(list(np.where(s == 1)[0]))["obj"]
    best_s, best_o = s.copy(), cur
    T = T0
    for _ in range(iters):
        j = rng.integers(P)
        t = s.copy()
        t[j] = 1 - t[j]
        if t.sum() < MIN_ITEMS:
            continue
        o = sc(list(np.where(t == 1)[0]))["obj"]
        if o >= cur or rng.random() < np.exp((o - cur) / max(T, 1e-9)):
            s, cur = t, o
            if o > best_o:
                best_s, best_o = s.copy(), o
        T *= cool
    return list(np.where(best_s == 1)[0])


def genetic_algorithm(rng, pop=30, gens=40, pm=0.05):
    def fit(vec):
        return sc(list(np.where(vec == 1)[0]))["obj"]
    P0 = []
    while len(P0) < pop:
        v = (rng.random(P) < 0.25).astype(int)
        if v.sum() >= MIN_ITEMS:
            P0.append(v)
    P0 = np.array(P0)
    fits = np.array([fit(v) for v in P0])
    best_v, best_f = P0[fits.argmax()].copy(), fits.max()
    for _ in range(gens):
        # tournament selection
        newP = []
        for _ in range(pop):
            i, j = rng.integers(pop), rng.integers(pop)
            p1 = P0[i] if fits[i] >= fits[j] else P0[j]
            i, j = rng.integers(pop), rng.integers(pop)
            p2 = P0[i] if fits[i] >= fits[j] else P0[j]
            cx = rng.integers(1, P)
            child = np.concatenate([p1[:cx], p2[cx:]])
            mut = rng.random(P) < pm
            child = np.where(mut, 1 - child, child)
            if child.sum() < MIN_ITEMS:
                child[rng.choice(P, MIN_ITEMS, replace=False)] = 1
            newP.append(child)
        P0 = np.array(newP)
        fits = np.array([fit(v) for v in P0])
        if fits.max() > best_f:
            best_v, best_f = P0[fits.argmax()].copy(), fits.max()
    return list(np.where(best_v == 1)[0])


def _swap_neighbour(sel, rng):
    """Exchange one retained item for one excluded item, preserving |S|."""
    sel = set(sel)
    out = rng.choice(sorted(sel))
    ins = rng.choice(sorted(set(range(P)) - sel))
    sel.discard(out)
    sel.add(ins)
    return sorted(sel)


def simulated_annealing_fixed(rng, target, iters=1500, T0=0.05, cool=0.995):
    """Simulated annealing constrained to exactly `target` items (swap moves)."""
    s = sorted(rng.choice(P, target, replace=False))
    cur = sc(s)["obj"]
    best_s, best_o = list(s), cur
    T = T0
    for _ in range(iters):
        t = _swap_neighbour(s, rng)
        o = sc(t)["obj"]
        if o >= cur or rng.random() < np.exp((o - cur) / max(T, 1e-9)):
            s, cur = t, o
            if o > best_o:
                best_s, best_o = list(t), o
        T *= cool
    return best_s


def genetic_algorithm_fixed(rng, target, pop=30, gens=40, pm=0.05):
    """Genetic algorithm with offspring repaired to exactly `target` items."""
    pool = [sorted(rng.choice(P, target, replace=False)) for _ in range(pop)]
    fits = np.array([sc(v)["obj"] for v in pool])
    best_v, best_f = list(pool[int(fits.argmax())]), fits.max()
    for _ in range(gens):
        new = []
        for _ in range(pop):
            i, j = rng.integers(pop), rng.integers(pop)
            p1 = pool[i] if fits[i] >= fits[j] else pool[j]
            i, j = rng.integers(pop), rng.integers(pop)
            p2 = pool[i] if fits[i] >= fits[j] else pool[j]
            union = list(dict.fromkeys(list(p1) + list(p2)))
            child = sorted(rng.choice(union, min(target, len(union)),
                                      replace=False))
            while len(child) < target:
                c = int(rng.integers(P))
                if c not in child:
                    child = sorted(child + [c])
            if rng.random() < pm:
                child = _swap_neighbour(child, rng)
            new.append(child)
        pool = new
        fits = np.array([sc(v)["obj"] for v in pool])
        if fits.max() > best_f:
            best_v, best_f = list(pool[int(fits.argmax())]), fits.max()
    return best_v


# ---------- filter/embedded baselines ----------
def sparse_pca_rank():
    sp = SparsePCA(n_components=K, alpha=1.0, random_state=0).fit(_Xs)
    imp = np.abs(sp.components_).sum(axis=0)
    return np.argsort(-imp)


def l1_logistic_rank():
    # With the saga solver and more than two classes the fit is multinomial by
    # default; the multi_class argument was removed in scikit-learn 1.8.
    # scikit-learn 1.8 also deprecated the penalty argument: the L1 penalty is
    # then requested with l1_ratio=1, and with penalty="l1" in earlier versions.
    if LogisticRegression().get_params().get("penalty", "deprecated") == "deprecated":
        clf = LogisticRegression(l1_ratio=1.0, solver="saga", C=0.5, max_iter=5000)
    else:
        clf = LogisticRegression(penalty="l1", solver="saga", C=0.5, max_iter=5000)
    clf.fit(_Xs, _ref)
    imp = np.abs(clf.coef_).sum(axis=0)
    return np.argsort(-imp)


def rf_rank():
    rf = RandomForestClassifier(n_estimators=500, random_state=0)
    rf.fit(_Xs, _ref)
    return np.argsort(-rf.feature_importances_)


def main():
    global _Xs, _Yf, _ref
    OUT_DIR.mkdir(exist_ok=True)
    df, item_cols = load_items()
    _Xs = StandardScaler().fit_transform(df[item_cols].values)
    _Yf, _ref = pca_ward_labels(_Xs, K, KPROF)

    print("Reference (Tabu Search consensus, size 7):")
    record("Tabu Search", [c - 1 for c in CONSENSUS], "metaheuristic")

    print("\nWrapper baselines:")
    t = time.time()
    record("Greedy forward", greedy_forward(TARGET))
    record("Backward elimination", backward_elim(TARGET))
    tgt, best = sffs(TARGET)
    record("Sequential floating forward (SFFS)", tgt, best_sel=best)
    record("Simulated annealing",
           simulated_annealing_fixed(np.random.default_rng(11), TARGET),
           best_sel=simulated_annealing(np.random.default_rng(11)))
    record("Genetic algorithm",
           genetic_algorithm_fixed(np.random.default_rng(12), TARGET),
           best_sel=genetic_algorithm(np.random.default_rng(12)))
    print(f"  (wrappers took {time.time()-t:.0f}s)")

    print("\nFilter / embedded baselines (top-7 by criterion):")
    record("Sparse PCA", sparse_pca_rank()[:TARGET].tolist(), "filter")
    record("L1-penalised logistic regression", l1_logistic_rank()[:TARGET].tolist(),
           "embedded")
    record("Random-forest importance", rf_rank()[:TARGET].tolist(), "embedded")

    # ---- table ----
    rows = list(_results.values())
    dfres = pd.DataFrame(rows)
    dfres.to_csv(OUT_DIR / "alternative_baselines.csv", index=False)
    print("Alternative-baselines analysis complete.")


if __name__ == "__main__":
    main()
