"""
deploy_constants.py
-------------------
Deployment constants for the seven-item rule (Table 4) and consistency checks.

Reuses the pipeline of 09_classification_rule.py (population-SD
standardisation, PCA(4) + Ward(4) reference labels, posterior exp(-d^2/2)
with equal priors, Equation (6) of the article) and checks that
  * the profile sizes and the raw seven-item profile means match Table 3;
  * the standardisation constants equal those used by the PCA step;
  * the standardised centroids equal (raw mean - mu) / sigma.
Writes results/rule_card_table4.csv (Table 4) and prints the accuracies and
the smallest maximum posteriors reported in Section 3.8.

Usage:
    python src/deploy_constants.py
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.decomposition import PCA

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from repo_io import RESULTS_DIR, load_items

np.set_printoptions(suppress=True)

SF = ["Q1", "Q4", "Q8", "Q9", "Q21", "Q32", "Q38"]
SF_IDX = [int(q[1:]) - 1 for q in SF]
NAME = {4: "Pragmatic Optimists", 2: "Committed Enthusiasts",
        3: "Operationally Cautious", 1: "Risk-Averse Sceptics"}
ORDER = ["Pragmatic Optimists", "Committed Enthusiasts",
         "Operationally Cautious", "Risk-Averse Sceptics"]
# Table 3 of the article: raw seven-item means by profile
TAB3 = {"Pragmatic Optimists": [3.14, 2.76, 2.48, 2.72, 1.76, 3.03, 2.83],
        "Committed Enthusiasts": [4.00, 4.00, 3.86, 3.86, -2.00, 4.00, 4.00],
        "Operationally Cautious": [3.00, 2.00, -2.00, 4.00, 3.00, 3.00, 3.00],
        "Risk-Averse Sceptics": [-2.00, 0.00, 1.00, -2.00, 4.00, 0.00, 0.00]}

df, item_cols = load_items()
X = df[item_cols].values.astype(float)
pilots = df["pilot"].values
n = len(X)
X7 = X[:, SF_IDX]


def ref_labels(A):
    Xs = (A - A.mean(0)) / A.std(0, ddof=0)
    Y = PCA(n_components=4).fit_transform(Xs)
    return fcluster(linkage(Y, method="ward"), t=4, criterion="maxclust")


ref = ref_labels(X)

# ---------- consistency: sizes and Table 3 raw means ----------
sizes = {NAME[c]: int((ref == c).sum()) for c in np.unique(ref)}
raw = {NAME[c]: X7[ref == c].mean(0) for c in np.unique(ref)}
print("sizes:", {k: sizes[k] for k in ORDER}, "sum", sum(sizes.values()))
t3ok = all(np.allclose(raw[p], TAB3[p], atol=0.005) for p in ORDER)
print("Table 3 raw means match (atol 0.005):", t3ok)
for p in ORDER:
    print("   ", p, "maxdiff", float(np.max(np.abs(raw[p] - np.array(TAB3[p])))))

# ---------- Table 4: standardisation constants and centroids ----------
mu = X7.mean(0)
sigma = X7.std(0, ddof=0)
print("\nsigma min:", float(sigma.min()), "-> any near zero:", bool((sigma < 1e-6).any()))
# constants identical to those the PCA step uses (40-item standardisation, 7 columns)
mu_full = X.mean(0)[SF_IDX]
sd_full = X.std(0, ddof=0)[SF_IDX]
print("constants identical to PCA step:",
      np.allclose(mu, mu_full) and np.allclose(sigma, sd_full))
Z = (X7 - mu) / np.where(sigma == 0, 1, sigma)
cent = {NAME[c]: Z[ref == c].mean(0) for c in np.unique(ref)}
# identity check: m_cj == (raw mean - mu) / sigma
iden = max(float(np.max(np.abs(cent[p] - (raw[p] - mu) / sigma))) for p in ORDER)
print("linear identity max discrepancy:", iden)

print("\n=== TABLE (Item | mu | sigma | 4 profile centroids), 3 dp ===")
print("| Item | mu | sigma | " + " | ".join(ORDER) + " |")
print("|" + "---|" * (3 + 4))
for j, it in enumerate(SF):
    row = [it, f"{mu[j]:.3f}", f"{sigma[j]:.3f}"] + [f"{cent[p][j]:.3f}" for p in ORDER]
    print("| " + " | ".join(row) + " |")

card = pd.DataFrame({"Item": SF, "mu": np.round(mu, 3), "sigma": np.round(sigma, 3)})
for p in ORDER:
    card[p] = np.round(cent[p], 3)
card.to_csv(RESULTS_DIR / "rule_card_table4.csv", index=False)
print("\nWrote results/rule_card_table4.csv")


# ---------- classifier ----------
def fit(A, lab):
    m = A.mean(0)
    s = A.std(0, ddof=0)
    s = np.where(s == 0, 1.0, s)
    Zc = (A - m) / s
    return m, s, {int(c): Zc[lab == c].mean(0) for c in np.unique(lab)}


def posterior(z, cents):
    """Equation (6): exp(-d^2/2) with equal priors."""
    cs = sorted(cents)
    d2 = np.array([np.sum((z - cents[c]) ** 2) for c in cs])
    p = np.exp(-0.5 * (d2 - d2.min()))
    p /= p.sum()
    return cs, p


def classify(x, m, s, cents):
    z = (x - m) / s
    cs, p = posterior(z, cents)
    k = int(np.argmax(p))
    return cs[k], float(p[k])


m0, s0, c0 = fit(X7, ref)


# ---------- accuracies and smallest maximum posteriors (Section 3.8) ----------
def run():
    ins = [classify(X7[i], m0, s0, c0) for i in range(n)]
    in_acc = np.mean([ins[i][0] == ref[i] for i in range(n)])
    in_minconf = min(c for _, c in ins)
    in_unc = sum(c < 0.5 for _, c in ins)
    # leave-one-respondent-out, refitting the constants in every fold
    lp, lt, lc = [], [], []
    for i in range(n):
        keep = np.arange(n) != i
        m, s, c = fit(X7[keep], ref[keep])
        lab, conf = classify(X7[i], m, s, c)
        lp.append(lab)
        lt.append(int(ref[i]))
        lc.append(conf)
    loro_acc = np.mean([lp[i] == lt[i] for i in range(n)])
    loro_minconf = min(lc)
    loro_unc = sum(c < 0.5 for c in lc)
    # leave-one-pilot-case-out
    pp, pt, per = [], [], []
    for pil in pd.unique(pilots):
        keep = pilots != pil
        m, s, c = fit(X7[keep], ref[keep])
        idx = np.where(~keep)[0]
        pr = [classify(X7[i], m, s, c)[0] for i in idx]
        tr = [int(ref[i]) for i in idx]
        per.append((str(pil), len(idx), float(np.mean([a == b for a, b in zip(pr, tr)])),
                    int(len(np.unique(ref[keep])))))
        pp += pr
        pt += tr
    lopo_acc = np.mean([a == b for a, b in zip(pp, pt)])
    return dict(in_acc=in_acc, in_minconf=in_minconf, in_unc=in_unc,
                loro_acc=loro_acc, loro_minconf=loro_minconf, loro_unc=loro_unc,
                lopo_acc=lopo_acc, per=per)


r = run()
print("\n=== nearest-centroid rule, posterior of Equation (6) ===")
print(f"in-sample   acc={r['in_acc']:.3f}  min max-posterior={r['in_minconf']:.3f}  "
      f"uncertain(<0.5)={r['in_unc']}")
print(f"LORO        acc={r['loro_acc']:.3f}  min max-posterior={r['loro_minconf']:.3f}  "
      f"uncertain(<0.5)={r['loro_unc']}")
print(f"LOPO pooled acc={r['lopo_acc']:.3f}")
for pil, k, a, tc in r["per"]:
    print(f"    LOPO hold '{pil}': n={k} acc={a:.3f} trained_classes={tc}")
