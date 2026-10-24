"""
09_classification_rule.py
-------------------------
Stage 9 of the pipeline (Section 3.8): a deployable classification procedure
for the seven-item short form, and leave-one-respondent-out and
leave-one-pilot-case-out validation of it.

So that a new respondent can be classified without re-running PCA and Ward
clustering, this script produces a self-contained "rule card":

    * fixed per-item standardisation constants (mean, SD) for the seven items,
      estimated on the full 42-respondent sample;
    * seven-item profile centroids in standardised space;
    * a nearest-centroid decision rule with a softmax confidence score and an
      "uncertain" flag (max posterior < 0.5).

The rule is then validated as a classifier that must reproduce the
full-instrument reference label of a held-out respondent:

    * leave-one-respondent-out (LORO), all 42 and distinct patterns only;
    * leave-one-pilot-case-out (LOPO), holding out each pilot's respondents.

For each held-out unit, standardisation constants and centroids are re-estimated
on the training respondents (using only the seven items and the fixed reference
labels), and the held-out respondent is classified. This is an internal
generalisation check; genuine external validation on a new sample is still
required and is stated as such in the article.

Outputs:
    results/classification_rule.json      the deployable rule card + validation
    results/shortform_centroids.csv       standardised 7-item profile centroids
    results/shortform_standardisation.csv per-item mean/SD constants
    results/classification_loro.csv       leave-one-respondent-out prediction
                                          for every respondent (Figure 15)

Usage:
    python src/09_classification_rule.py
"""

import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.decomposition import PCA

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from repo_io import RESULTS_DIR, load_items

OUT_DIR = RESULTS_DIR
SF = ["Q1", "Q4", "Q8", "Q9", "Q21", "Q32", "Q38"]
SF_IDX = [int(q[1:]) - 1 for q in SF]
K, KPROF = 4, 4
PROFILE_NAMES = {4: "Pragmatic Optimists", 2: "Committed Enthusiasts",
                 3: "Operationally Cautious", 1: "Risk-Averse Sceptics"}


def reference_labels(X_all):
    """Full-instrument PCA(4)+Ward(4) reference labels on the full sample."""
    Xs = (X_all - X_all.mean(0)) / X_all.std(0, ddof=0)
    Y = PCA(n_components=K).fit_transform(Xs)
    return fcluster(linkage(Y, method="ward"), t=KPROF, criterion="maxclust")


def fit_rule(X7, labels):
    """Standardisation constants + centroids from 7-item raw data and labels."""
    mean = X7.mean(0)
    sd = X7.std(0, ddof=0)
    sd = np.where(sd == 0, 1.0, sd)
    Z = (X7 - mean) / sd
    classes = np.unique(labels)
    centroids = {int(c): Z[labels == c].mean(0) for c in classes}
    return mean, sd, centroids


def classify(x7, mean, sd, centroids):
    """Nearest-centroid label + softmax posterior + uncertain flag."""
    z = (x7 - mean) / sd
    cs = sorted(centroids)
    d2 = np.array([np.sum((z - centroids[c]) ** 2) for c in cs])
    post = np.exp(-0.5 * (d2 - d2.min()))
    post = post / post.sum()
    k = int(np.argmax(post))
    return cs[k], float(post[k]), bool(post[k] < 0.5), dict(zip(cs, post))


def main():
    OUT_DIR.mkdir(exist_ok=True)
    df, item_cols = load_items()
    X = df[item_cols].values
    pilots = df["pilot"].values
    n = len(X)
    ref = reference_labels(X)
    X7 = X[:, SF_IDX]

    # ---- Deployable rule card on the full sample ----
    mean, sd, centroids = fit_rule(X7, ref)
    std_df = pd.DataFrame({"item": SF, "mean": mean, "sd": sd})
    std_df.to_csv(OUT_DIR / "shortform_standardisation.csv", index=False)
    cent_df = pd.DataFrame(
        {PROFILE_NAMES[c]: centroids[c] for c in sorted(centroids)},
        index=SF).T
    cent_df.to_csv(OUT_DIR / "shortform_centroids.csv")
    print("Deployable rule card (full-sample centroids, standardised units):")
    print(cent_df.round(2).to_string())

    # In-sample check
    in_cls = [classify(x, mean, sd, centroids) for x in X7]
    in_pred = np.array([c[0] for c in in_cls])
    in_min_post = float(min(c[1] for c in in_cls))
    in_acc = float((in_pred == ref).mean())
    print(f"\nIn-sample nearest-centroid accuracy vs reference: {in_acc:.3f}")

    # ---- LORO ----
    loro_pred, loro_true, loro_conf, loro_unc = [], [], [], []
    for i in range(n):
        keep = np.arange(n) != i
        m, s, c = fit_rule(X7[keep], ref[keep])
        lab, conf, unc, _ = classify(X7[i], m, s, c)
        loro_pred.append(lab)
        loro_true.append(int(ref[i]))
        loro_conf.append(conf)
        loro_unc.append(unc)
    loro_pred = np.array(loro_pred)
    loro_true = np.array(loro_true)
    loro_acc = float((loro_pred == loro_true).mean())
    pd.DataFrame({
        "respondent_id": df["respondent_id"].values,
        "pilot": pilots,
        "reference_profile": [PROFILE_NAMES[c] for c in loro_true],
        "predicted_profile": [PROFILE_NAMES[c] for c in loro_pred],
        "max_posterior": loro_conf,
        "uncertain": loro_unc,
    }).to_csv(OUT_DIR / "classification_loro.csv", index=False)

    # LORO on distinct patterns only
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
    reps = sorted({find(i) for i in range(n)})
    dp_correct = sum(loro_pred[i] == loro_true[i] for i in reps)
    loro_distinct_acc = dp_correct / len(reps)

    # ---- LOPO ----
    lopo_rows, lopo_pred, lopo_true = [], [], []
    for pilot in pd.unique(pilots):
        keep = pilots != pilot
        m, s, c = fit_rule(X7[keep], ref[keep])
        idx = np.where(~keep)[0]
        preds = [classify(X7[i], m, s, c)[0] for i in idx]
        trues = [int(ref[i]) for i in idx]
        acc = float(np.mean([p == t for p, t in zip(preds, trues)]))
        lopo_pred += preds
        lopo_true += trues
        lopo_rows.append({"pilot": str(pilot), "n": int(len(idx)),
                          "accuracy": acc,
                          "trained_classes": int(len(np.unique(ref[keep])))})
        print(f"LOPO hold out '{pilot}': acc={acc:.3f}")
    lopo_acc = float(np.mean([p == t for p, t in zip(lopo_pred, lopo_true)]))

    # ---- Confidence calibration (LORO) ----
    conf = np.array(loro_conf)
    correct = (loro_pred == loro_true).astype(int)
    bins = [0.0, 0.5, 0.7, 0.9, 1.01]
    calib = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (conf >= lo) & (conf < hi)
        if mask.sum():
            calib.append({"conf_bin": f"[{lo:.1f},{hi:.1f})",
                          "n": int(mask.sum()),
                          "accuracy": float(correct[mask].mean())})

    out = {
        "short_form": SF,
        "profile_names": {str(k): v for k, v in PROFILE_NAMES.items()},
        "standardisation": std_df.to_dict(orient="list"),
        "centroids_standardised": {PROFILE_NAMES[c]:
                                   [round(v, 4) for v in centroids[c]]
                                   for c in sorted(centroids)},
        "decision_rule": ("Standardise the 7 responses with the fixed "
                          "mean/SD constants; assign to the nearest centroid "
                          "in Euclidean distance; posterior = softmax(-0.5*d^2); "
                          "flag 'uncertain' when max posterior < 0.5."),
        "in_sample_accuracy": in_acc,
        "loro_accuracy_all": loro_acc,
        "loro_accuracy_distinct_patterns": loro_distinct_acc,
        "loro_fraction_flagged_uncertain": float(np.mean(loro_unc)),
        "in_sample_min_max_posterior": in_min_post,
        "loro_min_max_posterior": float(np.min(loro_conf)),
        "lopo_accuracy_pooled": lopo_acc,
        "lopo_per_pilot": lopo_rows,
        "confidence_calibration_loro": calib,
        "caveat": ("Many response vectors are duplicated, so LORO is optimistic; "
                   "LOPO removes an entire pilot (and, for the pilot that holds "
                   "both Risk-Averse Sceptics records, the only sceptics), giving "
                   "a stricter internal check. Genuine external validation on a "
                   "new independent sample is still required."),
    }
    json.dump(out, open(OUT_DIR / "classification_rule.json", "w"), indent=2)
    print(f"\nLORO accuracy (all)      : {loro_acc:.3f}")
    print(f"LORO accuracy (distinct) : {loro_distinct_acc:.3f}")
    print(f"LOPO accuracy (pooled)   : {lopo_acc:.3f}")
    print(f"Flagged uncertain (LORO) : {100*np.mean(loro_unc):.1f}%")
    print(f"Smallest max posterior   : in-sample {in_min_post:.3f}, "
          f"LORO {np.min(loro_conf):.3f}")

    print("Classification-rule analysis complete.")


if __name__ == "__main__":
    main()
