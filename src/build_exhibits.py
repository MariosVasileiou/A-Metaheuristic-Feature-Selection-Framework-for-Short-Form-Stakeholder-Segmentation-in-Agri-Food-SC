"""
build_exhibits.py
-----------------
Builds exhibits/: the numbers behind every figure and table of the article,
one or more CSV files per exhibit, named with the numbering of the article
(fig01_... to fig19_..., table1_... to table4_..., appendixA1_... and so on).
Everything is derived from results/ and the de-identified data in data/.

Run it after the analysis stages (run_all.py does this automatically).
Figure 14 is a schematic of the deployment workflow and has no data file;
Tables E1 and F1 are a specification and a nomenclature and have none either.

Usage:
    python src/build_exhibits.py
"""
import json
import os
import shutil
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from repo_io import RESULTS_DIR, ROOT, load_demographics, load_items

EXH = ROOT / "exhibits"
R = RESULTS_DIR
CONSENSUS = ["Q1", "Q4", "Q8", "Q9", "Q21", "Q32", "Q38"]
PROFILE_NAMES = {4: "Pragmatic Optimists", 2: "Committed Enthusiasts",
                 3: "Operationally Cautious", 1: "Risk-Averse Sceptics"}
CM_ORDER = ["Risk-Averse Sceptics", "Committed Enthusiasts",
            "Operationally Cautious", "Pragmatic Optimists"]
W_PROCRUSTES = 0.5   # weight on the Procrustes term, as in 02_tabu_search.py


def copy(src_name, dst_name):
    shutil.copyfile(R / src_name, EXH / dst_name)


def counts(df, fields):
    rows = []
    for field in fields:
        vc = df[field].value_counts()
        if field in ("pilot", "age_band"):
            vc = vc.sort_index()
        rows += [{"field": field, "category": k, "n": int(v)} for k, v in vc.items()]
    return pd.DataFrame(rows)


def last_improvement_median(histories):
    """Median over runs of the last iteration at which the best objective improved."""
    last = []
    for _, g in histories.groupby("run"):
        ob, it = g["obj_best"].values, g["iter"].values
        imp = [it[t] for t in range(1, len(ob)) if ob[t] > ob[t - 1]]
        last.append(imp[-1] if imp else 0)
    return float(np.median(last))


def main():
    if EXH.exists():
        shutil.rmtree(EXH)
    EXH.mkdir()
    items_df, items = load_items()
    X = items_df[items].values
    demo = load_demographics()

    # ---------------- Figures ----------------
    # Figures 1 and 2: sample composition (coarsened public-release fields)
    counts(demo, ["country", "pilot"]).to_csv(EXH / "fig01_sample_country_pilot.csv", index=False)
    counts(demo, ["org_type", "age_band"]).to_csv(EXH / "fig02_sample_orgtype_age.csv", index=False)

    # Figure 3: scree
    ps = json.load(open(R / "pca_stats.json"))
    pd.DataFrame({"component": range(1, len(ps["evr_top10"]) + 1),
                  "explained_variance_ratio": ps["evr_top10"],
                  "cumulative": ps["cum_top10"]}).to_csv(EXH / "fig03_scree.csv", index=False)

    # Figure 4: principal-component scores by profile
    Y = np.load(R / "Y_full.npy")
    lab = np.load(R / "profiles_ref.npy")
    f4 = pd.DataFrame(Y, columns=[f"PC{i + 1}" for i in range(Y.shape[1])])
    f4.insert(0, "profile", [PROFILE_NAMES[int(c)] for c in lab])
    f4.insert(0, "respondent_id", items_df["respondent_id"].values)
    f4.to_csv(EXH / "fig04_pc_scores_by_profile.csv", index=False)

    copy("histories.csv", "fig05_convergence_histories.csv")
    copy("item_frequencies.csv", "fig06_item_selection_frequencies.csv")
    copy("frontier.csv", "fig07_fidelity_size_frontier.csv")
    copy("profile_sf_means.csv", "fig08_profile_shortform_means.csv")

    # Figures 9 and 10: repeated subsampling
    per = pd.read_csv(R / "subsampling_per_run.csv")
    ov = per["overlap_consensus"].value_counts().reindex(range(8), fill_value=0)
    pd.DataFrame({"items_shared_with_consensus": ov.index, "subsamples": ov.values}).to_csv(
        EXH / "fig09_subsampling_overlap.csv", index=False)
    copy("subsampling_item_frequencies.csv", "fig09_subsampling_item_frequencies.csv")
    per[["seed", "size", "proc", "ari", "fidelity", "overlap_consensus", "items"]].to_csv(
        EXH / "fig10_subsampling_per_run.csv", index=False)

    # Figure 11: selection procedures at seven items, ordered as in Table E2
    ab = pd.read_csv(R / "alternative_baselines.csv").sort_values("objective", ascending=False)
    ab[["method", "type", "objective", "procrustes", "ari"]].to_csv(
        EXH / "fig11_selection_procedures.csv", index=False)

    copy("cluster_validity.csv", "fig12_cluster_validity.csv")
    copy("loro_stability.csv", "fig13_loro_stability.csv")
    copy("lopo_stability.csv", "fig13_lopo_stability.csv")

    # Figure 15: classification rule under leave-one-respondent-out
    cl = pd.read_csv(R / "classification_loro.csv")
    copy("classification_loro.csv", "fig15_classification_loro.csv")
    cm = pd.crosstab(pd.Categorical(cl["reference_profile"], CM_ORDER),
                     pd.Categorical(cl["predicted_profile"], CM_ORDER), dropna=False)
    cm.index.name, cm.columns.name = "reference_profile", None
    cm.to_csv(EXH / "fig15_confusion_matrix.csv")
    correct = cl["reference_profile"] == cl["predicted_profile"]
    rows = []
    for lo, hi, label in [(0.5, 0.7, "0.5-0.7"), (0.7, 0.9, "0.7-0.9"), (0.9, 1.01, "0.9-1.0")]:
        m = (cl["max_posterior"] >= lo) & (cl["max_posterior"] < hi)
        rows.append({"max_posterior_bin": label, "n": int(m.sum()),
                     "accuracy": float(correct[m].mean()) if m.any() else np.nan})
    pd.DataFrame(rows).to_csv(EXH / "fig15_accuracy_by_confidence.csv", index=False)

    # Figure 16: ordinal treatments
    o = json.load(open(R / "measurement_robustness.json"))["ordinal_robustness"]
    pd.DataFrame({"variant": ["Pearson (reference)", "Spearman correlation matrix",
                              "Kendall correlation matrix", "Rank-transformed items"],
                  "ari_vs_pearson_reference": [1.0, o["ari_spearman_corrmatrix_vs_pearson"],
                                               o["ari_kendall_corrmatrix_vs_pearson"],
                                               o["ari_rank_transform_variant"]]}).to_csv(
        EXH / "fig16_ordinal_robustness.csv", index=False)

    # Figure 17: respondent quality
    copy("respondent_quality.csv", "fig17_respondent_quality.csv")
    _, n_identical = np.unique(X, axis=0, return_counts=True)
    group_sizes = sorted([int(c) for c in n_identical if c > 1], reverse=True)
    pd.DataFrame({"group": range(1, len(group_sizes) + 1), "records_in_group": group_sizes}).to_csv(
        EXH / "fig17_identical_response_groups.csv", index=False)

    copy("alpha_sweep.csv", "fig18_alpha_sweep.csv")
    copy("weight_sweep.csv", "fig18_weight_sweep.csv")
    copy("pareto_frontier.csv", "fig19_pareto_frontier.csv")

    # ---------------- Tables ----------------
    # Table 1: parameter settings, read back from the stage outputs
    cfg = json.load(open(R / "tabu_results.json"))["config"]
    hist = pd.read_csv(R / "histories.csv")
    var_k = 100 * ps["variance_explained_k"]
    pd.DataFrame([
        ("Number of items (p)", X.shape[1], "Full instrument"),
        ("Number of PCA components (K)", cfg["K_COMPONENTS"],
         f"Scree criterion; explains {var_k:.1f}% of full-instrument variance"),
        ("Number of profiles (k)", cfg["K_PROFILES"],
         "Empirically supported by Ward dendrogram cut and PCA structure"),
        ("Clustering algorithm", "Ward hierarchical", "Standard for PC-space segmentation"),
        ("Penalty coefficient (alpha)", cfg["ALPHA"], "Targets short forms of 7 to 10 items"),
        ("Weight on Procrustes term (w)", W_PROCRUSTES,
         "Equal weighting; examined by sensitivity analysis (Section 3.8)"),
        ("Minimum subset size (m_min)", cfg["MIN_ITEMS"], "Lower bound for four-component PCA"),
        ("Tabu tenure (tau)", f"{cfg['TABU_TENURE']} iterations", "Standard mid-range value"),
        ("Iterations per run (T_max)", cfg["N_ITER"],
         f"Allows late refinements (last improvement at median iteration "
         f"{last_improvement_median(hist):.0f})"),
        ("Number of independent runs", cfg["N_RUNS"], "Statistical stability"),
        ("Initialisation", "Bernoulli(0.5)", "Uniform over subset space"),
        ("Aspiration criterion", "Global best", "Standard"),
    ], columns=["parameter", "value", "rationale"]).to_csv(
        EXH / "table1_parameter_settings.csv", index=False)

    # Table 2: the 15 most frequently selected items
    fr = pd.read_csv(R / "item_frequencies.csv").head(15).copy()
    fr.insert(0, "rank", range(1, len(fr) + 1))
    fr["pct_best_runs"] = fr["pct_best_runs"].round(1)
    fr["pct_all_iters"] = fr["pct_all_iters"].round(1)
    fr.to_csv(EXH / "table2_item_selection_frequencies.csv", index=False)

    copy("profile_sf_means.csv", "table3_profile_shortform_means.csv")
    copy("rule_card_table4.csv", "table4_deployment_constants.csv")
    copy("profile_all_means.csv", "appendixA1_profile_means_all_items.csv")
    pd.read_csv(R / "frontier.csv").round(3).to_csv(
        EXH / "appendixB1_fidelity_size_frontier.csv", index=False)

    # Table C1: per-run summary of the 30 Tabu Search runs
    tr = json.load(open(R / "tabu_results.json"))
    pd.DataFrame([{"run": r["run_id"], "objective": round(r["obj_best"], 3),
                   "size": r["size"], "procrustes": round(r["proc"], 3),
                   "ari": round(r["ari"], 3), "items": ", ".join(r["items"])}
                  for r in tr["per_run"]]).to_csv(EXH / "appendixC1_per_run_summary.csv",
                                                   index=False)

    # Tables E2 and E3: alternative selection procedures at seven items
    # (values as stored by stage 08, to four decimals)
    ab[["method", "type", "objective", "procrustes", "ari", "overlap_consensus", "items"]].to_csv(
        EXH / "appendixE2_fixed_cardinality.csv", index=False)
    alt = ab[ab["type"] != "metaheuristic"]
    n_sel = {}
    for its in alt["items"]:
        for it in its.split():
            n_sel[it] = n_sel.get(it, 0) + 1
    freq = pd.read_csv(R / "item_frequencies.csv")
    top11 = set(freq.sort_values("pct_best_runs", ascending=False).head(11)["item"])
    e3 = pd.DataFrame([{"item": k, "procedures_selecting": v, "of": len(alt),
                        "in_consensus": "Yes" if k in CONSENSUS else "No",
                        "among_eleven_most_frequent_tabu_items": "Yes" if k in top11 else "No"}
                       for k, v in n_sel.items() if v >= 2])
    e3["_n"] = e3["item"].str[1:].astype(int)
    e3 = e3.sort_values(["procedures_selecting", "_n"], ascending=[False, True]).drop(columns="_n")
    e3.to_csv(EXH / "appendixE3_item_agreement.csv", index=False)

    print(f"Wrote {len(list(EXH.iterdir()))} exhibit files to {EXH}")


if __name__ == "__main__":
    main()
