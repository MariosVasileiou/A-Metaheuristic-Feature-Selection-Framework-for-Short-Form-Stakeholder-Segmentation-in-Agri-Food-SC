"""
make_figures.py
---------------
Draws Figures 1 to 19 of the article at final print size (text width 6.5 in,
300 dpi) from the de-identified data in data/ and the outputs of the analysis
stages in results/. Run it after stages 01 to 10 (run_all.py does this
automatically); it takes a few seconds.

Figures 1, 2 and 13 are drawn from the coarsened public-release fields, so
they differ in presentation from the published versions, which were drawn
from the uncoarsened source file:
  * Figures 1(b) and 13(b) show the pilot codes P1 to P7 instead of the pilot
    names; the counts and the ARI values are the same;
  * Figure 2(a) pools the three organisation types that have a single
    respondent each into "Other";
  * Figure 2(b) shows age bands instead of a histogram of exact ages.
If a file data/pilot_labels_PRIVATE.csv (columns pilot,label) is present, the
pilot names are shown instead of the codes; that file is not distributed.
Figure 14 is a schematic of the deployment workflow and uses no data.

Usage:
    python src/make_figures.py

Output: figures/fig01_sample_country_pilot.png ... figures/fig19_pareto_frontier.png
"""
import json
import os
import sys

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Patch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from repo_io import DATA_DIR, FIGURES_DIR, RESULTS_DIR, load_demographics, load_items

RES = RESULTS_DIR
PILOT_LABELS_FILE = DATA_DIR / "pilot_labels_PRIVATE.csv"   # optional, not distributed

SF = ["Q1", "Q4", "Q8", "Q9", "Q21", "Q32", "Q38"]
SF_IDX = [int(q[1:]) - 1 for q in SF]
PROFILE_NAMES = {4: "Pragmatic Optimists", 2: "Committed Enthusiasts",
                 3: "Operationally Cautious", 1: "Risk-Averse Sceptics"}
ORDER = ["Pragmatic Optimists", "Committed Enthusiasts",
         "Operationally Cautious", "Risk-Averse Sceptics"]
AGE_BANDS = ["<35", "35-44", "45-54", "55+"]

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Liberation Serif", "Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 10, "axes.titlesize": 10.5, "axes.labelsize": 10,
    "xtick.labelsize": 9.5, "ytick.labelsize": 9.5, "legend.fontsize": 9,
    "axes.linewidth": 0.8, "axes.grid": True, "grid.alpha": 0.28,
    "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "axes.titlepad": 5, "legend.frameon": False, "savefig.dpi": 300,
    "xtick.major.width": 0.8, "ytick.major.width": 0.8, "axes.axisbelow": True,
    "text.hinting": "no_hinting",
})
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
DBLUE, LBLUE, MBLUE = "#184f95", "#b7d3f6", "#5598e7"
GRAY, INK = "#8c8c8a", "#1f1f1f"
PCOL = {"Pragmatic Optimists": BLUE, "Committed Enthusiasts": ORANGE,
        "Operationally Cautious": AQUA, "Risk-Averse Sceptics": YELLOW}
PMRK = {"Pragmatic Optimists": "o", "Committed Enthusiasts": "s",
        "Operationally Cautious": "^", "Risk-Averse Sceptics": "D"}
EDGE = dict(edgecolor=INK, linewidth=0.6)
W = 6.5   # text width of the journal page, inches


def save(fig, name, tight=True):
    path = FIGURES_DIR / name
    if tight:
        fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.03)
    else:
        fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"  {path.relative_to(FIGURES_DIR.parent)}")


def pilot_labels(codes):
    """Pilot codes, or the pilot names if the private label file is present."""
    if PILOT_LABELS_FILE.exists():
        names = pd.read_csv(PILOT_LABELS_FILE).set_index("pilot")["label"].to_dict()
        return [names.get(c, c) for c in codes]
    return list(codes)


def hbar(ax, s, color, title, xlabel="Respondents", sort=True):
    """Horizontal bar chart of counts; the first entry of s is drawn at the bottom."""
    if sort:
        s = s.sort_values(ascending=True)
    ax.barh(range(len(s)), s.values, color=color, height=0.7, **EDGE)
    ax.set_yticks(range(len(s)))
    ax.set_yticklabels(s.index)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.grid(axis="y", visible=False)
    for i, v in enumerate(s.values):
        ax.text(v + 0.15, i, str(int(v)), va="center", fontsize=9)
    ax.set_xlim(0, s.max() * 1.15)
    ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True))


# ---------------------------------------------------------------------
# Data and stage outputs
# ---------------------------------------------------------------------
items, item_cols = load_items()
X = items[item_cols].values.astype(float)
demo = load_demographics()
lab = np.load(RES / "profiles_ref.npy")
Y = np.load(RES / "Y_full.npy")
nm = np.array([PROFILE_NAMES[int(c)] for c in lab])
sizes = pd.Series(nm).value_counts()
evr = np.array(json.load(open(RES / "pca_stats.json"))["evr_top10"]) * 100
cum = np.cumsum(evr)

print("Drawing Figures 1 to 19:")

# ---------------- Figure 1: country and pilot case ----------------
fig, axs = plt.subplots(1, 2, figsize=(W, 2.9), constrained_layout=True)
hbar(axs[0], demo["country"].value_counts(), BLUE, "(a) Country")
pil = demo["pilot"].value_counts().sort_index()[::-1]       # P1 at the top
pil.index = pilot_labels(pil.index)
hbar(axs[1], pil, BLUE, "(b) Pilot case", sort=False)
save(fig, "fig01_sample_country_pilot.png")

# ---------------- Figure 2: organisation type and age ----------------
fig, axs = plt.subplots(1, 2, figsize=(W, 2.9), constrained_layout=True)
hbar(axs[0], demo["org_type"].value_counts(), BLUE, "(a) Organisation type")
ax = axs[1]
ages = demo["age_band"].value_counts().reindex(AGE_BANDS, fill_value=0)
ax.bar(range(len(ages)), ages.values, color=BLUE, width=0.7, **EDGE)
for i, v in enumerate(ages.values):
    ax.text(i, v + 0.2, str(int(v)), ha="center", va="bottom", fontsize=9)
ax.set_xticks(range(len(ages)))
ax.set_xticklabels([a.replace("-", "–") for a in AGE_BANDS])
ax.set_ylim(0, ages.max() * 1.15)
ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True))
ax.set_xlabel("Age band (years)"); ax.set_ylabel("Respondents"); ax.set_title("(b) Age")
ax.grid(axis="x", visible=False)
save(fig, "fig02_sample_orgtype_age.png")

# ---------------- Figure 3: scree ----------------
fig, ax = plt.subplots(figsize=(4.8, 3.0), constrained_layout=True)
ax.bar(np.arange(1, 11), evr, color=BLUE, width=0.7, label="Individual component", **EDGE)
ax.plot(np.arange(1, 11), cum, color=ORANGE, marker="o", ms=5, lw=1.8, label="Cumulative")
ax.axvline(4, color=GRAY, ls=":", lw=1.2)
ax.annotate(f"{cum[3]:.1f}% with four components", xy=(4, cum[3]), xytext=(5.0, 52),
            fontsize=9, arrowprops=dict(arrowstyle="->", lw=0.8, color=INK))
ax.set_xticks(range(1, 11)); ax.set_ylim(0, 100)
ax.set_xlabel("Principal component"); ax.set_ylabel("Explained variance (%)")
ax.legend(loc="center right", bbox_to_anchor=(1.0, 0.30)); ax.grid(axis="x", visible=False)
save(fig, "fig03_scree.png")

# ---------------- Figure 4: profiles in the PC1 x PC2 plane ----------------
fig, ax = plt.subplots(figsize=(5.2, 3.7), constrained_layout=True)
for p in ORDER:
    m = nm == p
    ax.scatter(Y[m, 0], Y[m, 1], s=46, c=PCOL[p], marker=PMRK[p], edgecolors=INK,
               linewidths=0.6, label=f"{p} ($n$ = {m.sum()})", zorder=3)
ax.axhline(0, color=GRAY, lw=0.6); ax.axvline(0, color=GRAY, lw=0.6)
ax.set_xlabel(f"PC1 ({evr[0]:.1f}% of variance)"); ax.set_ylabel(f"PC2 ({evr[1]:.1f}% of variance)")
ax.legend(loc="lower left", handletextpad=0.3, borderaxespad=0.3)
save(fig, "fig04_profiles_pc1_pc2.png")

# ---------------- Figure 5: convergence ----------------
h = pd.read_csv(RES / "histories.csv")
fig, axs = plt.subplots(1, 2, figsize=(W, 2.8), constrained_layout=True)
for col, ax, ttl, yl in [("obj_best", axs[0], "(a) Best objective so far", "Objective $F$"),
                         ("n_cur", axs[1], "(b) Current subset size", "Items in current solution")]:
    for r, g in h.groupby("run"):
        ax.plot(g["iter"], g[col], color=LBLUE, lw=0.7, alpha=0.9)
    mean = h.groupby("iter")[col].mean()
    ax.plot(mean.index, mean.values, color=DBLUE, lw=2.0)
    ax.set_xlabel("Iteration"); ax.set_ylabel(yl); ax.set_title(ttl)
axs[1].axhline(5, color=GRAY, ls="--", lw=1.0)
axs[1].text(99, 5.4, "minimum size $m_{\\min}$ = 5", ha="right", va="bottom", fontsize=8.5, color=INK)
handles = [Line2D([], [], color=LBLUE, lw=1.2, label="Individual runs"),
           Line2D([], [], color=DBLUE, lw=2.0, label="Mean of 30 runs")]
axs[0].legend(handles=handles, loc="lower right")
save(fig, "fig05_convergence.png")

# ---------------- Figure 6: item selection frequency ----------------
fr = pd.read_csv(RES / "item_frequencies.csv").sort_values("freq_best_runs", ascending=False).head(20)
fr = fr.iloc[::-1]
cols = [DBLUE if c >= 15 else (MBLUE if c >= 10 else LBLUE) for c in fr["freq_best_runs"]]
fig, ax = plt.subplots(figsize=(5.4, 4.1), constrained_layout=True)
ax.barh(range(len(fr)), fr["pct_best_runs"], color=cols, height=0.72, **EDGE)
ax.set_yticks(range(len(fr))); ax.set_yticklabels(fr["item"])
ax.axvline(50, color=INK, ls="--", lw=1.0)
ax.set_xlabel("Selection frequency in best-of-run solutions (%)"); ax.set_xlim(0, 105)
ax.grid(axis="y", visible=False)
ax.legend(handles=[Patch(facecolor=DBLUE, **EDGE, label="At least 50% of runs"),
                   Patch(facecolor=MBLUE, **EDGE, label="At least one-third of runs"),
                   Patch(facecolor=LBLUE, **EDGE, label="Fewer than one-third of runs"),
                   Line2D([], [], color=INK, ls="--", lw=1.0, label="50% threshold")],
          loc="lower right")
save(fig, "fig06_item_frequency.png")

# ---------------- Figure 7: fidelity-size frontier ----------------
fz = pd.read_csv(RES / "frontier.csv")
fig, axs = plt.subplots(1, 2, figsize=(W, 2.8), constrained_layout=True)
for key, ax, ttl, yl in [("proc", axs[0], "(a) Geometric fidelity", "Procrustes similarity"),
                         ("ari", axs[1], "(b) Profile recovery", "ARI")]:
    ax.plot(fz["size"], fz[f"random_{key}"], color=GRAY, marker="o", ms=4.5, lw=1.4,
            label="Random subsets (mean of 100)")
    ax.plot(fz["size"], fz[f"toploading_{key}"], color=ORANGE, marker="s", ms=4.5, lw=1.4,
            label="Top-loading")
    tb = fz.dropna(subset=[f"tabu_best_{key}"])
    ax.plot(tb["size"], tb[f"tabu_best_{key}"], color=BLUE, marker="D", ms=6.5, lw=1.6, mec=INK,
            mew=0.6, label="Tabu Search (best found)")
    ax.set_xlabel("Number of items"); ax.set_ylabel(yl); ax.set_title(ttl)
axs[1].axhline(1.0, color=GRAY, ls=":", lw=1.0); axs[1].set_ylim(0, 1.05)
axs[0].legend(loc="lower right")
save(fig, "fig07_fidelity_size_frontier.png")

# ---------------- Figure 8: profile signatures ----------------
Xsf = pd.DataFrame(X[:, SF_IDX], columns=SF)
Xsf["p"] = nm
means = Xsf.groupby("p").mean().loc[ORDER]
fig, ax = plt.subplots(figsize=(W, 3.1), constrained_layout=True)
wd = 0.2
for i, p in enumerate(ORDER):
    ax.bar(np.arange(7) + (i - 1.5) * wd, means.loc[p].values, width=wd, color=PCOL[p],
           label=f"{p} ($n$ = {sizes[p]})", **EDGE)
ax.axhline(0, color=INK, lw=0.8)
ax.set_xticks(range(7)); ax.set_xticklabels(SF); ax.set_ylim(-2.7, 4.5)
ax.set_xlabel("Short-form item"); ax.set_ylabel("Mean response ($-4$ to $+4$)")
ax.grid(axis="x", visible=False)
ax.legend(ncol=2, loc="lower center", bbox_to_anchor=(0.5, 1.0), columnspacing=1.2, handlelength=1.2)
save(fig, "fig08_profile_signatures.png")

# ---------------- Figures 9 and 10: repeated subsampling ----------------
bp = pd.read_csv(RES / "subsampling_per_run.csv")
bf = (pd.read_csv(RES / "subsampling_item_frequencies.csv")
      .sort_values("original_pct", ascending=False).head(15).iloc[::-1])
fig, axs = plt.subplots(1, 2, figsize=(W, 3.2), constrained_layout=True)
ax = axs[0]
cnt = bp["overlap_consensus"].value_counts().reindex(range(8), fill_value=0)
ax.bar(cnt.index, cnt.values, color=[BLUE if k < 5 else ORANGE for k in cnt.index], width=0.75, **EDGE)
mu = bp["overlap_consensus"].mean()
ax.axvline(mu, color=INK, ls="--", lw=1.0)
ax.text(mu + 0.3, cnt.max() * 0.80, f"mean = {mu:.1f}", fontsize=9, va="top")
ax.set_xticks(range(8)); ax.set_xlabel("Items shared with the consensus form")
ax.set_ylabel("Subsamples"); ax.set_title("(a) Overlap with the consensus")
ax.grid(axis="x", visible=False)
share5 = 100 * (bp["overlap_consensus"] >= 5).mean()
ax.text(6.0, 30, f"five or more\nshared: {share5:.1f}%", ha="center", va="bottom", fontsize=9)
ax = axs[1]
yy = np.arange(len(bf)); hh = 0.38
ax.barh(yy + hh / 2, bf["original_pct"], height=hh, color=BLUE, label="Main analysis (30 runs)", **EDGE)
ax.barh(yy - hh / 2, bf["subsample_pct"], height=hh, color=ORANGE, label="Subsamples (200)", **EDGE)
ax.set_yticks(yy); ax.set_yticklabels(bf["item"]); ax.axvline(50, color=INK, ls="--", lw=1.0)
ax.set_xlabel("Selection frequency (%)"); ax.set_title("(b) Selection frequency by item")
ax.set_xlim(0, 105); ax.grid(axis="y", visible=False); ax.legend(loc="lower right")
save(fig, "fig09_subsampling_overlap_items.png")

fig, ax = plt.subplots(figsize=(4.8, 2.9), constrained_layout=True)
ax.hist(bp["proc"], bins=np.arange(0.870, 0.965, 0.005), color=BLUE, rwidth=0.9, **EDGE)
ax.axvline(bp["proc"].mean(), color=INK, ls="--", lw=1.0)
ax.text(0.02, 0.95, f"mean = {bp['proc'].mean():.3f} (dashed line)", transform=ax.transAxes,
        ha="left", fontsize=9, va="top")
n_ari1 = int((bp["ari"] == 1.0).sum())
ari_note = (f"ARI = 1.000 in all {len(bp)} subsamples" if n_ari1 == len(bp)
            else f"ARI = 1.000 in {n_ari1} of {len(bp)} subsamples")
ax.text(0.02, 0.86, ari_note, transform=ax.transAxes, ha="left", fontsize=9, va="top")
ax.set_xlabel("Procrustes similarity"); ax.set_ylabel("Subsamples"); ax.grid(axis="x", visible=False)
save(fig, "fig10_subsampling_procrustes.png")

# ---------------- Figure 11: selection procedures at seven items ----------------
ab = pd.read_csv(RES / "alternative_baselines.csv").sort_values("objective", ascending=False)
fam = {"metaheuristic": BLUE, "wrapper": AQUA, "filter": ORANGE, "embedded": ORANGE}
short_name = {"L1-penalised logistic regression": "L1 logistic regression"}   # label as printed in the figure
fig, axs = plt.subplots(1, 2, figsize=(W, 3.1), constrained_layout=True, sharey=True)
yy = np.arange(len(ab))[::-1]
for ax, col, ttl, xl in [(axs[0], "ari", "(a) Profile recovery", "ARI"),
                         (axs[1], "procrustes", "(b) Geometric fidelity", "Procrustes similarity")]:
    ax.barh(yy, ab[col], color=[fam[t] for t in ab["type"]], height=0.7, **EDGE)
    ax.set_xlim(0, 1.0); ax.set_xlabel(xl); ax.set_title(ttl); ax.grid(axis="y", visible=False)
axs[0].set_yticks(yy); axs[0].set_yticklabels([short_name.get(m, m) for m in ab["method"]])
fig.legend(handles=[Patch(facecolor=BLUE, **EDGE, label="Tabu Search"),
                    Patch(facecolor=AQUA, **EDGE, label="Other wrapper searches"),
                    Patch(facecolor=ORANGE, **EDGE, label="Embedded and filter procedures")],
           loc="outside lower center", ncol=3, handlelength=1.2, columnspacing=1.4)
save(fig, "fig11_selection_procedures.png")

# ---------------- Figure 12: internal cluster validity ----------------
cv = pd.read_csv(RES / "cluster_validity.csv")
fig, ax = plt.subplots(figsize=(4.8, 3.0), constrained_layout=True)
ax.plot(cv["k"], cv["silhouette"], color=BLUE, marker="o", ms=6, lw=1.8,
        label="Silhouette (higher is better)")
ax.plot(cv["k"], cv["davies_bouldin"], color=ORANGE, marker="s", ms=6, lw=1.8, ls="--",
        label="Davies–Bouldin (lower is better)")
ax.axvline(4, color=GRAY, ls=":", lw=1.2)
ax.set_xticks(range(2, 7)); ax.set_ylim(0, 0.85)
ax.set_xlabel("Number of clusters $k$"); ax.set_ylabel("Index value"); ax.legend(loc="upper right")
save(fig, "fig12_cluster_validity.png")

# ---------------- Figure 13: leave-one-out stability of the reference ----------------
loro = pd.read_csv(RES / "loro_stability.csv")["loro_ari"].values
lopo = pd.read_csv(RES / "lopo_stability.csv")
fig, axs = plt.subplots(1, 2, figsize=(W, 2.9), constrained_layout=True,
                        gridspec_kw={"width_ratios": [0.85, 1.15]})
ax = axs[0]
ax.hist(loro, bins=np.arange(0.70, 1.0001 + 0.025, 0.025), color=BLUE, rwidth=0.9, **EDGE)
ax.set_xlim(0.69, 1.03); ax.set_xlabel("ARI against the reference"); ax.set_ylabel("Held-out refits")
ax.set_title("(a) Leave-one-respondent-out"); ax.grid(axis="x", visible=False)
ax.text(0.71, 40, f"{int((loro == 1).sum())} of {len(loro)} refits\nwith ARI = 1.000", fontsize=9, va="top")
ax = axs[1]
names = pilot_labels(lopo["pilot_removed"])[::-1]
vals = lopo["ari_vs_reference"].values[::-1]
ax.barh(range(len(vals)), vals, color=BLUE, height=0.7, **EDGE)
for i, v in enumerate(vals):
    ax.text(v - 0.02, i, f"{v:.3f}", va="center", ha="right", fontsize=8.5, color="white")
ax.set_yticks(range(len(vals))); ax.set_yticklabels(names); ax.set_xlim(0, 1.0)
ax.set_xlabel("ARI against the reference"); ax.set_title("(b) Leave-one-pilot-case-out")
ax.grid(axis="y", visible=False)
save(fig, "fig13_partition_stability.png")


# ---------------- Figure 14: deployment workflow (schematic) ----------------
def workflow():
    steps = [
        (1, "New respondent", BLUE,
         "Collect the seven item responses\non the $-4$ to $+4$ scale.",
         r"$\mathbf{x} = (x_1, x_4, x_8, x_9, x_{21}, x_{32}, x_{38})^{\mathrm{T}}$"),
        (2, "Standardise", AQUA,
         "Standardise each item with the fixed\nconstants $\\mu_j$ and $\\sigma_j$ of Table 4.",
         r"$z_j = \dfrac{x_j - \mu_j}{\sigma_j},\quad j \in S$"),
        (3, "Compute distances", ORANGE,
         "Euclidean distance from $\\mathbf{z}$ to each\nprofile centroid $\\mathbf{m}_c$ of Table 4.",
         r"$d_c = \sqrt{\sum_{j \in S}\,(z_j - m_{cj})^2}$"),
        (4, "Assign profile", "#6250d6",
         "Assign the nearest centroid, which is\nalso the maximum-posterior profile.",
         r"$\hat{c} = \mathrm{arg\,min}_{\,c \in \{1,\ldots,4\}}\; d_c$"),
        (5, "Compute confidence", "#c43a3a",
         "Flag as uncertain, and refer for the full\ninstrument, if the largest posterior is below 0.5.",
         r"$P(c \mid \mathbf{z}) = \dfrac{\exp(-d_c^{\,2}/2)}{\sum_{r=1}^{4}\exp(-d_r^{\,2}/2)}$"),
    ]
    eqno = ["", "(3)", "(4)", "(5)", "(6)"]   # equation numbers in the article
    bh, gap = 0.80, 0.14
    H = len(steps) * bh + (len(steps) - 1) * gap + 0.06
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")
    for i, (num, title, col, desc, eq) in enumerate(steps):
        y0 = H - 0.03 - (i + 1) * bh - i * gap
        ax.add_patch(FancyBboxPatch((0.04, y0), W - 0.08, bh, boxstyle="round,pad=0.0,rounding_size=0.07",
                                    fc="#f6f6f4", ec="#c4c4c4", lw=0.9))
        ax.add_patch(Circle((0.30, y0 + bh / 2), 0.17, fc=col, ec="none"))
        ax.text(0.30, y0 + bh / 2 - 0.005, str(num), ha="center", va="center", color="white",
                fontsize=11, weight="bold")
        ax.text(0.60, y0 + bh - 0.19, title, ha="left", va="center", fontsize=10.5, weight="bold", color=INK)
        ax.text(0.60, y0 + bh - 0.35, desc, ha="left", va="top", fontsize=9, color=INK, linespacing=1.3)
        ax.text(4.95, y0 + bh / 2, eq, ha="center", va="center", fontsize=10, color=INK)
        ax.text(W - 0.12, y0 + bh / 2, eqno[i], ha="right", va="center", fontsize=9.5, color=INK)
        ax.plot([3.62, 3.62], [y0 + 0.08, y0 + bh - 0.08], color="#d6d6d6", lw=0.8)
        if i < len(steps) - 1:
            ax.add_patch(FancyArrowPatch((0.30, y0 - 0.005), (0.30, y0 - gap + 0.005), arrowstyle="-|>",
                                         mutation_scale=11, lw=1.2, color="#6f6f6f"))
    return fig


save(workflow(), "fig14_deployment_workflow.png", tight=False)

# ---------------- Figure 15: classification rule under leave-one-respondent-out ----------------
cl = pd.read_csv(RES / "classification_loro.csv")
ref_nm = cl["reference_profile"].values
pred = cl["predicted_profile"].values
conf = cl["max_posterior"].values
order = ["Risk-Averse Sceptics", "Committed Enthusiasts", "Operationally Cautious", "Pragmatic Optimists"]
cm = np.array([[np.sum((ref_nm == t) & (pred == q)) for q in order] for t in order])
cmap = LinearSegmentedColormap.from_list("b", ["#f4f8fd", "#9ec5f4", "#2a78d6", "#104281"])
fig, axs = plt.subplots(1, 2, figsize=(W, 3.15), constrained_layout=True,
                        gridspec_kw={"width_ratios": [1.1, 0.9]})
ax = axs[0]
ax.imshow(np.log1p(cm), cmap=cmap, vmin=0, vmax=np.log1p(30)); ax.grid(False)
short = ["Sceptics", "Enthusiasts", "Cautious", "Optimists"]
ax.set_xticks(range(4)); ax.set_xticklabels(short, fontsize=9)
ax.set_yticks(range(4)); ax.set_yticklabels(short, fontsize=9)
for i in range(4):
    for j in range(4):
        ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=10,
                color="white" if cm[i, j] > 6 else INK)
ax.set_xlabel("Predicted (nearest centroid)"); ax.set_ylabel("Reference profile")
ax.set_title(f"(a) Confusion matrix (accuracy {np.mean(pred == ref_nm):.3f})")
for sp in ax.spines.values():
    sp.set_visible(True)
ax = axs[1]
bins = [(0.5, 0.7, "0.5–0.7"), (0.7, 0.9, "0.7–0.9"), (0.9, 1.01, "0.9–1.0")]
ns = [int(((conf >= a) & (conf < b)).sum()) for a, b, _ in bins]
acc = [float((pred == ref_nm)[(conf >= a) & (conf < b)].mean()) for a, b, _ in bins]
ax.bar(range(3), acc, color=BLUE, width=0.6, **EDGE)
for i, (n_, a_) in enumerate(zip(ns, acc)):
    ax.text(i, a_ + 0.02, f"$n$ = {n_}", ha="center", va="bottom", fontsize=9)
ax.set_xticks(range(3)); ax.set_xticklabels([b[2] for b in bins]); ax.set_ylim(0, 1.15)
ax.set_xlabel("Maximum posterior probability"); ax.set_ylabel("Accuracy within bin")
ax.set_title("(b) Accuracy by confidence"); ax.grid(axis="x", visible=False)
save(fig, "fig15_classification_rule.png")

# ---------------- Figure 16: ordinal robustness ----------------
mr = json.load(open(RES / "measurement_robustness.json"))["ordinal_robustness"]
vals = [1.0, mr["ari_spearman_corrmatrix_vs_pearson"], mr["ari_kendall_corrmatrix_vs_pearson"],
        mr["ari_rank_transform_variant"]]
labs = ["Pearson\n(reference)", "Spearman\ncorrelation", "Kendall\ncorrelation", "Rank-\ntransformed items"]
fig, ax = plt.subplots(figsize=(4.8, 3.0), constrained_layout=True)
ax.bar(range(4), vals, color=[GRAY, BLUE, BLUE, BLUE], width=0.62, **EDGE)
for i, v in enumerate(vals):
    ax.text(i, v + 0.02, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
ax.set_xticks(range(4)); ax.set_xticklabels(labs); ax.set_ylim(0, 1.15)
ax.set_ylabel("ARI against the Pearson partition"); ax.grid(axis="x", visible=False)
save(fig, "fig16_ordinal_robustness.png")

# ---------------- Figure 17: respondent quality ----------------
ls = pd.read_csv(RES / "respondent_quality.csv")["longstring"].values
fig, axs = plt.subplots(1, 2, figsize=(W, 2.9), constrained_layout=True)
ax = axs[0]
vc = pd.Series(ls).value_counts().reindex(range(0, 17), fill_value=0)
ax.bar(vc.index, vc.values, color=[ORANGE if k >= 10 else BLUE for k in vc.index], width=0.8, **EDGE)
ax.axvline(9.5, color=INK, ls="--", lw=1.0)
ax.text(9.8, vc.max() * 0.98, "flagged\n(10 or more)", fontsize=8.5, va="top")
ax.set_xlabel("Longest run of identical consecutive responses"); ax.set_ylabel("Respondents")
ax.set_title("(a) Longstring index"); ax.set_xticks(range(0, 17, 2)); ax.grid(axis="x", visible=False)
ax = axs[1]
keys = [tuple(r) for r in X.astype(int)]
grp = pd.Series(keys).value_counts()
grp = grp[grp > 1].sort_values(ascending=False).values
ax.bar(range(1, len(grp) + 1), grp, color=BLUE, width=0.7, **EDGE)
ax.set_xticks(range(1, len(grp) + 1)); ax.set_xlabel("Group of identical records (by size)")
ax.set_ylabel("Records in group"); ax.set_title("(b) Identical response vectors")
ax.grid(axis="x", visible=False)
ax.text(len(grp) + 0.4, grp.max() * 0.98, f"{int(grp.sum())} of {len(X)} records in {len(grp)} groups;\n"
        f"{len(set(keys))} distinct vectors", ha="right", va="top", fontsize=8.5)
save(fig, "fig17_respondent_quality.png")

# ---------------- Figure 18: sensitivity to alpha and w ----------------
asw = pd.read_csv(RES / "alpha_sweep.csv")
wsw = pd.read_csv(RES / "weight_sweep.csv")
fig, axs = plt.subplots(1, 2, figsize=(W, 3.0), constrained_layout=True, sharey=True)
for ax, d, xc, ttl, xl, ref in [
        (axs[0], asw, "alpha", "(a) Sparsity penalty ($w$ = 0.5)", r"Sparsity penalty $\alpha$", 0.3),
        (axs[1], wsw, "w_procrustes", r"(b) Procrustes weight ($\alpha$ = 0.3)",
         r"Weight on Procrustes term $w$", 0.5)]:
    ax.plot(d[xc], d["ari"], color=BLUE, marker="s", ms=5.5, lw=1.6, label="ARI")
    ax.plot(d[xc], d["procrustes"], color=ORANGE, marker="^", ms=6, lw=1.6, ls="--",
            label="Procrustes similarity")
    sec = ax.secondary_xaxis("top"); sec.set_ticks(list(d[xc]))
    sec.set_xticklabels([str(int(v)) for v in d["size"]], fontsize=8.5); sec.tick_params(length=2.5, pad=1.5)
    sec.set_xlabel("Items selected", fontsize=9, labelpad=3)
    ax.axvline(ref, color=GRAY, ls=":", lw=1.2)
    ax.set_title(ttl, pad=4, y=1.17); ax.set_xlabel(xl); ax.set_ylim(0.70, 1.03)
axs[0].set_ylabel("Value of fidelity term")
axs[0].legend(loc="lower right")
save(fig, "fig18_objective_sensitivity.png")

# ---------------- Figure 19: best-found fidelity-size frontier ----------------
pf = pd.read_csv(RES / "pareto_frontier.csv")
fig, ax = plt.subplots(figsize=(4.8, 3.0), constrained_layout=True)
ax.plot(pf["size"], pf["fidelity"], color=BLUE, marker="o", ms=4.5, lw=1.6, label="Best-found frontier")
c7 = pf[pf["size"] == 7].iloc[0]
ax.plot([7], [c7["fidelity"]], marker="o", ms=11, color=ORANGE, mec=INK, mew=0.8, ls="none",
        label="Seven-item consensus")
ax.annotate(f"consensus: 7 items, $\\varphi$ = {c7['fidelity']:.3f}", xy=(7, c7["fidelity"]),
            xytext=(12, 0.925), fontsize=9, arrowprops=dict(arrowstyle="->", lw=0.8, color=INK))
ax.set_xlabel("Short-form size $|s|$"); ax.set_ylabel(r"Composite fidelity $\varphi$")
ax.legend(loc="lower right")
save(fig, "fig19_pareto_frontier.png")
print("Figures complete.")
