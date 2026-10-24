# A Metaheuristic Feature Selection Framework for Short-Form Stakeholder Segmentation

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23625696.svg)](https://doi.org/10.5281/zenodo.23625696)

Code, de-identified data and numeric outputs accompanying the article
*A Metaheuristic Feature Selection Framework for Short-Form Stakeholder
Segmentation in Agri-Food Supply Chains* (Vasileiou, Lappas, Kyrgiakos,
Kleisiari, Kleftodimos and Vlontzos), submitted to *Supply Chain Analytics*.
The repository reproduces every figure and every numerical table of the article.

## What the study does

A 40-item stakeholder survey (42 respondents across seven Geographical-Indication
and organic agri-food pilot value chains) is reduced to a short form with Tabu
Search. Item selection is cast as combinatorial optimisation: a candidate subset
is scored by how well the segmentation it induces reproduces the full-instrument
reference structure. The reference structure is obtained by standardising the
items, running PCA (four components, 68.1% of the variance) and Ward clustering
(k = 4). The objective for a subset s is

    F(s) = w * Procrustes_similarity + (1 - w) * max(0, ARI) - alpha * |s| / p

with w = 0.5 and alpha = 0.3. Tabu Search (tenure 10, minimum size 5,
100 iterations, 30 restarts) returns the seven-item consensus form
{Q1, Q4, Q8, Q9, Q21, Q32, Q38}, which recovers the four-profile reference
partition (ARI = 1.000, Procrustes similarity 0.890, objective 0.892). The
repository also contains the repeated-subsampling, robustness, sensitivity,
alternative-procedure, measurement-quality and deployment-rule analyses.

The four reference profiles are Pragmatic Optimists (n = 29), Committed
Enthusiasts (n = 7), Operationally Cautious (n = 4) and Risk-Averse Sceptics
(n = 2). Only 23 of the 42 response vectors are distinct; the consequences of
this are analysed in stages 05, 06 and 10 and discussed in the article.

## Repository layout

    run_all.py            entry point: reproduces everything in order
    requirements.txt      pinned dependencies (Python 3.12)
    LICENSE               MIT (code)
    CITATION.cff          how to cite
    README.md             this file
    REPRODUCIBILITY.md    exhibit-to-script map, seeds, determinism, runtimes
    src/                  analysis code, one file per stage
    data/                 de-identified data, data licence and DATA_CHANGES.md
    results/              numeric outputs of the stages (CSV, JSON, NPY)
    figures/              Figures 1 to 19, numbered as in the article (300 dpi)
    exhibits/             the numbers behind each figure and table, as CSV files

## Installation

    python3 -m venv .venv
    source .venv/bin/activate          # on Windows: .venv\Scripts\activate
    pip install -r requirements.txt

## Reproducing every figure and table

    python run_all.py            # full pipeline
    python run_all.py --quick    # skip the two long stages (04 and 08) and
                                 # reuse their committed results

Any stage can also be run on its own, for example
`python src/05_reference_robustness.py`. The figures alone can be redrawn from
the committed results in a few seconds with `python src/make_figures.py`.

| Stage | Script | Produces (article numbering) |
|-------|--------|------------------------------|
| 01 | `01_pca.py` | reference structure: PCA and Ward clustering (Sections 2.3 and 3.1) |
| 02 | `02_tabu_search.py` | 30 Tabu Search runs (Sections 3.2 and 3.3); Tables 2 and C1 |
| 03 | `03_baselines_and_profiles.py` | random and top-loading baselines, profile means (Sections 3.4 and 3.5); Tables 3, A1 and B1 |
| 04 | `04_subsampling_stability.py` | repeated subsampling (Section 3.6) |
| 05 | `05_reference_robustness.py` | robustness of the reference segmentation (Section 3.7) |
| 06 | `06_sceptics_sensitivity.py` | short forms re-derived against alternative references (Section 3.7) |
| 07 | `07_objective_sensitivity.py` | sensitivity of the composite objective (Sections 2.7 and 3.8) |
| 08 | `08_alternative_baselines.py` | eight alternative selection procedures (Section 3.7); Tables E2 and E3 |
| 09 | `09_classification_rule.py` | deployable classification rule and its validation (Section 3.8) |
| 10 | `10_measurement_robustness.py` | ordinal treatments and respondent quality (Section 3.8) |
| T4 | `deploy_constants.py` | Table 4 and the accuracies reported in Section 3.8 |
| F | `make_figures.py` | Figures 1 to 19 |
| X | `build_exhibits.py` | `exhibits/` |

`REPRODUCIBILITY.md` maps every figure, table and number of the article to the
script and the result file that produce it.

## Notes on specific figures

- Figures 1, 2 and 13 are drawn from the coarsened public-release fields, so
  they differ in presentation from the published figures, which were drawn
  from the uncoarsened source file. Figures 1(b) and 13(b) show the pilot
  codes P1 to P7 instead of the pilot names, with the same counts and ARI
  values; Figure 2(a) pools the three organisation types that have a single
  respondent each into "Other"; and Figure 2(b) shows age bands instead of a
  histogram of exact ages.
- Figure 14 is a schematic of the deployment workflow. It uses no data and is
  drawn by `make_figures.py` like the other figures.
- All other figures match the published ones.

## Expected runtime

The full pipeline runs on a single core in about 15 to 25 minutes, depending on
the machine. Stage 04 accounts for more than half of the time, stage 02 for a
few minutes and stages 06 and 07 for one to two minutes each; every other stage
takes less than half a minute. `REPRODUCIBILITY.md` lists the time of each
stage in a reference run. The pipeline is deterministic given the fixed seeds
listed there.

## Data, licences and conditions of use

The code is released under the MIT licence (`LICENSE`) and the data under
CC BY 4.0 (`data/LICENSE`). The data are de-identified survey responses
collected in the ALLIANCE project: `data/DATA_CHANGES.md` documents exactly
which fields were removed or coarsened, and `data/LICENSE` states the
conditions of use. With 42 respondents, some residual risk of
re-identification remains, so the data must not be used to identify
individual respondents or their organisations. The original questionnaire file
is not part of the repository; the wording of the 40 items is given in
Appendix D of the article.

## Citation

If you use the code or the data, please cite the article and this repository
(https://doi.org/10.5281/zenodo.23625696); `CITATION.cff` gives both entries.

## Funding

This project has received funding from the European Union's Horizon Europe
research and innovation programme under grant agreement No 101084188
(Acronym: ALLIANCE). Views and opinions expressed are, however, those of the
authors only and do not necessarily reflect those of the European Union or the
European Research Executive Agency (REA). Neither the European Union nor the
granting authority can be held responsible for them.
