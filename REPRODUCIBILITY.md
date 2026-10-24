# Reproducibility

Companion to `README.md`. It maps every exhibit and every number reported in
the article to the script and the result file that produce it, and documents
the environment, the random seeds and the runtimes.

## 1. Exhibits of the article

Figure and table numbers are those of the article. The figure files in
`figures/` carry the same numbers, and `exhibits/` holds the numbers behind
each exhibit as CSV files, rebuilt from `results/` by `src/build_exhibits.py`.
All figures are drawn by `src/make_figures.py`; the scripts listed below
produce the result files the figures are drawn from.

| Exhibit | Script(s) | Result file(s) | Figure or exhibit file |
|---|---|---|---|
| Figure 1 (sample: country, pilot case) | data only | `data/alliance_demographics_coarsened.csv` | `figures/fig01_sample_country_pilot.png`* |
| Figure 2 (sample: organisation type, age) | data only | `data/alliance_demographics_coarsened.csv` | `figures/fig02_sample_orgtype_age.png`* |
| Figure 3 (scree) | `01_pca.py` | `results/pca_stats.json` | `figures/fig03_scree.png` |
| Figure 4 (profiles in the PC1 x PC2 plane) | `01_pca.py` | `results/Y_full.npy`, `results/profiles_ref.npy` | `figures/fig04_profiles_pc1_pc2.png` |
| Figure 5 (convergence) | `02_tabu_search.py` | `results/histories.csv` | `figures/fig05_convergence.png` |
| Figure 6 (item selection frequency) | `02_tabu_search.py` | `results/item_frequencies.csv` | `figures/fig06_item_frequency.png` |
| Figure 7 (fidelity-size frontier) | `03_baselines_and_profiles.py` | `results/frontier.csv` | `figures/fig07_fidelity_size_frontier.png` |
| Figure 8 (profile signatures) | `01_pca.py` | `results/profiles_ref.npy`, `data/alliance_items.csv` | `figures/fig08_profile_signatures.png` |
| Figure 9 (subsampling: overlap, item frequencies) | `04_subsampling_stability.py` | `results/subsampling_per_run.csv`, `results/subsampling_item_frequencies.csv` | `figures/fig09_subsampling_overlap_items.png` |
| Figure 10 (subsampling: Procrustes similarity) | `04_subsampling_stability.py` | `results/subsampling_per_run.csv` | `figures/fig10_subsampling_procrustes.png` |
| Figure 11 (selection procedures at seven items) | `08_alternative_baselines.py` | `results/alternative_baselines.csv` | `figures/fig11_selection_procedures.png` |
| Figure 12 (internal cluster validity) | `05_reference_robustness.py` | `results/cluster_validity.csv` | `figures/fig12_cluster_validity.png` |
| Figure 13 (stability of the reference partition) | `05_reference_robustness.py` | `results/loro_stability.csv`, `results/lopo_stability.csv` | `figures/fig13_partition_stability.png`* |
| Figure 14 (deployment workflow) | none: schematic | none | `figures/fig14_deployment_workflow.png` |
| Figure 15 (classification rule) | `09_classification_rule.py` | `results/classification_loro.csv` | `figures/fig15_classification_rule.png` |
| Figure 16 (ordinal treatments) | `10_measurement_robustness.py` | `results/measurement_robustness.json` | `figures/fig16_ordinal_robustness.png` |
| Figure 17 (respondent quality) | `10_measurement_robustness.py` | `results/respondent_quality.csv`, `data/alliance_items.csv` | `figures/fig17_respondent_quality.png` |
| Figure 18 (sensitivity to alpha and w) | `07_objective_sensitivity.py` | `results/alpha_sweep.csv`, `results/weight_sweep.csv` | `figures/fig18_objective_sensitivity.png` |
| Figure 19 (best-found fidelity-size frontier) | `07_objective_sensitivity.py` | `results/pareto_frontier.csv` | `figures/fig19_pareto_frontier.png` |
| Table 1 (parameter settings) | settings of `02_tabu_search.py` | `results/tabu_results.json` (`config`) | `exhibits/table1_parameter_settings.csv` |
| Table 2 (item selection frequencies) | `02_tabu_search.py` | `results/item_frequencies.csv` | `exhibits/table2_item_selection_frequencies.csv` |
| Table 3 (profile means, short form) | `03_baselines_and_profiles.py` | `results/profile_sf_means.csv` | `exhibits/table3_profile_shortform_means.csv` |
| Table 4 (deployment constants) | `deploy_constants.py` | `results/rule_card_table4.csv` | `exhibits/table4_deployment_constants.csv` |
| Table A1 (profile means, all items) | `03_baselines_and_profiles.py` | `results/profile_all_means.csv` | `exhibits/appendixA1_profile_means_all_items.csv` |
| Table B1 (frontier values) | `03_baselines_and_profiles.py` | `results/frontier.csv` | `exhibits/appendixB1_fidelity_size_frontier.csv` |
| Table C1 (per-run summary) | `02_tabu_search.py` | `results/tabu_results.json` (`per_run`) | `exhibits/appendixC1_per_run_summary.csv` |
| Table E1 (specification of the procedures) | settings of `08_alternative_baselines.py` | none | none |
| Tables E2 and E3 (alternative procedures) | `08_alternative_baselines.py` | `results/alternative_baselines.csv` | `exhibits/appendixE2_fixed_cardinality.csv`, `exhibits/appendixE3_item_agreement.csv` |

\* Figures 1, 2 and 13 are drawn from the coarsened public-release fields and
differ in presentation from the published figures, which were drawn from the
uncoarsened source file: Figures 1(b) and 13(b) show the pilot codes P1 to P7
instead of the pilot names, Figure 2(a) pools the three organisation types
that have a single respondent each into "Other", and Figure 2(b) shows age
bands instead of a histogram of exact ages. The counts by country and by pilot
case, and the ARI values of Figure 13, are the same. If a file
`data/pilot_labels_PRIVATE.csv` (columns `pilot,label`) is present,
`make_figures.py` shows the pilot names instead of the codes; that file is not
part of the repository.

## 2. Numbers reported in the text

All files are in `results/`.

| Section | Quantity | Result file (key) |
|---|---|---|
| 3.1 | variance explained, profile sizes, Ward versus k-means agreement | `pca_stats.json` |
| 3.2 | objective, size and fidelity statistics of the 30 runs | `tabu_results.json` (`aggregate`, `per_run`) |
| 3.2 | convergence (iteration of the last improvement in each run) | `histories.csv` |
| 3.3 | item selection frequencies, unanimous and majority items | `item_frequencies.csv`, `tabu_results.json` |
| 3.4 | random and top-loading baselines at seven items | `baselines.json` |
| 3.5 | profile means on the short form | `profile_sf_means.csv` |
| 3.6 | subsampling statistics (ARI, Procrustes, sizes, overlap, item frequencies) | `subsampling_results.json`, `subsampling_per_run.csv`, `subsampling_item_frequencies.csv` |
| 3.7 | alternative procedures, fixed and variable cardinality | `alternative_baselines.csv` |
| 3.7 | rank of the correlation matrix, KMO, cluster-validity indices | `reference_robustness.json` (`factorability`, `cluster_validity`) |
| 3.7 | sensitivity to the number of components and of profiles | `reference_robustness.json` (`component_sensitivity`, `profile_count_sensitivity`) |
| 3.7 | leave-one-respondent-out and leave-one-pilot-case-out stability | `loro_stability.csv`, `lopo_stability.csv` |
| 3.7 | the two Risk-Averse Sceptics records | `reference_robustness.json` (`risk_averse_sceptics`) |
| 3.7 | reference re-derived on the 23 distinct response vectors | `reference_robustness.json` (`distinct_vectors_reference`) |
| 3.7 | short forms re-derived against alternative references | `sceptics_sensitivity.json` |
| 3.8 | deployment constants, accuracies, smallest maximum posteriors | `rule_card_table4.csv`, `classification_rule.json`, `classification_loro.csv` |
| 3.8 | ordinal checks, Cronbach's alpha, longstring screening and exclusion, duplicate structure | `measurement_robustness.json`, `respondent_quality.csv` |
| 3.8 | sparsity and weight sweeps, negative-ARI evaluations, best-found frontier | `objective_sensitivity.json`, `alpha_sweep.csv`, `weight_sweep.csv`, `pareto_frontier.csv` |

## 3. Environment

Reference interpreter: Python 3.12.3, with the pinned packages of
`requirements.txt` (numpy 2.4.4, pandas 3.0.2, scipy 1.17.1, scikit-learn 1.8.0,
matplotlib 3.11.2). No GPU or network access is needed. matplotlib affects only
the figures, not the numeric results. The figures use the Liberation Serif font
when it is installed and fall back to Times New Roman or DejaVu Serif
otherwise; another font or another matplotlib version changes the rendering
slightly but not the content.

## 4. Seeds and determinism

| Stage | Random number generation |
|---|---|
| 01 | `KMeans(n_init=30, random_state=42)` in the k-means cross-check |
| 02 | `default_rng(1000 + r)` for restart r |
| 03 | `np.random.seed(42)` for the random-subset baselines |
| 04 | `default_rng(7000 + i)` for subsample i |
| 05, 09, 10, deploy_constants, make_figures, build_exhibits | none needed (deterministic) |
| 06 | `multi_start_tabu(..., seed_base=3100)` |
| 07 | `default_rng(4000 + int(100 alpha + 10 w) + r)` |
| 08 | `SparsePCA(random_state=0)`, `RandomForestClassifier(random_state=0)`, simulated annealing `default_rng(11)`, genetic algorithm `default_rng(12)` |

Every stochastic stage seeds its generator explicitly, so re-runs in the same
environment reproduce the reported statistics. Two caveats apply.
`sf_common.tabu_search` falls back to an unseeded generator if a caller passes
none; every caller in this pipeline passes a seed. Bit-for-bit identity across
machines is not guaranteed for floating point, because the BLAS/LAPACK build
and the thread count can change the last digits of PCA and Procrustes values;
the reported statistics are stable to the precision given in the article. The
run times recorded in `results/tabu_results.json` vary from run to run.

## 5. Runtime

Wall-clock times of a reference run of `run_all.py` (Python 3.12.3 with the
pinned packages, one core in use), about 21 minutes in total:

| Stage | Time |
|---|---|
| 01, 03, 05, 09, 10, deploy_constants, build_exhibits | a few seconds each |
| make_figures | about 10 seconds |
| 08 | about 20 seconds |
| 07 | about 1.5 minutes |
| 06 | about 2.5 minutes |
| 02 | about 3.5 minutes |
| 04 | about 13 minutes |

On a faster machine the total is closer to 15 minutes. `run_all.py --quick`
skips stages 04 and 08 and reuses their committed results.
