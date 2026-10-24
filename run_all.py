#!/usr/bin/env python3
"""Reproduce every result, figure and table of the article, in order, from a
fresh checkout.

    python run_all.py            # full pipeline
    python run_all.py --quick    # skip the two long stages (04 and 08) and
                                 # reuse their committed results

Runtimes are hardware-dependent; see REPRODUCIBILITY.md. Every stage is
deterministic given the fixed seeds listed there. Figure and table numbers
below are those of the article."""
import subprocess
import sys
import time
from pathlib import Path

SRC = Path(__file__).resolve().parent / "src"
LONG = {"04_subsampling_stability.py", "08_alternative_baselines.py"}
STAGES = [
    ("01  Reference structure: PCA and Ward clustering",   "01_pca.py"),
    ("02  Tabu Search item selection (Tables 2, C1)",      "02_tabu_search.py"),
    ("03  Baselines and profile means (Tables 3, A1, B1)", "03_baselines_and_profiles.py"),
    ("04  Repeated subsampling",                           "04_subsampling_stability.py"),
    ("05  Robustness of the reference segmentation",       "05_reference_robustness.py"),
    ("06  Alternative reference structures",               "06_sceptics_sensitivity.py"),
    ("07  Sensitivity of the composite objective",         "07_objective_sensitivity.py"),
    ("08  Alternative selection procedures (App. E)",      "08_alternative_baselines.py"),
    ("09  Deployable classification rule",                 "09_classification_rule.py"),
    ("10  Measurement and respondent quality",             "10_measurement_robustness.py"),
    ("T4  Deployment constants (Table 4)",                 "deploy_constants.py"),
    ("F   Figures 1 to 19",                                "make_figures.py"),
    ("X   Exhibit CSV files",                              "build_exhibits.py"),
]


def main():
    quick = "--quick" in sys.argv
    t0 = time.time()
    for label, script in STAGES:
        if quick and script in LONG:
            print(f"\n=== SKIP (quick): {label} ===", flush=True)
            continue
        print(f"\n=== {label}  ({script}) ===", flush=True)
        s = time.time()
        r = subprocess.run([sys.executable, str(SRC / script)], cwd=str(SRC))
        if r.returncode != 0:
            print(f"\nSTAGE FAILED: {script} (exit {r.returncode})")
            sys.exit(r.returncode)
        print(f"    done in {time.time() - s:.0f}s", flush=True)
    print(f"\nAll requested stages complete in {time.time() - t0:.0f}s.")


if __name__ == "__main__":
    main()
