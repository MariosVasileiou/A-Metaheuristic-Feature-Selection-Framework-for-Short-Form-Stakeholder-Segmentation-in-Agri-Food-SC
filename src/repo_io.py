"""Shared paths and data loaders for the repository.

All scripts import from here, so the pipeline runs from a fresh checkout with
no absolute paths. The data are read from the de-identified CSV files in
data/."""
from pathlib import Path

import pandas as pd

_SRC = Path(__file__).resolve().parent
ROOT = _SRC.parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# 40 items in fixed order Q1..Q40 (Q1 = first survey item, ..., Q40 = last)
ITEMS = [f"Q{i}" for i in range(1, 41)]


def load_items(path=None):
    """Return (df, item_columns); df has respondent_id, pilot and Q1..Q40."""
    df = pd.read_csv(path or DATA_DIR / "alliance_items.csv")
    assert list(ITEMS) == [c for c in df.columns if c.startswith("Q") and c[1:].isdigit()][:40]
    return df, list(ITEMS)


def load_demographics(path=None):
    """Coarsened demographics: respondent_id, pilot, country, age_band, gender, org_type."""
    return pd.read_csv(path or DATA_DIR / "alliance_demographics_coarsened.csv")
