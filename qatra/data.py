"""Load the synthetic dataset, generating it on first run."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .config import DATA_DIR
from .generate import generate


def load_all(data_dir: Path = DATA_DIR) -> dict:
    data_dir = Path(data_dir)
    if not (data_dir / "meta.json").exists():
        generate(data_dir)
    donors = pd.read_csv(data_dir / "donors.csv", parse_dates=["last_donation", "last_contacted", "registered_on"])
    return {
        "meta": json.loads((data_dir / "meta.json").read_text()),
        "donors": donors,
        "hospitals": pd.read_csv(data_dir / "hospitals.csv"),
        "history": pd.read_csv(data_dir / "history_daily.csv", parse_dates=["date"]),
        "inventory": pd.read_csv(data_dir / "inventory.csv"),
        "patients": pd.read_csv(data_dir / "thalassemia_patients.csv", parse_dates=["next_transfusion"]),
    }
