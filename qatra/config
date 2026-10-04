"""Static configuration: cities, blood-type shares, calendar effects.

Every number here is an assumption used to generate SYNTHETIC data. Blood-group
shares are approximate figures in line with published Pakistani donor studies
(B+ is the most common group in Pakistan); they vary by region and study.
"""
from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"

ELIGIBILITY_DAYS = 90          # minimum gap between whole-blood donations
MIN_AGE, MAX_AGE = 18, 60      # typical donor criteria used for form validation
MIN_WEIGHT_KG = 50

BLOOD_TYPES = ["O+", "O-", "A+", "A-", "B+", "B-", "AB+", "AB-"]

# Share of each group among donors (approximate, sums to 1.0).
DONOR_TYPE_SHARE = {
    "B+": 0.32, "O+": 0.29, "A+": 0.22, "AB+": 0.08,
    "B-": 0.03, "O-": 0.03, "A-": 0.02, "AB-": 0.01,
}
# Share of each group among requests. Negatives are over-represented relative to
# donors because O- is used as the emergency universal red-cell group and
# Rh-negative patients cannot take Rh-positive blood.
DEMAND_TYPE_SHARE = {
    "B+": 0.30, "O+": 0.28, "A+": 0.21, "AB+": 0.07,
    "B-": 0.035, "O-": 0.055, "A-": 0.03, "AB-": 0.02,
}

# weight ~ relative population (millions, rounded); dengue = relative dengue exposure
CITIES = {
    "Karachi":    dict(code="KHI", lat=24.8607, lon=67.0011, weight=16.0, dengue=1.0),
    "Lahore":     dict(code="LHE", lat=31.5204, lon=74.3587, weight=13.0, dengue=1.0),
    "Faisalabad": dict(code="LYP", lat=31.4504, lon=73.1350, weight=3.6, dengue=0.6),
    "Rawalpindi": dict(code="RWP", lat=33.5651, lon=73.0169, weight=2.3, dengue=1.0),
    "Gujranwala": dict(code="GRW", lat=32.1877, lon=74.1945, weight=2.3, dengue=0.6),
    "Peshawar":   dict(code="PEW", lat=34.0151, lon=71.5249, weight=2.3, dengue=0.7),
    "Multan":     dict(code="MUX", lat=30.1968, lon=71.4782, weight=2.2, dengue=0.5),
    "Hyderabad":  dict(code="HDD", lat=25.3960, lon=68.3578, weight=1.9, dengue=0.5),
    "Quetta":     dict(code="UET", lat=30.1798, lon=66.9750, weight=1.6, dengue=0.2),
    "Islamabad":  dict(code="ISB", lat=33.6844, lon=73.0479, weight=1.2, dengue=1.0),
    "Sialkot":    dict(code="SKT", lat=32.4945, lon=74.5229, weight=0.9, dengue=0.5),
}

# Approximate Ramadan windows and Eid days (dates shift with moon sighting).
RAMADAN = [
    (date(2023, 3, 23), date(2023, 4, 21)),
    (date(2024, 3, 11), date(2024, 4, 9)),
    (date(2025, 3, 1), date(2025, 3, 30)),
    (date(2026, 2, 18), date(2026, 3, 19)),
    (date(2027, 2, 8), date(2027, 3, 9)),
]
EIDS = [
    date(2023, 4, 22), date(2023, 6, 29),
    date(2024, 4, 10), date(2024, 6, 17),
    date(2025, 3, 31), date(2025, 6, 7),
    date(2026, 3, 20), date(2026, 5, 27),
    date(2027, 3, 10), date(2027, 5, 16),
]


def _as_index(d) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(pd.to_datetime(d))


def dengue_shape(d) -> np.ndarray:
    """Seasonal dengue curve: peaks mid-October (post-monsoon), 0..1."""
    doy = _as_index(d).dayofyear.to_numpy()
    return np.exp(-0.5 * ((doy - 288) / 28.0) ** 2)


def summer_shape(d) -> np.ndarray:
    """Peak-summer heat curve (donations dip), 0..1, centred late June."""
    doy = _as_index(d).dayofyear.to_numpy()
    return np.exp(-0.5 * ((doy - 180) / 25.0) ** 2)


def ramadan_flag(d) -> np.ndarray:
    idx = _as_index(d)
    out = np.zeros(len(idx))
    for start, end in RAMADAN:
        out[(idx >= pd.Timestamp(start)) & (idx <= pd.Timestamp(end))] = 1.0
    return out


def eid_flag(d, window: int = 3) -> np.ndarray:
    """1 within +/- `window` days of an Eid (travel, road accidents)."""
    idx = _as_index(d)
    out = np.zeros(len(idx))
    for e in EIDS:
        delta = (idx - pd.Timestamp(e)).days
        out[np.abs(delta) <= window] = 1.0
    return out
