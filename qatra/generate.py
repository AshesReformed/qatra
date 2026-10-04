"""Generate the SYNTHETIC dataset used by the demo.

No real person, donor, patient or hospital is represented. Donors carry no
names and no phone numbers; their location is stored only at ~1 km precision.

    python -m qatra.generate            # writes ./data/*.csv
"""
from __future__ import annotations

import hashlib
import json
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from .config import (
    CITIES, DATA_DIR, DEMAND_TYPE_SHARE, DONOR_TYPE_SHARE,
    dengue_shape, eid_flag, ramadan_flag, summer_shape,
)

AVAILABILITY = ["Available now", "Evenings & weekends", "Paused"]
DENGUE_YEAR_AMPLITUDE = {2023: 0.8, 2024: 1.0, 2025: 1.25, 2026: 1.1, 2027: 1.0}


def _pid(prefix: str, seed: int, i: int) -> str:
    return prefix + hashlib.sha256(f"{prefix}{seed}-{i}".encode()).hexdigest()[:8].upper()


def make_donors(rng, as_of: date, n: int, seed: int) -> pd.DataFrame:
    cities = list(CITIES)
    w = np.array([CITIES[c]["weight"] for c in cities])
    city = rng.choice(cities, n, p=w / w.sum())
    types, shares = zip(*DONOR_TYPE_SHARE.items())
    btype = rng.choice(types, n, p=shares)
    lat = np.array([CITIES[c]["lat"] for c in city]) + rng.normal(0, 0.06, n)
    lon = np.array([CITIES[c]["lon"] for c in city]) + rng.normal(0, 0.06, n)

    never = rng.random(n) < 0.25
    days_ago = rng.integers(5, 420, n)
    last = pd.to_datetime([as_of - timedelta(days=int(d)) for d in days_ago])
    last = pd.Series(last).where(~never)

    contacted_days = rng.integers(0, 60, n)
    last_contacted = pd.Series(pd.to_datetime(
        [as_of - timedelta(days=int(d)) for d in contacted_days])).where(rng.random(n) < 0.35)

    return pd.DataFrame({
        "donor_id": [_pid("D", seed, i) for i in range(n)],
        "blood_type": btype,
        "city": city,
        "lat": np.round(lat, 2),            # ~1 km grid: data minimisation
        "lon": np.round(lon, 2),
        "last_donation": last.dt.date,
        "availability": rng.choice(AVAILABILITY, n, p=[0.5, 0.35, 0.15]),
        "response_rate": np.round(rng.beta(2.5, 3.0, n), 2),
        "last_contacted": last_contacted.dt.date,
        "consent_matching": True,          # registry only holds opted-in donors
        "consent_contact": True,
        "registered_on": [as_of - timedelta(days=int(d)) for d in rng.integers(10, 900, n)],
    })


def make_hospitals(rng) -> pd.DataFrame:
    rows = []
    for city, c in CITIES.items():
        k = int(min(6, max(2, round(c["weight"] / 3))))
        for i in range(k):
            rows.append({
                "hospital_id": f"H-{c['code']}-{i + 1:02d}",
                "name": f"{city} Hospital {chr(65 + i)} (synthetic)",
                "city": city,
                "lat": round(c["lat"] + rng.normal(0, 0.04), 4),
                "lon": round(c["lon"] + rng.normal(0, 0.04), 4),
            })
    return pd.DataFrame(rows)


def make_history(rng, as_of: date, years: int = 3) -> pd.DataFrame:
    """Daily units requested (demand) and donated (supply) per city x blood type."""
    dates = pd.date_range(as_of - timedelta(days=365 * years), as_of - timedelta(days=1), freq="D")
    yrs = (dates - dates[0]).days.to_numpy() / 365.0
    amp = np.array([DENGUE_YEAR_AMPLITUDE.get(y, 1.0) for y in dates.year])
    dengue, summer = dengue_shape(dates) * amp, summer_shape(dates)
    ramadan, eid = ramadan_flag(dates), eid_flag(dates)
    sunday = (dates.dayofweek == 6).astype(float)

    frames = []
    for city, c in CITIES.items():
        base = c["weight"] * 2.5
        drives = (rng.random(len(dates)) < 1 / 7) * 0.8          # donation camps
        d_season = (1 + 0.6 * c["dengue"] * dengue + 0.25 * eid) * (1 - 0.25 * sunday) * (1 + 0.08 * yrs)
        s_season = ((1 - 0.35 * ramadan - 0.2 * summer + 0.15 * c["dengue"] * dengue)
                    * (1 + 0.3 * sunday) * (1 + drives) * (1 + 0.05 * yrs))
        for bt in DONOR_TYPE_SHARE:
            rare_boost = 1.3 if bt.endswith("-") else 1.0          # rare-donor call-backs
            frames.append(pd.DataFrame({
                "date": dates.date,
                "city": city,
                "blood_type": bt,
                "demand": rng.poisson(base * DEMAND_TYPE_SHARE[bt] * d_season),
                "supply": rng.poisson(base * 1.06 * DONOR_TYPE_SHARE[bt] * rare_boost * s_season),
            }))
    return pd.concat(frames, ignore_index=True)


def make_inventory(rng, hospitals: pd.DataFrame) -> pd.DataFrame:
    """Current units on the shelf: roughly a week of normal demand."""
    rows = []
    per_city = hospitals.groupby("city").size()
    for h in hospitals.itertuples():
        base = CITIES[h.city]["weight"] * 2.5 / per_city[h.city]
        for bt, share in DEMAND_TYPE_SHARE.items():
            rows.append({"hospital_id": h.hospital_id, "city": h.city, "blood_type": bt,
                         "units": int(rng.poisson(base * share * 7))})
    return pd.DataFrame(rows)


def make_patients(rng, as_of: date, hospitals: pd.DataFrame, seed: int, n: int = 140) -> pd.DataFrame:
    """Repeat-transfusion (thalassemia major) patients. IDs only, no names."""
    cities = list(CITIES)
    w = np.array([CITIES[c]["weight"] for c in cities])
    city = rng.choice(cities, n, p=w / w.sum())
    types, shares = zip(*DONOR_TYPE_SHARE.items())
    rows = []
    for i in range(n):
        hosp = hospitals[hospitals.city == city[i]].sample(1, random_state=int(rng.integers(1e9))).iloc[0]
        rows.append({
            "patient_id": _pid("P", seed, i),
            "blood_type": rng.choice(types, p=shares),
            "city": city[i],
            "hospital_id": hosp.hospital_id,
            "units_per_session": int(rng.choice([1, 2], p=[0.6, 0.4])),
            "interval_days": int(rng.choice([21, 24, 28])),
            "next_transfusion": as_of + timedelta(days=int(rng.integers(0, 28))),
        })
    return pd.DataFrame(rows)


def generate(out_dir: Path = DATA_DIR, as_of: date | None = None,
             n_donors: int = 6000, seed: int = 42) -> Path:
    as_of = as_of or date.today()
    rng = np.random.default_rng(seed)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    hospitals = make_hospitals(rng)
    make_donors(rng, as_of, n_donors, seed).to_csv(out_dir / "donors.csv", index=False)
    hospitals.to_csv(out_dir / "hospitals.csv", index=False)
    make_history(rng, as_of).to_csv(out_dir / "history_daily.csv", index=False)
    make_inventory(rng, hospitals).to_csv(out_dir / "inventory.csv", index=False)
    make_patients(rng, as_of, hospitals, seed).to_csv(out_dir / "thalassemia_patients.csv", index=False)
    (out_dir / "meta.json").write_text(json.dumps({
        "synthetic": True,
        "note": "All records are synthetic. No real donors, patients or hospitals.",
        "as_of": as_of.isoformat(), "seed": seed, "n_donors": n_donors,
    }, indent=2))
    return out_dir


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Generate synthetic Qatra data")
    p.add_argument("--donors", type=int, default=6000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=Path, default=DATA_DIR)
    a = p.parse_args()
    print("Wrote synthetic data to", generate(a.out, n_donors=a.donors, seed=a.seed))
