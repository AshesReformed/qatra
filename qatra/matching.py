"""Donor eligibility, ABO/Rh compatibility and request matching."""
from __future__ import annotations

import math
from datetime import date, timedelta

import numpy as np
import pandas as pd

from .config import ELIGIBILITY_DAYS

# Red-cell / whole-blood compatibility: recipient -> acceptable donor groups.
# (Plasma and platelet rules differ; the MVP matches whole blood / red cells.)
COMPATIBLE_DONORS = {
    "O-":  ["O-"],
    "O+":  ["O+", "O-"],
    "A-":  ["A-", "O-"],
    "A+":  ["A+", "A-", "O+", "O-"],
    "B-":  ["B-", "O-"],
    "B+":  ["B+", "B-", "O+", "O-"],
    "AB-": ["AB-", "A-", "B-", "O-"],
    "AB+": ["AB+", "AB-", "A+", "A-", "B+", "B-", "O+", "O-"],
}

URGENCY_WEIGHTS = {   # (compatibility, distance, availability)
    "Critical (within hours)": (0.30, 0.50, 0.20),
    "Urgent (today)": (0.35, 0.40, 0.25),
    "Planned (surgery / scheduled)": (0.45, 0.25, 0.30),
}


def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 6371.0 * 2 * np.arcsin(np.sqrt(a))


def add_eligibility(donors: pd.DataFrame, on: date) -> pd.DataFrame:
    """Enforce the 90-day rule. Never-donated donors are eligible."""
    df = donors.copy()
    last = pd.to_datetime(df["last_donation"])
    df["days_since_donation"] = (pd.Timestamp(on) - last).dt.days
    df["next_eligible"] = last + pd.Timedelta(days=ELIGIBILITY_DAYS)
    df["eligible"] = last.isna() | (df["days_since_donation"] >= ELIGIBILITY_DAYS)
    return df


def is_eligible(last_donation: date | None, on: date) -> bool:
    return last_donation is None or (on - last_donation).days >= ELIGIBILITY_DAYS


def compatibility_score(recipient: str, donor_types: pd.Series) -> np.ndarray:
    """Exact match preferred; O- is penalised for non-O- patients to conserve it."""
    exact = donor_types.to_numpy() == recipient
    universal = donor_types.to_numpy() == "O-"
    return np.where(exact, 1.0, np.where(universal, 0.55, 0.8))


def availability_slot(avail: pd.Series, on: date) -> np.ndarray:
    weekend = pd.Timestamp(on).dayofweek >= 5
    return avail.map({"Available now": 1.0,
                      "Evenings & weekends": 0.85 if weekend else 0.6,
                      "Paused": 0.0}).fillna(0).to_numpy()


def _score(df, recipient, lat, lon, on, weights, dist_scale_km=15.0):
    df = df.copy()
    df["distance_km"] = haversine_km(lat, lon, df.lat.to_numpy(), df.lon.to_numpy())
    compat = compatibility_score(recipient, df.blood_type)
    dist = np.exp(-df.distance_km.to_numpy() / dist_scale_km)
    avail = 0.6 * availability_slot(df.availability, on) + 0.4 * df.response_rate.to_numpy()
    contacted = pd.to_datetime(df.get("last_contacted"))
    recent = ((pd.Timestamp(on) - contacted).dt.days < 7).fillna(False).to_numpy()
    fatigue = np.where(recent, 0.6, 1.0)              # don't ping the same people every day
    wc, wd, wa = weights
    df["score"] = (wc * compat + wd * dist + wa * avail) * fatigue
    df["recently_contacted"] = recent
    return df


def rank_donors(donors: pd.DataFrame, recipient: str, lat: float, lon: float, on: date,
                units: int = 1, urgency: str = "Urgent (today)", radius_km: float = 10,
                max_radius_km: float = 100, top_n: int = 25) -> tuple[pd.DataFrame, dict]:
    """Rank eligible, consenting, compatible donors near (lat, lon).

    Starts at `radius_km` and widens automatically until there are enough
    candidates (about 4 per unit, since most people don't answer).
    """
    df = donors if "eligible" in donors else add_eligibility(donors, on)
    pool = df[df.blood_type.isin(COMPATIBLE_DONORS[recipient]) & df.eligible
              & df.consent_contact.astype(bool) & (df.availability != "Paused")]
    pool = _score(pool, recipient, lat, lon, on, URGENCY_WEIGHTS[urgency])

    target = max(3, units * 4)
    radii = sorted({r for r in (radius_km, radius_km * 2, 25, 50, max_radius_km)
                    if radius_km <= r <= max_radius_km})
    used = radii[-1]
    for r in radii:
        if (pool.distance_km <= r).sum() >= target:
            used = r
            break
    cand = pool[pool.distance_km <= used].sort_values("score", ascending=False)
    top = cand.head(top_n)
    meta = {
        "eligible_compatible_nationwide": int(len(pool)),
        "radius_used_km": used,
        "radius_expanded": used > radius_km,
        "candidates_in_radius": int(len(cand)),
        "exact_matches_in_radius": int((cand.blood_type == recipient).sum()),
        "target_contacts": target,
        "expected_acceptances": round(float(top.head(target).response_rate.sum() * 0.8), 1),
        "excluded_ineligible_90d": int((df.blood_type.isin(COMPATIBLE_DONORS[recipient]) & ~df.eligible).sum()),
    }
    return top, meta


def plan_recurring(patient: pd.Series, donors: pd.DataFrame, hospitals: pd.DataFrame,
                   on: date, sessions: int = 4, radius_km: float = 30) -> tuple[pd.DataFrame, dict]:
    """Build a 'donor circle' for a repeat-transfusion patient.

    Assigns primary donors to each upcoming session so that nobody is asked to
    donate within 90 days of their last donation (or of an earlier assignment),
    plus one backup per session.
    """
    hosp = hospitals.set_index("hospital_id").loc[patient.hospital_id]
    df = add_eligibility(donors, on)
    pool = df[df.blood_type.isin(COMPATIBLE_DONORS[patient.blood_type])
              & df.consent_contact.astype(bool) & (df.availability != "Paused")]
    pool = _score(pool, patient.blood_type, hosp.lat, hosp.lon, on,
                  URGENCY_WEIGHTS["Planned (surgery / scheduled)"])
    pool = pool[pool.distance_km <= radius_km].sort_values("score", ascending=False)

    free_from = {d: (pd.Timestamp(n) if pd.notna(n) else pd.Timestamp.min)
                 for d, n in zip(pool.donor_id, pool.next_eligible)}
    start = pd.Timestamp(patient.next_transfusion)
    dates = [start + pd.Timedelta(days=int(patient.interval_days) * k) for k in range(sessions)]
    units = int(patient.units_per_session)

    rows, uncovered = [], 0
    for d in dates:
        primaries, backup = [], None
        for row in pool.itertuples():
            if free_from[row.donor_id] > d or row.donor_id in primaries:
                continue
            if len(primaries) < units:
                primaries.append(row.donor_id)
                free_from[row.donor_id] = d + pd.Timedelta(days=ELIGIBILITY_DAYS)
            else:
                # a backup may be called in, so it also blocks the next 90 days
                backup = row.donor_id
                free_from[row.donor_id] = d + pd.Timedelta(days=ELIGIBILITY_DAYS)
                break
        uncovered += units - len(primaries)
        for i, did in enumerate(primaries + ([backup] if backup else [])):
            rows.append({"session": d.date(), "role": "Primary" if i < len(primaries) else "Backup",
                         "donor_id": did})
    sched = pd.DataFrame(rows)
    if not sched.empty:
        sched = sched.merge(pool[["donor_id", "blood_type", "distance_km", "response_rate", "availability"]],
                            on="donor_id", how="left")
    summary = {
        "compatible_donors_in_radius": int(len(pool)),
        "recommended_circle_size": int(math.ceil(units * ELIGIBILITY_DAYS / patient.interval_days) * 2),
        "donors_assigned": int(sched.donor_id.nunique()) if len(sched) else 0,
        "sessions": sessions,
        "uncovered_units": int(uncovered),
    }
    return sched, summary
