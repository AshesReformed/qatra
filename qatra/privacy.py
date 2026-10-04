"""Privacy layer: what a hospital is allowed to see, and the consent-based contact relay.

Principles enforced in code:
  * Hospitals never see donor names, phone numbers or exact locations.
  * Donor references are request-scoped: the same donor gets a different
    reference in every request, so a hospital cannot build a donor list.
  * Contact goes through a relay. The donor is notified, decides, and only on
    "accept" receives the blood bank's details. The donor's number is never
    passed to the hospital.
  * Every notification is audit-logged and rate-limited per hospital.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

GRID_DECIMALS = 2   # ~1.1 km at Pakistan's latitudes


def coarsen(lat, lon, decimals: int = GRID_DECIMALS):
    return np.round(lat, decimals), np.round(lon, decimals)


def request_ref(donor_id: str, scope_id: str) -> str:
    """One-time reference for a donor within a single request (or patient circle)."""
    return "R-" + hashlib.sha256(f"{scope_id}:{donor_id}".encode()).hexdigest()[:6].upper()


def distance_bucket(km: float) -> str:
    for limit, label in [(2, "< 2 km"), (5, "2-5 km"), (10, "5-10 km"),
                         (25, "10-25 km"), (50, "25-50 km")]:
        if km < limit:
            return label
    return "> 50 km"


def likelihood_label(rate: float) -> str:
    return "High" if rate >= 0.55 else "Medium" if rate >= 0.35 else "Low"


def match_label(recipient: str, donor: str) -> str:
    if donor == recipient:
        return "Exact"
    return "Universal O-" if donor == "O-" else "Compatible"


def hospital_view(ranked: pd.DataFrame, scope_id: str, recipient: str) -> tuple[pd.DataFrame, dict]:
    """Strip a ranked donor list down to what a hospital may see.

    Returns the public table and a server-side {ref: donor_id} map that only the
    relay uses. The map must never be rendered.
    """
    if ranked.empty:
        return pd.DataFrame(), {}
    refs = [request_ref(d, scope_id) for d in ranked.donor_id]
    view = pd.DataFrame({
        "Donor ref": refs,
        "Blood type": ranked.blood_type.values,
        "Match": [match_label(recipient, b) for b in ranked.blood_type],
        "Distance": [distance_bucket(k) for k in ranked.distance_km],
        "Availability": ranked.availability.values,
        "Response likelihood": [likelihood_label(r) for r in ranked.response_rate],
        "Score": np.round(ranked.score.values, 2),
    })
    return view, dict(zip(refs, ranked.donor_id))


class RateLimitExceeded(Exception):
    pass


@dataclass
class ContactRelay:
    """Simulated SMS/WhatsApp relay. In production this would call a messaging
    API from a server holding the contact vault; the app never sees numbers."""
    max_per_hour: int = 60
    audit: list = field(default_factory=list)

    def sent_last_hour(self, hospital_id: str, now: datetime) -> int:
        return sum(a["donors_notified"] for a in self.audit
                   if a["hospital_id"] == hospital_id and now - a["time"] < timedelta(hours=1))

    def notify(self, hospital_id: str, request_id: str, ref_map: dict, donors: pd.DataFrame,
               blood_type: str, distance_by_ref: dict, now: datetime | None = None,
               seed: int | None = None) -> pd.DataFrame:
        now = now or datetime.now()
        n = len(ref_map)
        if self.sent_last_hour(hospital_id, now) + n > self.max_per_hour:
            raise RateLimitExceeded(
                f"{hospital_id} would exceed {self.max_per_hour} donor notifications per hour.")
        rng = np.random.default_rng(seed)
        rates = donors.set_index("donor_id").response_rate
        rows = []
        for ref, donor_id in ref_map.items():
            u, rr = rng.random(), float(rates.get(donor_id, 0.3))
            if u < rr * 0.8:
                status, step = "Accepted", "Donor received the blood bank's address and helpline. Expect arrival."
            elif u < rr * 0.8 + 0.15:
                status, step = "Declined", "No details shared either way."
            else:
                status, step = "No response yet", "Relay sends one reminder after 30 minutes, then stops."
            rows.append({"Donor ref": ref, "Status": status, "Next step": step})
        self.audit.append({
            "time": now, "hospital_id": hospital_id, "request_id": request_id,
            "blood_type": blood_type, "donors_notified": n,
            "accepted": sum(r["Status"] == "Accepted" for r in rows),
            "purpose": "urgent transfusion request",
        })
        return pd.DataFrame(rows)

    def audit_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.audit)


def donor_message(blood_type: str, hospital_id: str, distance: str) -> str:
    return (f"Qatra: a patient at {hospital_id} ({distance} from you) needs {blood_type} blood. "
            "Reply 1 to accept or 2 to decline. Your number is not shared with the hospital. "
            "Reply STOP to leave the registry.")
