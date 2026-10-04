"""Qatra: blood donor matching and shortage forecasting for Pakistan.

    streamlit run app.py

ALL DATA IN THIS DEMO IS SYNTHETIC. No real donors, patients or hospitals.
"""
from __future__ import annotations

import hashlib
import uuid
from datetime import date, datetime, timedelta

import altair as alt
import folium
import numpy as np
import pandas as pd
import streamlit as st
from folium.plugins import HeatMap
from streamlit_folium import st_folium

from qatra.config import (BLOOD_TYPES, CITIES, ELIGIBILITY_DAYS, MAX_AGE, MIN_AGE,
                          MIN_WEIGHT_KG)
from qatra.data import load_all
from qatra.forecast import (ShortageForecaster, prophet_available, prophet_forecast,
                            shortage_table, to_weekly)
from qatra.matching import (COMPATIBLE_DONORS, URGENCY_WEIGHTS, add_eligibility,
                            plan_recurring, rank_donors)
from qatra.privacy import (ContactRelay, RateLimitExceeded, coarsen, distance_bucket,
                           donor_message, hospital_view, likelihood_label, request_ref)

st.set_page_config(page_title="Qatra · blood matching", page_icon="🩸", layout="wide")

RISK_COLOR = {"High": "#c0392b", "Watch": "#e67e22", "OK": "#27ae60"}
TODAY = date.today()


# ---------------------------------------------------------------- data & models
@st.cache_data(show_spinner="Loading synthetic data…")
def load():
    return load_all()


@st.cache_resource(show_spinner="Training shortage forecaster…")
def models():
    d = load()
    weekly = to_weekly(d["history"])
    f = ShortageForecaster().fit(weekly)
    return weekly, f.forecast(8), f.backtest(12)


def state():
    ss = st.session_state
    ss.setdefault("new_donors", [])
    ss.setdefault("deleted_ids", set())
    ss.setdefault("relay", ContactRelay(max_per_hour=60))
    ss.setdefault("match", None)
    ss.setdefault("contact_vault_size", 0)
    return ss


def donors_now() -> pd.DataFrame:
    ss = state()
    df = load()["donors"]
    if ss.new_donors:
        df = pd.concat([df, pd.DataFrame(ss.new_donors)], ignore_index=True)
    df = df[~df.donor_id.isin(ss.deleted_ids)]
    return add_eligibility(df, TODAY)


def show_map(m: folium.Map, height: int = 480):
    try:
        st_folium(m, height=height, use_container_width=True, returned_objects=[])
    except TypeError:          # older streamlit-folium
        st_folium(m, height=height, width=None, returned_objects=[])


def synthetic_banner():
    st.info("**Demo on synthetic data.** Every donor, patient, hospital and history record here is "
            "generated. Hospitals see request-scoped references only, never names, numbers or exact "
            "locations.", icon="🧪")


# ---------------------------------------------------------------- pages
def page_dashboard():
    st.title("🩸 Hospital dashboard")
    synthetic_banner()
    d = load()
    _, fc, _ = models()
    donors = donors_now()
    risk = shortage_table(fc, d["inventory"], weeks=4)
    hospitals, patients = d["hospitals"], d["patients"]

    focus = st.selectbox("Focus city", ["All Pakistan"] + list(CITIES))
    city_filter = (lambda df: df) if focus == "All Pakistan" else (lambda df: df[df.city == focus])

    dn, rk = city_filter(donors), city_filter(risk)
    usable = dn[dn.eligible & (dn.availability != "Paused")]
    upcoming = city_filter(patients)
    upcoming = upcoming[upcoming.next_transfusion <= pd.Timestamp(TODAY + timedelta(days=7))]

    c = st.columns(5)
    c[0].metric("Registered donors", f"{len(dn):,}")
    c[1].metric("Eligible & available today", f"{len(usable):,}",
                help=f"Not paused and last donation ≥ {ELIGIBILITY_DAYS} days ago")
    c[2].metric("Rh-negative ready", f"{usable.blood_type.str.endswith('-').sum():,}")
    c[3].metric("High-risk shortages (4 wk)", int((rk.risk == "High").sum()),
                help="City × blood type where expected donations + stock cover < 85% of forecast demand")
    c[4].metric("Thalassemia transfusions (7 d)", len(upcoming))

    left, right = st.columns([3, 2])
    with left:
        st.subheader("Donor density & shortage risk")
        center = ([30.2, 70.0], 5) if focus == "All Pakistan" else \
            ([CITIES[focus]["lat"], CITIES[focus]["lon"]], 11)
        m = folium.Map(location=center[0], zoom_start=center[1], tiles="OpenStreetMap")
        HeatMap(usable[["lat", "lon"]].values.tolist(), radius=14, blur=18,
                name="Eligible donors (aggregated heatmap)").add_to(m)
        risk_layer = folium.FeatureGroup(name="City shortage risk")
        order = {"High": 2, "Watch": 1, "OK": 0}
        for city, g in risk.groupby("city"):
            worst = max(g.risk, key=order.get)
            short = g[g.risk != "OK"].sort_values("coverage")
            lines = "".join(f"<br>{r.blood_type}: {r.risk} ({r.coverage:.0%} covered)"
                            for r in short.itertuples()) or "<br>All groups covered"
            folium.CircleMarker(
                [CITIES[city]["lat"], CITIES[city]["lon"]],
                radius=6 + CITIES[city]["weight"] ** 0.7 * 2, color=RISK_COLOR[worst],
                fill=True, fill_opacity=0.35, weight=2,
                popup=folium.Popup(f"<b>{city}</b> — next 4 weeks{lines}", max_width=260),
                tooltip=f"{city}: {worst}").add_to(risk_layer)
        risk_layer.add_to(m)
        hosp_layer = folium.FeatureGroup(name="Hospitals (synthetic)")
        for h in hospitals.itertuples():
            folium.Marker([h.lat, h.lon], tooltip=f"{h.hospital_id} · {h.name}",
                          icon=folium.Icon(color="red", icon="plus", prefix="fa")).add_to(hosp_layer)
        hosp_layer.add_to(m)
        folium.LayerControl(collapsed=True).add_to(m)
        show_map(m, 500)
        st.caption("Donors appear only as an aggregated heatmap built from ~1 km grid cells. "
                   "No individual donor pins exist anywhere in the app.")

    with right:
        st.subheader("Shortage alerts · next 4 weeks")
        alerts = rk[rk.risk != "OK"].sort_values(["risk", "shortfall"], ascending=[True, False]).copy()
        alerts["coverage"] = (alerts.coverage * 100).round()
        st.dataframe(
            alerts[["city", "blood_type", "risk", "demand", "supply", "stock", "coverage", "shortfall"]]
            .rename(columns={"city": "City", "blood_type": "Type", "risk": "Risk",
                             "demand": "Forecast demand", "supply": "Expected donations",
                             "stock": "Stock now", "coverage": "Coverage", "shortfall": "Gap (units)"}),
            hide_index=True, height=300,
            column_config={"Coverage": st.column_config.ProgressColumn(format="%.0f%%", min_value=0,
                                                                       max_value=150)})
        if len(alerts):
            top = alerts.sort_values("shortfall", ascending=False).iloc[0]
            st.warning(f"**Suggested action:** run a targeted drive for **{top.blood_type}** in "
                       f"**{top.city}**: forecast gap of about {int(top.shortfall)} units. "
                       f"{int(((usable.city == top.city) & (usable.blood_type == top.blood_type)).sum())} "
                       "eligible registered donors can be invited through the relay.")

        st.subheader("Upcoming thalassemia transfusions")
        st.dataframe(upcoming.sort_values("next_transfusion")[
            ["patient_id", "blood_type", "hospital_id", "units_per_session", "next_transfusion"]]
            .rename(columns={"patient_id": "Patient ID", "blood_type": "Type", "hospital_id": "Hospital",
                             "units_per_session": "Units", "next_transfusion": "Date"}),
            hide_index=True, height=220,
            column_config={"Date": st.column_config.DateColumn(format="ddd D MMM")})


def page_match():
    st.title("🔎 Find donors")
    synthetic_banner()
    d = load()
    hospitals = d["hospitals"]
    tab_urgent, tab_recurring = st.tabs(["Urgent / surgical request", "Thalassemia donor circle"])
    with tab_urgent:
        urgent_request(hospitals)
    with tab_recurring:
        recurring_circle(d, hospitals)


def urgent_request(hospitals: pd.DataFrame):
    with st.form("request"):
        c1, c2, c3 = st.columns(3)
        hosp_id = c1.selectbox("Requesting hospital", hospitals.hospital_id,
                               format_func=lambda h: f"{h} · {hospitals.set_index('hospital_id').name[h]}")
        recipient = c2.selectbox("Patient blood type", BLOOD_TYPES, index=BLOOD_TYPES.index("B+"))
        units = c3.number_input("Units needed", 1, 10, 2)
        c4, c5, c6 = st.columns(3)
        urgency = c4.selectbox("Urgency", list(URGENCY_WEIGHTS), index=0)
        reason = c5.selectbox("Reason", ["Emergency / trauma", "Surgery", "Dengue (platelets)*",
                                         "Obstetric", "Thalassemia (one-off)"])
        radius = c6.slider("Start search radius (km)", 5, 50, 10,
                           help="Widens automatically to 25/50/100 km if too few donors")
        go = st.form_submit_button("Find eligible donors", type="primary")
    if reason.startswith("Dengue"):
        st.caption("*MVP matches whole blood / red cells. Platelet donation (apheresis) has "
                   "different intervals and is listed as future work.")

    if go:
        h = hospitals.set_index("hospital_id").loc[hosp_id]
        top, meta = rank_donors(donors_now(), recipient, h.lat, h.lon, TODAY, units=int(units),
                                urgency=urgency, radius_km=radius)
        req_id = f"REQ-{datetime.now():%y%m%d%H%M%S}-{uuid.uuid4().hex[:4].upper()}"
        view, ref_map = hospital_view(top, req_id, recipient)
        st.session_state.match = dict(req_id=req_id, hosp_id=hosp_id, recipient=recipient,
                                      units=int(units), meta=meta, view=view, ref_map=ref_map,
                                      top=top, h=(h.lat, h.lon), notified=None)

    mt = st.session_state.match
    if not mt:
        st.caption("Compatible groups are fixed by ABO/Rh rules, e.g. "
                   f"B+ can receive {', '.join(COMPATIBLE_DONORS['B+'])}.")
        return
    meta = mt["meta"]
    st.markdown(f"#### Request `{mt['req_id']}` · {mt['recipient']} · {mt['units']} unit(s)")
    c = st.columns(4)
    c[0].metric("Candidates in radius", meta["candidates_in_radius"])
    c[1].metric("Radius used", f"{meta['radius_used_km']:.0f} km",
                "auto-expanded" if meta["radius_expanded"] else None, delta_color="off")
    c[2].metric("Expected acceptances", meta["expected_acceptances"],
                help=f"From the top {meta['target_contacts']} donors' historical response rates")
    c[3].metric("In 90-day cooldown", meta["excluded_ineligible_90d"],
                help="Compatible donors nationwide hidden from matching because they donated "
                     f"less than {ELIGIBILITY_DAYS} days ago")
    if meta["expected_acceptances"] < mt["units"]:
        st.error("Likely not enough acceptances. Alert the nearest blood bank and widen the radius.")

    left, right = st.columns([3, 2])
    with left:
        st.dataframe(mt["view"], hide_index=True, height=380)
        n_rows = len(mt["view"])
        n_notify = n_rows if n_rows <= 1 else st.slider(
            "Notify the top N donors through the relay", 1, n_rows,
            min(meta["target_contacts"], n_rows))
        if st.button(f"📨 Send consent-based request to {n_notify} donors", type="primary",
                     disabled=mt["view"].empty):
            refs = dict(list(mt["ref_map"].items())[:n_notify])
            try:
                res = st.session_state.relay.notify(mt["hosp_id"], mt["req_id"], refs, donors_now(),
                                                    mt["recipient"], {},
                                                    seed=int(hashlib.sha256(mt["req_id"].encode()).hexdigest()[:8], 16))
                mt["notified"] = res
            except RateLimitExceeded as e:
                st.error(f"Blocked by rate limit: {e}")
        if mt.get("notified") is not None:
            res = mt["notified"]
            acc = int((res.Status == "Accepted").sum())
            (st.success if acc >= mt["units"] else st.warning)(
                f"{acc} accepted · {int((res.Status == 'Declined').sum())} declined · "
                f"{int((res.Status == 'No response yet').sum())} pending (simulated responses)")
            st.dataframe(res, hide_index=True)
    with right:
        m = folium.Map(location=mt["h"], zoom_start=11, tiles="OpenStreetMap")
        folium.Circle(mt["h"], radius=meta["radius_used_km"] * 1000, color="#c0392b",
                      fill=False, dash_array="6").add_to(m)
        folium.Marker(mt["h"], tooltip=mt["hosp_id"],
                      icon=folium.Icon(color="red", icon="plus", prefix="fa")).add_to(m)
        if len(mt["top"]):
            HeatMap(mt["top"][["lat", "lon"]].values.tolist(), radius=18, blur=22).add_to(m)
        show_map(m, 380)
        if len(mt["view"]):
            st.markdown("**What the donor receives (SMS / WhatsApp template):**")
            st.code(donor_message(mt["recipient"], mt["hosp_id"], mt["view"].Distance.iloc[0]),
                    language=None)



def recurring_circle(d: dict, hospitals: pd.DataFrame):
    st.markdown("Repeat-transfusion patients need the same blood every 3–4 weeks. A **donor circle** "
                "spreads that load so no one is asked inside their 90-day window, with a backup per session.")
    patients = d["patients"].sort_values("next_transfusion")
    pid = st.selectbox("Patient", patients.patient_id, format_func=lambda p: (
        lambda r: f"{p} · {r.blood_type} · {r.city} · every {r.interval_days} d · "
                  f"{r.units_per_session} unit(s)")(patients.set_index('patient_id').loc[p]))
    sessions = st.slider("Sessions to plan", 2, 8, 4)
    patient = patients.set_index("patient_id").loc[pid]
    sched, summ = plan_recurring(patient, donors_now(), hospitals, TODAY, sessions=sessions)
    c = st.columns(4)
    c[0].metric("Compatible donors ≤ 30 km", summ["compatible_donors_in_radius"])
    c[1].metric("Recommended circle size", summ["recommended_circle_size"],
                help="ceil(units × 90 / interval) × 2 for no-shows")
    c[2].metric("Donors assigned", summ["donors_assigned"])
    c[3].metric("Uncovered units", summ["uncovered_units"])
    if summ["uncovered_units"]:
        st.error("Not enough compatible donors nearby for every session. Flag this patient for a "
                 "targeted recruitment drive or a regional blood bank allocation.")
    if len(sched):
        show = pd.DataFrame({
            "Session": sched.session, "Role": sched.role,
            "Donor ref": [request_ref(x, f"CIRCLE-{pid}") for x in sched.donor_id],
            "Type": sched.blood_type,
            "Distance": [distance_bucket(k) for k in sched.distance_km],
            "Availability": sched.availability,
            "Response likelihood": [likelihood_label(r) for r in sched.response_rate]})
        st.dataframe(show, hide_index=True)
        st.caption("Circle references stay stable for this patient so the coordinator can track "
                   "sessions, but they differ from the donor's references in any other request.")


def page_forecast():
    st.title("📈 Shortage forecast")
    synthetic_banner()
    d = load()
    weekly, fc, bt = models()
    risk = shortage_table(fc, d["inventory"], weeks=4)

    st.subheader("Where will blood run short? (next 4 weeks)")
    heat = alt.Chart(risk).mark_rect().encode(
        x=alt.X("blood_type:N", sort=BLOOD_TYPES, title="Blood type"),
        y=alt.Y("city:N", sort=list(CITIES), title=None),
        color=alt.Color("coverage:Q", title="Coverage",
                        scale=alt.Scale(domain=[0.5, 1, 1.5], range=["#c0392b", "#f7f7f7", "#27ae60"],
                                        clamp=True)),
        tooltip=["city", "blood_type", "risk", "demand", "demand_high", "supply", "stock",
                 alt.Tooltip("coverage:Q", format=".0%"), "shortfall"])
    text = alt.Chart(risk).mark_text(fontSize=11).encode(
        x=alt.X("blood_type:N", sort=BLOOD_TYPES), y=alt.Y("city:N", sort=list(CITIES)),
        text=alt.Text("coverage:Q", format=".0%"))
    st.altair_chart((heat + text).properties(height=380), use_container_width=True)
    st.caption("Coverage = (expected donations + current stock) ÷ median forecast demand. "
               "Red < 85% is High risk, below 100% is Watch.")

    st.subheader("Drill down")
    c1, c2, c3 = st.columns(3)
    city = c1.selectbox("City", list(CITIES), index=1)
    btype = c2.selectbox("Blood type", BLOOD_TYPES, index=BLOOD_TYPES.index("O-"))
    engines = ["scikit-learn (global model)"] + (["Prophet (this series)"] if prophet_available() else [])
    engine = c3.selectbox("Engine", engines, help="Install `prophet` to enable the Prophet engine")

    hist = weekly[(weekly.city == city) & (weekly.blood_type == btype)].tail(78)
    hist_long = hist.melt(id_vars="week", value_vars=["demand", "supply"], var_name="series",
                          value_name="units")
    hist_long["series"] = hist_long.series.map({"demand": "Demand (actual)", "supply": "Donations (actual)"})
    if engine.startswith("Prophet"):
        pf = prophet_forecast(weekly, city, btype, "demand", 8)
        fut = pd.DataFrame({"week": pf.ds, "demand_p10": pf.yhat_lower, "demand_p50": pf.yhat,
                            "demand_p90": pf.yhat_upper})
        sup = fc[(fc.city == city) & (fc.blood_type == btype)][["week", "supply_p50"]]
        fut = fut.merge(sup, on="week", how="left")
    else:
        fut = fc[(fc.city == city) & (fc.blood_type == btype)]

    color = alt.Color("series:N", title=None, scale=alt.Scale(
        domain=["Demand (actual)", "Donations (actual)", "Demand (forecast)", "Donations (forecast)"],
        range=["#c0392b", "#2c7fb8", "#e74c3c", "#5dade2"]))
    fut_long = fut.melt(id_vars="week", value_vars=["demand_p50", "supply_p50"], var_name="series",
                        value_name="units")
    fut_long["series"] = fut_long.series.map({"demand_p50": "Demand (forecast)",
                                              "supply_p50": "Donations (forecast)"})
    band = alt.Chart(fut).mark_area(opacity=0.2, color="#c0392b").encode(
        x="week:T", y=alt.Y("demand_p10:Q", title="Units per week"), y2="demand_p90:Q")
    fline = alt.Chart(fut_long).mark_line(strokeDash=[5, 3], point=True).encode(
        x="week:T", y="units:Q", color=color)
    hline = alt.Chart(hist_long).mark_line().encode(x=alt.X("week:T", title=None), y="units:Q",
                                                    color=color)
    st.altair_chart((band + hline + fline).properties(height=340), use_container_width=True)

    row = risk[(risk.city == city) & (risk.blood_type == btype)].iloc[0]
    c = st.columns(4)
    c[0].metric("Forecast demand (4 wk)", int(row.demand), f"pessimistic {int(row.demand_high)}",
                delta_color="off")
    c[1].metric("Expected donations (4 wk)", int(row.supply))
    c[2].metric("Stock now", int(row.stock), f"{row.days_of_stock} days", delta_color="off")
    c[3].metric("Risk", row.risk, f"{row.coverage:.0%} covered", delta_color="off")

    with st.expander("How the forecast works and how good it is"):
        st.markdown(f"""
**Model.** One global gradient-boosting model (scikit-learn `HistGradientBoostingRegressor`, Poisson loss)
across all {len(CITIES)} cities × 8 blood types, plus quantile models for an 80% band. Features: city,
blood type, week-of-year, same week last year, and calendar effects that drive blood in Pakistan:
**dengue season** (Sep–Nov demand surge), **Ramadan** (donations drop), **Eid** (road-accident spike),
and **summer heat** (donations dip). Supply is forecast the same way.

**Backtest** (train on all but the last {bt['holdout_weeks']} weeks, score on those weeks):

| | Model WAPE | Seasonal-naive WAPE |
|---|---|---|
| Demand | {bt['demand_wape_model']:.1%} | {bt['demand_wape_seasonal_naive']:.1%} |
| Supply | {bt['supply_wape_model']:.1%} | {bt['supply_wape_seasonal_naive']:.1%} |

80% interval actual coverage: **{bt['demand_interval_coverage']:.0%}**.
Errors are high for small series (rare types in small cities), which is exactly why we report a band.

**Data.** Synthetic, generated with these seasonal patterns built in. A production version would train on
blood-bank issue/collection logs (for example from provincial blood transfusion authorities) under a data-sharing agreement.
""")


def page_register():
    st.title("📝 Register as a donor")
    synthetic_banner()
    ss = state()
    with st.form("register", clear_on_submit=False):
        c1, c2, c3 = st.columns(3)
        btype = c1.selectbox("Blood type", BLOOD_TYPES, index=BLOOD_TYPES.index("B+"))
        city = c2.selectbox("City", list(CITIES), index=1)
        avail = c3.selectbox("Availability", ["Available now", "Evenings & weekends", "Paused"])
        c4, c5, c6 = st.columns(3)
        age = c4.number_input("Age", 16, 75, 25)
        weight = c5.number_input("Weight (kg)", 35, 150, 65)
        phone = c6.text_input("Mobile number", placeholder="03xx xxxxxxx",
                              help="Goes to the contact vault only. Never shown to hospitals.")
        never = st.checkbox("I have never donated blood")
        last = None if never else st.date_input("Last donation date", value=TODAY - timedelta(days=120),
                                                max_value=TODAY)
        st.markdown("**Consent** (both required, withdrawable any time)")
        k1 = st.checkbox("Use my blood type, city and approximate (~1 km) location to match me to "
                         "patients. My exact address is not collected.")
        k2 = st.checkbox("Contact me through the Qatra relay about urgent requests. I decide each time "
                         "whether to respond; my number is not shared with hospitals.")
        submitted = st.form_submit_button("Register", type="primary")

    if submitted:
        errors = []
        if not (MIN_AGE <= age <= MAX_AGE):
            errors.append(f"Donors must be {MIN_AGE}–{MAX_AGE} years old.")
        if weight < MIN_WEIGHT_KG:
            errors.append(f"Donors must weigh at least {MIN_WEIGHT_KG} kg.")
        digits = "".join(ch for ch in phone if ch.isdigit())
        if digits.startswith("92") and len(digits) == 12:          # +92 3xx xxxxxxx
            digits = "0" + digits[2:]
        if not (len(digits) == 11 and digits.startswith("03")):
            errors.append("Enter a Pakistani mobile number (03xx xxxxxxx or +92 3xx xxxxxxx).")
        if not (k1 and k2):
            errors.append("Both consents are required to join the registry.")
        if errors:
            for e in errors:
                st.error(e)
            return
        rng = np.random.default_rng()
        c = CITIES[city]
        lat, lon = coarsen(c["lat"] + rng.normal(0, 0.05), c["lon"] + rng.normal(0, 0.05))
        donor_id = "D" + uuid.uuid4().hex[:8].upper()
        # Contact vault (simulated): production stores the encrypted number in a separate
        # service keyed by donor_id. This demo keeps only a salted hash and discards the number.
        _ = hashlib.sha256((uuid.uuid4().hex + digits).encode()).hexdigest()
        ss.contact_vault_size += 1
        ss.new_donors.append({
            "donor_id": donor_id, "blood_type": btype, "city": city, "lat": float(lat), "lon": float(lon),
            "last_donation": pd.Timestamp(last) if last else pd.NaT, "availability": avail,
            "response_rate": 0.5, "last_contacted": pd.NaT, "consent_matching": True,
            "consent_contact": True, "registered_on": pd.Timestamp(TODAY)})
        st.success(f"Registered. Your private donor ID is **{donor_id}** (keep it to update or delete "
                   "your record).")
        if last and (TODAY - last).days < ELIGIBILITY_DAYS:
            nxt = last + timedelta(days=ELIGIBILITY_DAYS)
            st.warning(f"You donated {(TODAY - last).days} days ago. The 90-day rule keeps you out of "
                       f"matching until **{nxt:%d %b %Y}**; we won't contact you before then.")
        else:
            st.info("You're eligible now and will appear in matching for compatible requests nearby.")

    st.divider()
    st.subheader("Withdraw consent / delete my record")
    with st.form("delete"):
        did = st.text_input("Your donor ID")
        if st.form_submit_button("Delete my data"):
            ids = set(donors_now().donor_id)
            if did.strip().upper() in ids:
                ss.deleted_ids.add(did.strip().upper())
                ss.new_donors = [x for x in ss.new_donors if x["donor_id"] != did.strip().upper()]
                st.success("Deleted. You will not be matched or contacted again.")
            else:
                st.error("No record with that ID.")


def page_privacy():
    st.title("🔒 Privacy & data")
    synthetic_banner()
    ss = state()
    st.markdown("""
### Data used in this demo
Everything is **synthetic**, generated by `qatra/generate.py`: 6,000 donors, 11 cities, 3 years of daily
demand and donations, hospital stock, and thalassemia patients. Blood-group shares and seasonal patterns
(dengue, Ramadan, Eid, summer) are realistic assumptions, not real records.

### What a hospital can see
| Field | Hospital sees | Stored |
|---|---|---|
| Name | ✗ never collected | ✗ |
| Phone | ✗ | Separate encrypted contact vault, used only by the relay |
| Location | Distance bucket (e.g. "2–5 km") | ~1 km grid cell, never an address |
| Donor ID | ✗, a **request-scoped** reference (`R-xxxxxx`) that changes every request | Pseudonymous ID |
| Blood type, availability | ✓ | ✓ |
| Response history | Only as High/Medium/Low | Rate (0–1) |

### Consent-based contact
1. Hospital submits a request → the system ranks donors.
2. The **relay** sends the donor an SMS/WhatsApp template. The hospital never gets the number.
3. The donor accepts or declines. On accept, the donor receives the blood bank's address and helpline.
4. Every notification is audit-logged; each hospital is rate-limited (60 donors/hour) to stop list harvesting.
5. Donors already contacted in the last 7 days are de-prioritised to avoid fatigue.

### Donor rights
Explicit two-part opt-in, pause any time, delete with your donor ID. No data is sold or used for anything
but transfusion matching and aggregate forecasting.
""")
    st.subheader("Relay audit log (this session)")
    audit = ss.relay.audit_frame()
    if audit.empty:
        st.caption("No notifications sent yet. Send one from **Find donors**.")
    else:
        st.dataframe(audit, hide_index=True)
    st.caption(f"Contact vault entries created this session: {ss.contact_vault_size} "
               "(demo stores only a salted hash; numbers are discarded).")


# ---------------------------------------------------------------- nav
PAGES = {"Dashboard": page_dashboard, "Find donors": page_match, "Shortage forecast": page_forecast,
         "Register donor": page_register, "Privacy & data": page_privacy}

state()
with st.sidebar:
    st.markdown("## 🩸 Qatra")
    st.caption("Blood donor matching & shortage forecasting for Pakistan")
    page = st.radio("Go to", list(PAGES), label_visibility="collapsed")
    st.divider()
    meta = load()["meta"]
    st.caption(f"Synthetic dataset · generated {meta['as_of']} · seed {meta['seed']}")
PAGES[page]()