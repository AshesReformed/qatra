# 🩸 Qatra — blood donor matching & shortage forecasting for Pakistan

*Qatra* (قطرہ) means "a drop".

**Team:** Muhammad Hamza Yaseen (Team Lead) · Shaheer Azmat Shah · Fiza Ahmad · Zuhaib Amer

> **All data in this project is synthetic.** No real donors, patients, hospitals or blood-bank
> records are used. The generator (`qatra/generate.py`) builds a realistic but fake dataset so
> the system can be demoed without touching sensitive health data.

## The problem
Urgent blood requests in Pakistan (thalassemia patients, surgeries, emergencies, dengue season)
are handled through scattered WhatsApp and Facebook posts. Those posts expose donors' and
patients' phone numbers, reach the wrong people, ignore the 90-day donation gap, and fail
hardest for **rare (Rh-negative) types** and **repeat-transfusion patients**.

## What the MVP does
| Feature | How |
|---|---|
| **Donor registry** | Blood type, city, ~1 km location, availability, last donation. The **90-day rule** is enforced in matching and at registration. Age/weight checks, two-part consent, self-delete. |
| **Request matching** | ABO/Rh red-cell compatibility → hard filters (eligible, consented, not paused) → score = compatibility + distance + availability, weighted by urgency. Radius auto-widens 10→25→50→100 km until there are ~4 candidates per unit. Exact matches beat O-; O- is conserved. Donors contacted in the last 7 days are de-prioritised. |
| **Thalassemia donor circles** | For patients transfused every 21–28 days, schedules primary + backup donors across upcoming sessions so nobody is asked inside 90 days, and flags uncovered sessions. |
| **Shortage forecast** | Weekly demand and donations per city × blood type, 8 weeks ahead, with an 80% band. Risk = (expected donations + current stock) ÷ forecast demand over 4 weeks. |
| **Hospital dashboard** | KPIs, Folium map (aggregated donor heatmap, city risk circles, hospitals), shortage alerts with a suggested drive, upcoming thalassemia transfusions. |
| **Privacy layer** | Request-scoped donor references, distance buckets, contact relay, audit log, rate limit. See below. |

## Run it
```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```
The first launch generates `data/` (about 5 s) and trains the forecaster (about 10 s; cached after that).
Regenerate with `python -m qatra.generate --donors 6000 --seed 42`.

Tests: `python -m pytest -q` (90-day rule, compatibility, radius expansion, circle scheduling, privacy).

Optional: `pip install prophet` adds a Prophet engine on the forecast page.

## Project layout
```
app.py                 Streamlit UI (5 pages)
qatra/config.py        cities, blood-type shares, Ramadan/Eid/dengue calendar, rules
qatra/generate.py      synthetic data generator
qatra/data.py          loader (generates on first run)
qatra/matching.py      eligibility, compatibility, ranking, donor circles
qatra/forecast.py      weekly features, sklearn models, backtest, shortage table, Prophet option
qatra/privacy.py       hospital view, request-scoped refs, contact relay, audit, rate limit
tests/test_core.py     rule tests
```

## Synthetic data: what's in it
* **6,000 donors** across 11 cities, weighted by population. Blood-group shares are approximate
  Pakistani figures (B+ ≈ 32%, O+ ≈ 29%, A+ ≈ 22%, AB+ ≈ 8%, all Rh-negative ≈ 9%).
  No names, no phone numbers, location rounded to a ~1 km grid.
* **3 years of daily demand and donations** per city × type, with effects built in:
  dengue season (Sep–Nov demand surge, strongest in Lahore, Karachi, Rawalpindi, Islamabad),
  Ramadan (donations −35%), Eid (accident spike), summer heat (donations dip), donation camps.
  Rh-negative demand is over-represented relative to donors, so negatives run short.
* **27 hospitals** (named "Lahore Hospital A (synthetic)" etc.), current stock, and **140 thalassemia patients** (IDs only).

## Forecasting
One global `HistGradientBoostingRegressor` (Poisson loss) across all 88 series, plus quantile
models for p10/p90. Features: city, type, week-of-year, same-week-last-year lags, dengue / Ramadan /
Eid / summer intensity. Because lags are 51+ weeks, an 8-week forecast needs no recursion.

Backtest on the last 12 weeks of synthetic data (shown in the app):

| | Model WAPE | Seasonal-naive WAPE |
|---|---|---|
| Demand | ~23% | ~25% |
| Supply | ~22% | ~27% |

The 80% band covers ~70% of actuals, so it is slightly too narrow. Error is dominated by small
series (rare types in small cities). Be upfront about this with judges: the point of the forecast
is *ranking* where shortages are likely so drives can be planned 2–4 weeks early, not unit-exact numbers.

## Privacy: answers for judges
**"Isn't donor data sensitive?"** Yes, it is health data. That's why the demo is fully synthetic, and why the design minimises what is collected and who sees it.

**What does a hospital see?** Blood type, a match label (Exact / Compatible / Universal O-),
a distance bucket ("2–5 km"), availability, and response likelihood (High/Medium/Low).
**Never** a name, phone number, address, coordinates or permanent donor ID.

**How do donors get contacted then?** Through a **relay**. The hospital presses "send request";
the platform messages the donor (SMS/WhatsApp template); the donor accepts or declines. On accept,
the *donor* receives the blood bank's address and helpline. The donor's number is never handed to the hospital.

**Can a hospital scrape a donor list?** Donor references are **request-scoped**: the same donor
gets a different `R-xxxxxx` in every request, so references can't be joined across requests.
Each hospital is **rate-limited** (60 donors/hour) and every notification is **audit-logged**.

**Consent?** Two explicit opt-ins at registration (use for matching; contact via relay). Donors
can pause at any time, reply STOP, or delete their record with their private donor ID.

**Where do phone numbers live?** In production, in a separate encrypted contact vault that only
the relay service can read, keyed by pseudonymous donor ID. The demo discards numbers after validation.

**Location?** Stored on a ~1 km grid only. The map shows donors as an aggregated heatmap; there are no donor pins anywhere.

**Spam and fatigue?** Donors pinged in the last 7 days are de-prioritised, and the 90-day rule means nobody is asked when they can't donate.

## 3-minute demo script
1. **Dashboard** (30 s): "It's dengue season. 23 city × type pairs are high-risk over the next four weeks. O- is short almost everywhere; Lahore A+ has the biggest gap. Here's the suggested drive."
2. **Find donors** (60 s): Critical B+ request at a Lahore hospital → ranked list with no personal data → "send to top 8" → simulated acceptances. Point at the 90-day cooldown count and the SMS template.
3. **Thalassemia circle** (30 s): Pick a patient → donor circle across four sessions, nobody inside 90 days, a backup each time.
4. **Forecast** (30 s): Heatmap, then drill into Lahore O-: history, forecast band, backtest table.
5. **Privacy page** (30 s): The visibility table and the audit log of the request you just sent.

## Limitations and next steps
* Matches **whole blood / red cells** only; platelet (apheresis) and plasma rules differ, which matters in dengue season.
* Medical deferrals beyond the 90-day rule (haemoglobin, recent illness, medication, travel) are left to blood-bank screening.
* Gender-specific donation intervals are not modelled.
* Simulated relay. Production needs an SMS gateway or WhatsApp Business API, a real contact vault, hospital authentication and roles.
* Real forecasting would need blood-bank issue and collection logs under a data-sharing agreement, plus a review against Pakistan's data protection requirements in force at the time.
* Ramadan/Eid dates are approximate and should come from an official calendar.