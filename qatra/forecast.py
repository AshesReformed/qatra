"""Weekly demand/supply forecasting and shortage risk.

Default engine: one global scikit-learn HistGradientBoosting model across all
city x blood-type series (Poisson loss for the point forecast, quantile loss
for an 80% interval). Features are calendar effects plus same-week-last-year
lags, so forecasts up to 51 weeks ahead need no recursion.

Optional engine: Prophet, per series, if `prophet` is installed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from .config import BLOOD_TYPES, CITIES, dengue_shape, eid_flag, ramadan_flag, summer_shape

CITY_LIST = list(CITIES)
FEATURES = ["city_code", "type_code", "woy_sin", "woy_cos", "dengue", "ramadan",
            "eid", "summer", "t", "lag52", "lag52_avg3"]
EPOCH = pd.Timestamp("2023-01-01")
HIGH_RISK_COVERAGE = 0.85


def to_weekly(daily: pd.DataFrame) -> pd.DataFrame:
    df = daily.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["week"] = df["date"].dt.to_period("W-SUN").dt.end_time.dt.normalize()
    g = (df.groupby(["city", "blood_type", "week"])
           .agg(demand=("demand", "sum"), supply=("supply", "sum"), days=("date", "nunique"))
           .reset_index())
    return g[g.days == 7].drop(columns="days").reset_index(drop=True)


def _week_calendar(weeks: pd.Series) -> pd.DataFrame:
    """Average daily calendar effects within each week (ending Sunday)."""
    weeks = pd.to_datetime(pd.Series(weeks).drop_duplicates())
    days = pd.date_range(weeks.min() - pd.Timedelta(days=6), weeks.max(), freq="D")
    cal = pd.DataFrame({"dengue_raw": dengue_shape(days), "ramadan": ramadan_flag(days),
                        "eid": eid_flag(days), "summer": summer_shape(days)})
    cal["week"] = days.to_period("W-SUN").end_time.normalize()
    return cal.groupby("week").mean().reset_index()


def build_frame(weekly: pd.DataFrame, target: str, horizon: int = 0) -> pd.DataFrame:
    """Feature frame for all history plus `horizon` future weeks (target NaN)."""
    w = weekly[["city", "blood_type", "week", target]].rename(columns={target: "y"})
    if horizon:
        last = w.week.max()
        fut_weeks = [last + pd.Timedelta(weeks=h) for h in range(1, horizon + 1)]
        fut = pd.MultiIndex.from_product([CITY_LIST, BLOOD_TYPES, fut_weeks],
                                         names=["city", "blood_type", "week"]).to_frame(index=False)
        fut["y"] = np.nan
        w = pd.concat([w, fut], ignore_index=True)
    w = w.sort_values(["city", "blood_type", "week"]).reset_index(drop=True)
    grp = w.groupby(["city", "blood_type"])["y"]
    w["lag52"] = grp.shift(52)
    w["lag52_avg3"] = (grp.shift(51) + grp.shift(52) + grp.shift(53)) / 3
    w = w.merge(_week_calendar(w.week), on="week", how="left")
    w["dengue"] = w.dengue_raw * w.city.map({c: v["dengue"] for c, v in CITIES.items()})
    woy = w.week.dt.isocalendar().week.astype(float).to_numpy()
    w["woy_sin"], w["woy_cos"] = np.sin(2 * np.pi * woy / 52), np.cos(2 * np.pi * woy / 52)
    w["t"] = (w.week - EPOCH).dt.days / 7.0
    w["city_code"] = w.city.map({c: i for i, c in enumerate(CITY_LIST)})
    w["type_code"] = w.blood_type.map({b: i for i, b in enumerate(BLOOD_TYPES)})
    return w


def _model(loss: str, **kw) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(loss=loss, max_iter=300, learning_rate=0.06,
                                         max_leaf_nodes=31, categorical_features=[0, 1],
                                         random_state=0, **kw)


def wape(y, yhat) -> float:
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    return float(np.abs(y - yhat).sum() / max(y.sum(), 1e-9))


class ShortageForecaster:
    def __init__(self):
        self.models: dict = {}
        self.weekly: pd.DataFrame | None = None

    def _fit_target(self, frame: pd.DataFrame, target: str, quantiles: bool):
        tr = frame[frame.y.notna() & frame.lag52_avg3.notna()]
        X, y = tr[FEATURES], tr.y
        self.models[(target, "p50")] = _model("poisson").fit(X, y)
        if quantiles:
            self.models[(target, "p10")] = _model("quantile", quantile=0.1).fit(X, y)
            self.models[(target, "p90")] = _model("quantile", quantile=0.9).fit(X, y)

    def fit(self, weekly: pd.DataFrame) -> "ShortageForecaster":
        self.weekly = weekly
        self._fit_target(build_frame(weekly, "demand"), "demand", quantiles=True)
        self._fit_target(build_frame(weekly, "supply"), "supply", quantiles=False)
        return self

    def forecast(self, horizon: int = 8) -> pd.DataFrame:
        out = None
        for target, qs in [("demand", ["p10", "p50", "p90"]), ("supply", ["p50"])]:
            f = build_frame(self.weekly, target, horizon)
            f = f[f.y.isna()].copy()
            for q in qs:
                f[f"{target}_{q}"] = np.clip(self.models[(target, q)].predict(f[FEATURES]), 0, None)
            cols = ["city", "blood_type", "week"] + [f"{target}_{q}" for q in qs]
            out = f[cols] if out is None else out.merge(f[cols], on=["city", "blood_type", "week"])
        out["demand_p10"] = np.minimum(out.demand_p10, out.demand_p50)
        out["demand_p90"] = np.maximum(out.demand_p90, out.demand_p50)
        return out.reset_index(drop=True)

    def backtest(self, holdout_weeks: int = 12) -> dict:
        """Refit on history minus the last `holdout_weeks`, score on the holdout."""
        cutoff = self.weekly.week.max() - pd.Timedelta(weeks=holdout_weeks)
        result = {"holdout_weeks": holdout_weeks}
        for target in ["demand", "supply"]:
            frame = build_frame(self.weekly, target)
            tr = frame[(frame.week <= cutoff) & frame.lag52_avg3.notna()]
            te = frame[frame.week > cutoff]
            m = _model("poisson").fit(tr[FEATURES], tr.y)
            pred = np.clip(m.predict(te[FEATURES]), 0, None)
            result[f"{target}_wape_model"] = round(wape(te.y, pred), 3)
            result[f"{target}_wape_seasonal_naive"] = round(wape(te.y, te.lag52), 3)
            if target == "demand":
                lo = _model("quantile", quantile=0.1).fit(tr[FEATURES], tr.y).predict(te[FEATURES])
                hi = _model("quantile", quantile=0.9).fit(tr[FEATURES], tr.y).predict(te[FEATURES])
                result["demand_interval_coverage"] = round(float(((te.y >= lo) & (te.y <= hi)).mean()), 3)
        return result


def shortage_table(fc: pd.DataFrame, inventory: pd.DataFrame, weeks: int = 4) -> pd.DataFrame:
    """Risk per city x type over the next `weeks`: expected donations + current stock vs demand."""
    first = sorted(fc.week.unique())[:weeks]
    agg = (fc[fc.week.isin(first)].groupby(["city", "blood_type"])
             .agg(demand=("demand_p50", "sum"), demand_high=("demand_p90", "sum"),
                  supply=("supply_p50", "sum")).reset_index())
    stock = inventory.groupby(["city", "blood_type"]).units.sum().rename("stock").reset_index()
    t = agg.merge(stock, on=["city", "blood_type"], how="left").fillna({"stock": 0})
    available = t.supply + t.stock
    t["coverage"] = available / t.demand.clip(lower=1e-9)
    t["shortfall"] = (t.demand - available).clip(lower=0)
    # High: expected donations + stock cover < 85% of median demand.
    # Watch: cover less than 100% of median demand.
    # (demand_high, the summed p90, is shown as the pessimistic case.)
    t["risk"] = np.select([t.coverage < HIGH_RISK_COVERAGE, t.coverage < 1.0],
                          ["High", "Watch"], "OK")
    t["days_of_stock"] = t.stock / (t.demand / (7 * weeks)).clip(lower=1e-9)
    return t.round({"demand": 0, "demand_high": 0, "supply": 0, "coverage": 2,
                    "shortfall": 0, "days_of_stock": 1})


def prophet_available() -> bool:
    try:
        import prophet  # noqa: F401
        return True
    except Exception:
        return False


def prophet_forecast(weekly: pd.DataFrame, city: str, blood_type: str, target: str = "demand",
                     horizon: int = 8) -> pd.DataFrame:
    """Per-series Prophet forecast with Ramadan/Eid/dengue regressors (optional engine)."""
    from prophet import Prophet

    s = weekly[(weekly.city == city) & (weekly.blood_type == blood_type)][["week", target]]
    s = s.rename(columns={"week": "ds", target: "y"})
    future_weeks = pd.date_range(s.ds.max() + pd.Timedelta(weeks=1), periods=horizon, freq="W-SUN")
    full = pd.concat([s, pd.DataFrame({"ds": future_weeks})], ignore_index=True)
    cal = _week_calendar(full.ds).rename(columns={"week": "ds"})
    full = full.merge(cal, on="ds", how="left")
    m = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False,
                interval_width=0.8)
    for reg in ["dengue_raw", "ramadan", "eid", "summer"]:
        m.add_regressor(reg)
    m.fit(full[full.y.notna()])
    out = m.predict(full[full.y.isna()].drop(columns="y"))
    out = out[["ds", "yhat_lower", "yhat", "yhat_upper"]].copy()
    num = ["yhat_lower", "yhat", "yhat_upper"]
    out[num] = out[num].clip(lower=0)
    return out
