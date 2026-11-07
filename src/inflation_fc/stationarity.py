"""Unit-root and stationarity tests.

ADF and KPSS have opposite null hypotheses, so they are read together:
  ADF   H0: unit root (non-stationary)  -> small p-value = stationary
  KPSS  H0: stationary                  -> small p-value = non-stationary
Zivot-Andrews is an ADF-type test that allows one structural break chosen by
the data, useful because Indian inflation shifted after 2016.
"""
from __future__ import annotations

import warnings

import pandas as pd
from statsmodels.tsa.stattools import adfuller, kpss, zivot_andrews

DEFAULT_VARS = {
    "log_cpi": "Log CPI (level)",
    "inflation": "CPI inflation, y-o-y %",
    "repo": "Repo rate",
    "output_gap": "Output gap (IIP)",
    "iip_yoy": "IIP growth, y-o-y %",
    "oil_yoy": "Brent, y-o-y %",
    "inr_yoy": "INR/USD, y-o-y %",
    "credit_yoy": "Non-food credit growth, y-o-y %",
}


def _verdict(adf_p: float, kpss_p: float, alpha: float = 0.05) -> str:
    adf_stat = adf_p < alpha
    kpss_stat = kpss_p >= alpha
    if adf_stat and kpss_stat:
        return "Stationary"
    if not adf_stat and not kpss_stat:
        return "Unit root"
    return "Inconclusive"


def test_series(s: pd.Series, regression: str = "c") -> dict:
    s = s.dropna()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # KPSS warns when p is outside its table
        adf_stat, adf_p, adf_lags, *_ = adfuller(s, regression=regression, autolag="AIC")
        kpss_stat, kpss_p, *_ = kpss(s, regression=regression, nlags="auto")
        try:
            za = zivot_andrews(s, regression="c", autolag="AIC")
            za_stat, za_p, za_break = za[0], za[1], s.index[za[4]]
        except Exception:
            za_stat, za_p, za_break = float("nan"), float("nan"), None
    return {
        "n": len(s),
        "adf_stat": adf_stat, "adf_p": adf_p, "adf_lags": adf_lags,
        "kpss_stat": kpss_stat, "kpss_p": kpss_p,
        "za_stat": za_stat, "za_p": za_p,
        "za_break": f"{za_break:%Y-%m}" if za_break is not None else "",
        "verdict": _verdict(adf_p, kpss_p),
    }


def stationarity_table(panel: pd.DataFrame, variables: dict | None = None) -> pd.DataFrame:
    variables = variables or DEFAULT_VARS
    rows = []
    for col, label in variables.items():
        if col not in panel or panel[col].notna().sum() < 36:
            continue
        r = test_series(panel[col])
        rows.append({"variable": col, "description": label, **r})
        # First difference as well, for anything that looks non-stationary.
        if r["verdict"] != "Stationary":
            rd = test_series(panel[col].diff())
            rows.append({"variable": f"d_{col}", "description": f"{label}, first difference", **rd})
    return pd.DataFrame(rows)
