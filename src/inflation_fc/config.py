"""Project-wide settings: paths, series definitions and model choices.

Change values here rather than inside the modules, so every script and the
Streamlit app use the same assumptions.
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
PANEL_PATH = DATA_PROCESSED / "india_macro_monthly.csv"
PANEL_META_PATH = DATA_PROCESSED / "panel_meta.json"
RESULTS = ROOT / "results"
FIGURES = ROOT / "reports" / "figures"

# Sample window. CPI-Combined (base 2012=100) starts Jan 2011, so the first
# year-on-year inflation value is Jan 2012.
START = "2011-01-01"

# RBI's flexible inflation targeting: 4% target with a 2-6% tolerance band,
# notified in August 2016; the first MPC meeting was October 2016.
INFLATION_TARGET = 4.0
TOLERANCE_BAND = (2.0, 6.0)
FIT_BREAK_DATE = "2016-09-01"

# COVID months that distort year-on-year growth rates (lockdown collapse in
# 2020 and the mirror-image base effect in 2021). Used as dummy variables.
# In the VAR each month gets its own impulse dummy, which removes that
# observation's influence on the estimates (one shared dummy cannot fit both
# the 2020 collapse and the 2021 rebound).
COVID_MONTHS = ["2020-03-01", "2020-04-01", "2020-05-01", "2020-06-01", "2020-07-01",
                "2020-08-01", "2020-09-01", "2021-03-01", "2021-04-01", "2021-05-01",
                "2021-06-01"]

# Raw series. Each one is a CSV in data/raw/ with two columns: date,value.
#   required: the pipeline stops if it is missing.
#   fred:     FRED series id used as an automatic fallback / source.
SERIES = {
    "cpi": {
        "file": "cpi_combined_index.csv",
        "label": "CPI Combined, all-India general index (2012=100)",
        "source": "MoSPI / RBI DBIE",
        "required": True,
        "fred": "INDCPIALLMINMEI",  # OECD MEI; historically CPI-IW, fallback only
        "how": "last",
    },
    "repo": {
        "file": "repo_rate.csv",
        "label": "Policy repo rate (%), month-end",
        "source": "RBI DBIE / RBI press releases",
        "required": True,
        "fred": None,
        "how": "last",
    },
    "iip": {
        "file": "iip_general_index.csv",
        "label": "Index of Industrial Production, general (2011-12=100)",
        "source": "MoSPI",
        "required": True,
        "fred": "INDPROINDMISMEI",
        "how": "last",
    },
    "inr": {
        "file": "usd_inr.csv",
        "label": "INR per USD, monthly average",
        "source": "RBI reference rate (DBIE) / FRED EXINUS",
        "required": True,
        "fred": "EXINUS",
        "how": "mean",
    },
    "oil": {
        "file": "brent_usd.csv",
        "label": "Brent crude, USD per barrel, monthly average",
        "source": "FRED POILBREUSDM (IMF)",
        "required": True,
        "fred": "POILBREUSDM",
        "how": "mean",
    },
    "credit": {
        "file": "scb_nonfood_credit.csv",
        "label": "Scheduled commercial banks, non-food credit (Rs crore)",
        "source": "RBI DBIE, Money and Banking",
        "required": False,
        "fred": None,
        "how": "last",  # fortnightly data -> last reporting Friday of the month
    },
    "wacr": {
        "file": "wacr.csv",
        "label": "Weighted average call rate (%)",
        "source": "RBI DBIE",
        "required": False,
        "fred": "IRSTCI01INM156N",
        "how": "mean",
    },
}

# Forecasting setup
HORIZONS = [1, 3, 12]
BACKTEST_START = "2018-01-01"   # first forecast origin for the rolling backtest
SARIMA_SEASONAL_PERIOD = 12

# VAR: Cholesky ordering, most exogenous first. Credit is dropped
# automatically when the series is not available.
# The output gap is used rather than IIP y-o-y growth: y-o-y IIP swings from
# -57% (Apr 2020) to +134% (Apr 2021) on base effects, which wrecks a VAR.
# In the backtest the gap is re-estimated at each origin from data up to that
# month only (a real-time, one-sided estimate), so there is no look-ahead.
VAR_ORDER = ["oil_yoy", "output_gap", "inflation", "repo", "credit_yoy", "inr_yoy"]
VAR_MAXLAGS = 6        # 6 variables x 12 lags would be ~70 parameters per equation
IRF_HORIZON = 24

RANDOM_STATE = 42
