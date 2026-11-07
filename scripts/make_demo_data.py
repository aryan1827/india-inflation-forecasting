"""Generate SYNTHETIC demo data so you can check that the code and the app run
before you have downloaded the real series.

The numbers are simulated, not real Indian data. Never report results from
them. They are written to data/demo_raw/, separate from data/raw/.

    python scripts/make_demo_data.py
    python scripts/run_all.py --demo
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from inflation_fc import config as C  # noqa: E402

DEMO_DIR = ROOT / "data" / "demo_raw"


def simulate(seed: int = 7, end: str = "2026-09-01") -> dict[str, pd.Series]:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2010-01-01", end, freq="MS")
    n = len(idx)
    t = np.arange(n)

    # Oil: log random walk with a 2014-15 crash, a 2020 collapse, a 2022 spike.
    shocks = rng.normal(0, 0.06, n)
    for d, s in {"2014-10-01": -0.15, "2014-12-01": -0.2, "2015-07-01": -0.15, "2020-03-01": -0.35,
                 "2020-04-01": -0.4, "2020-06-01": 0.35, "2021-03-01": 0.15, "2022-03-01": 0.25,
                 "2022-10-01": -0.15}.items():
        shocks[idx.get_loc(pd.Timestamp(d))] += s
    oil = pd.Series(np.exp(np.log(80) + np.cumsum(shocks) * 0.8), idx).clip(20, 140)

    # INR/USD: ~3% a year trend depreciation with noise.
    inr = pd.Series(46 * np.exp(0.0026 * t + np.cumsum(rng.normal(0, 0.012, n)) * 0.6), idx)

    # IIP: trend growth, March peak, COVID collapse.
    season = np.tile([0.0, -0.01, 0.08, -0.03, -0.01, -0.03, -0.03, -0.04, -0.03, -0.02, -0.01, 0.02], n // 12 + 1)[:n]
    log_iip = np.log(105) + 0.0035 * t + season + np.cumsum(rng.normal(0, 0.008, n)) * 0.5
    covid = {"2020-03-01": -0.18, "2020-04-01": -0.75, "2020-05-01": -0.4, "2020-06-01": -0.18,
             "2020-07-01": -0.1, "2020-08-01": -0.08, "2021-04-01": -0.12, "2021-05-01": -0.15}
    for d, s in covid.items():
        log_iip[idx.get_loc(pd.Timestamp(d))] += s
    iip = pd.Series(np.exp(log_iip), idx)

    # CPI: underlying annual inflation drifts from ~9% to ~4-5%, food-type
    # seasonality in m-o-m changes, and some oil/INR pass-through.
    trend = np.interp(t, [0, 36, 60, 96, 120, 144, 150, 176, n - 1], [9.5, 9.5, 5.5, 3.8, 6.0, 6.5, 5.0, 2.5, 3.5])
    seas = np.tile([-0.3, -0.4, -0.2, 0.2, 0.4, 0.6, 1.1, 0.6, 0.3, 0.4, 0.0, -0.6], n // 12 + 1)[:n]
    oil_mom = np.r_[0, np.diff(np.log(oil.values))] * 100
    inr_mom = np.r_[0, np.diff(np.log(inr.values))] * 100
    mom = trend / 12 + seas + 0.012 * oil_mom + 0.05 * inr_mom + rng.normal(0, 0.35, n)
    cpi = pd.Series(100 * np.exp(np.cumsum(mom) / 100), idx)
    cpi = cpi / cpi.loc["2012"].mean() * 100

    # Non-food credit: ~11-14% growth, slowing in 2020-21, faster after.
    g = np.interp(t, [0, 60, 120, 132, 150, 170, n - 1], [16, 11, 7, 6, 15, 16, 11]) / 1200
    credit = pd.Series(3.6e6 * np.exp(np.cumsum(g + rng.normal(0, 0.004, n))), idx)

    return {"cpi": cpi, "iip": iip, "inr": inr, "oil": oil, "credit": credit}


def main():
    DEMO_DIR.mkdir(parents=True, exist_ok=True)
    for name, s in simulate().items():
        df = s.round(4).rename("value").to_frame()
        df.index.name = "date"
        path = DEMO_DIR / C.SERIES[name]["file"]
        path.write_text("# SYNTHETIC DEMO DATA - not real\n" + df.to_csv(date_format="%Y-%m-%d"))
    # The repo rate seed file holds real RBI decisions; reuse it.
    shutil.copy(C.DATA_RAW / "repo_rate.csv", DEMO_DIR / "repo_rate.csv")
    print(f"Wrote synthetic demo series to {DEMO_DIR.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
