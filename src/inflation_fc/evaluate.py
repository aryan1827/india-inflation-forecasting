"""Pseudo out-of-sample evaluation.

Rolling-origin (expanding window) backtest: at every month t from
BACKTEST_START, each model is re-estimated on data up to t only and asked to
forecast inflation at t+1, t+3 and t+12. Forecast errors are then compared
with RMSE/MAE and the Diebold-Mariano test.

Caveat: this uses the latest vintage of the
data, not the data RBI had in real time (CPI and IIP get revised), so it is
a pseudo real-time exercise.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from scipy import stats

from . import config as C

COVID_TARGETS = ("2020-03-01", "2021-09-01")


def rolling_backtest(panel: pd.DataFrame, models: list, horizons=C.HORIZONS,
                     start: str = C.BACKTEST_START, step: int = 1, verbose: bool = True) -> pd.DataFrame:
    infl = panel["inflation"].dropna()
    h_max = max(horizons)
    origins = infl.loc[start:].index[:-1][::step]
    rows = []
    t0 = time.time()
    for n, origin in enumerate(origins, 1):
        train = panel.loc[:origin]
        for m in models:
            try:
                if hasattr(m, "kind"):  # ML models: only fit the horizons we score
                    fc = m.forecast(train, h_max, horizons=horizons)
                else:
                    fc = m.forecast(train, h_max)
            except Exception as exc:
                if verbose:
                    print(f"    {m.name} failed at {origin:%Y-%m}: {exc}")
                continue
            for h in horizons:
                target = origin + pd.DateOffset(months=h)
                if target in fc.index:
                    rows.append({"origin": origin, "target": target, "h": h, "model": m.name,
                                 "forecast": float(fc.loc[target]),
                                 "actual": float(infl.get(target, np.nan))})
        if verbose and (n % 12 == 0 or n == len(origins)):
            print(f"    origin {origin:%Y-%m}  ({n}/{len(origins)}, {time.time() - t0:.0f}s)")
    out = pd.DataFrame(rows)
    out["error"] = out["actual"] - out["forecast"]
    return out


def _metrics(g: pd.DataFrame) -> pd.Series:
    e = g["error"].dropna()
    return pd.Series({"n": len(e), "rmse": np.sqrt((e ** 2).mean()),
                      "mae": e.abs().mean(), "bias": e.mean()})


def metrics_table(bt: pd.DataFrame, benchmark: str = "Random walk", exclude_covid: bool = False) -> pd.DataFrame:
    d = bt.dropna(subset=["actual"])
    if exclude_covid:
        lo, hi = map(pd.Timestamp, COVID_TARGETS)
        d = d[(d["target"] < lo) | (d["target"] > hi)]
    m = d.groupby(["h", "model"]).apply(_metrics, include_groups=False).reset_index()
    bench = m[m["model"] == benchmark].set_index("h")["rmse"]
    m["rel_rmse_vs_rw"] = m["rmse"] / m["h"].map(bench)
    return m.sort_values(["h", "rmse"]).reset_index(drop=True)


def diebold_mariano(e1: np.ndarray, e2: np.ndarray, h: int, power: int = 2) -> tuple[float, float]:
    """Diebold-Mariano test with the Harvey-Leybourne-Newbold small-sample
    correction. H0: equal accuracy. Negative statistic = model 1 more accurate.
    Returns (statistic, two-sided p-value)."""
    d = np.abs(e1) ** power - np.abs(e2) ** power
    T = len(d)
    d_bar = d.mean()
    # Long-run variance with h-1 autocovariances (h-step errors are MA(h-1)).
    gamma = [np.sum((d[k:] - d_bar) * (d[:T - k] - d_bar)) / T for k in range(h)]
    lrv = gamma[0] + 2 * sum(gamma[1:])
    if lrv <= 0:
        lrv = gamma[0]
    if lrv == 0:  # identical losses: no evidence either way
        return 0.0, 1.0
    dm = d_bar / np.sqrt(lrv / T)
    hln = np.sqrt((T + 1 - 2 * h + h * (h - 1) / T) / T)
    stat = dm * hln
    p = 2 * stats.t.sf(abs(stat), df=T - 1)
    return float(stat), float(p)


def dm_table(bt: pd.DataFrame, against: str = "Random walk", exclude_covid: bool = False) -> pd.DataFrame:
    d = bt.dropna(subset=["actual"])
    if exclude_covid:
        lo, hi = map(pd.Timestamp, COVID_TARGETS)
        d = d[(d["target"] < lo) | (d["target"] > hi)]
    rows = []
    for h, g in d.groupby("h"):
        wide = g.pivot_table(index="target", columns="model", values="error")
        if against not in wide:
            continue
        for model in wide.columns:
            if model == against:
                continue
            pair = wide[[model, against]].dropna()
            stat, p = diebold_mariano(pair[model].values, pair[against].values, h)
            rows.append({"h": h, "model": model, "vs": against, "n": len(pair),
                         "dm_stat": stat, "p_value": p,
                         "verdict": ("better" if stat < 0 else "worse") + (" (significant at 5%)" if p < 0.05 else " (not significant)")})
    return pd.DataFrame(rows)


def future_forecasts(panel: pd.DataFrame, models: list, h: int = 12) -> pd.DataFrame:
    """Forecast the next h months from the end of the data with every model."""
    rows = []
    for m in models:
        fc = m.forecast(panel, h)
        for date, val in fc.items():
            rows.append({"date": date, "model": m.name, "forecast": float(val)})
    return pd.DataFrame(rows)
