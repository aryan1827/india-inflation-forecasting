"""Vector autoregression: lag selection, Granger causality, impulse responses,
variance decomposition, and multi-step forecasts.

Variables enter in y-o-y growth form (plus the repo rate in levels), the way
many RBI working papers set up small monetary-transmission VARs. Structural
shocks are identified recursively (Cholesky), so the column order in
config.VAR_ORDER is an assumption: oil is the most exogenous, financial
variables react within the month to everything ordered before them.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.api import VAR

from . import config as C
from .data import output_gap, var_columns


def _prepare(panel: pd.DataFrame, cols: list[str] | None = None):
    """Endogenous variables plus one impulse dummy per COVID month that falls
    inside the sample (dummies that are all zero are dropped).

    Pass `cols` (decided once on the full sample) whenever the panel is a
    shorter sub-sample, so every window uses the same variables."""
    cols = list(cols) if cols else var_columns(panel)
    endog = panel[cols].dropna()
    exog = pd.DataFrame(index=endog.index)
    for d in pd.to_datetime(C.COVID_MONTHS):
        if d in endog.index:
            exog[f"covid_{d:%Y_%m}"] = (endog.index == d).astype(float)
    if exog.shape[1] == 0:
        exog = None
    return endog, exog, cols


def lag_selection(panel: pd.DataFrame, maxlags: int = C.VAR_MAXLAGS, cols=None) -> pd.DataFrame:
    endog, exog, cols = _prepare(panel, cols)
    # Keep enough observations per parameter in short samples.
    maxlags = max(1, min(maxlags, (len(endog) - 10) // (len(cols) + 1)))
    sel = VAR(endog, exog=exog).select_order(maxlags=maxlags)
    table = pd.DataFrame(sel.ics)
    table.index.name = "lags"
    table.attrs["selected"] = sel.selected_orders
    return table


def fit_var(panel: pd.DataFrame, lags: int | None = None, criterion: str = "aic", cols=None):
    endog, exog, cols = _prepare(panel, cols)
    model = VAR(endog, exog=exog)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if lags is None:
            lags = max(1, model.select_order(maxlags=C.VAR_MAXLAGS).selected_orders[criterion])
        return model.fit(lags)


def granger_table(res) -> pd.DataFrame:
    """p-values of 'row Granger-causes column'. Small values mean the row
    variable's past helps predict the column variable (predictive content,
    not structural causality)."""
    names = res.names
    out = pd.DataFrame(np.nan, index=names, columns=names)
    for caused in names:
        for causing in names:
            if caused != causing:
                out.loc[causing, caused] = res.test_causality(caused, [causing], kind="f").pvalue
    out.index.name = "causing \\ caused"
    return out


def impulse_responses(res, shock: str = "repo", horizon: int = C.IRF_HORIZON,
                      repl: int = 500, scale_to: float | None = 1.0) -> pd.DataFrame:
    """Orthogonalised impulse responses to `shock` with Monte Carlo 90% bands.

    With scale_to=1.0 and shock='repo', responses are rescaled to a shock that
    raises the repo rate by 1 percentage point (100 bp) on impact.
    """
    irf = res.irf(horizon)
    j = res.names.index(shock)
    point = irf.orth_irfs[:, :, j]
    lower, upper = irf.errband_mc(orth=True, repl=repl, signif=0.10, seed=C.RANDOM_STATE)
    lo, hi = lower[:, :, j], upper[:, :, j]
    scale = 1.0
    if scale_to is not None and point[0, j] != 0:
        scale = scale_to / point[0, j]
    rows = []
    for i, name in enumerate(res.names):
        for hh in range(horizon + 1):
            rows.append({"shock": shock, "response": name, "h": hh,
                         "irf": point[hh, i] * scale,
                         "lower": lo[hh, i] * scale, "upper": hi[hh, i] * scale})
    return pd.DataFrame(rows)


def fevd_table(res, variable: str = "inflation", horizons=(1, 3, 6, 12, 24)) -> pd.DataFrame:
    """Share of the forecast-error variance of `variable` explained by each
    shock, at selected horizons (rows sum to 100)."""
    fevd = res.fevd(max(horizons) + 1)
    i = res.names.index(variable)
    shares = fevd.decomp[i]  # (horizon, shock)
    df = pd.DataFrame(100 * shares, columns=res.names)
    df.index = np.arange(1, len(df) + 1)
    df.index.name = "h"
    return df.loc[list(horizons)]


def stability(res) -> dict:
    roots = np.abs(res.roots)
    return {"is_stable": bool(res.is_stable()), "min_root_modulus": float(roots.min()),
            "whiteness_p": float(res.test_whiteness(nlags=max(res.k_ar + 1, 12)).pvalue)}


class VarForecaster:
    name = "VAR"

    def __init__(self, lags: int, cols: list[str] | None = None):
        self.lags = lags
        self.cols = cols

    def forecast(self, panel: pd.DataFrame, h: int) -> pd.Series:
        if "output_gap" in C.VAR_ORDER:
            # Real-time gap: HP-filter only the IIP data available at the origin.
            panel = panel.copy()
            panel["output_gap"] = output_gap(panel["iip"])
        endog, exog, cols = _prepare(panel, self.cols)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = VAR(endog, exog=exog).fit(self.lags)
        # If an optional series (credit) ends earlier than CPI, the VAR sample
        # ends earlier too: forecast the extra months so the output lines up
        # with the same target dates as the other models.
        origin = panel["inflation"].dropna().index[-1]
        gap = (origin.year - endog.index[-1].year) * 12 + origin.month - endog.index[-1].month
        steps = h + gap
        exog_future = None if exog is None else np.zeros((steps, exog.shape[1]))
        fc = res.forecast(endog.values[-self.lags:], steps=steps, exog_future=exog_future)
        idx = pd.date_range(endog.index[-1] + pd.offsets.MonthBegin(1), periods=steps, freq="MS")
        return pd.Series(fc[:, cols.index("inflation")], index=idx).iloc[gap:]
