"""Box-Jenkins SARIMA on log CPI, plus the two naive benchmarks.

SARIMA is fitted to the log CPI index (not to y-o-y inflation) because the
index carries the monthly seasonal pattern (vegetable prices etc.). The
forecast index path is then turned into y-o-y inflation, so all models are
compared on the same target.
"""
from __future__ import annotations

import itertools
import warnings

import numpy as np
import pandas as pd
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.statespace.sarimax import SARIMAX

from . import config as C


def yoy_from_log_path(hist_log: pd.Series, fc_log: pd.Series) -> pd.Series:
    """Y-o-y inflation (%) for forecast dates, using history for the base
    month when it is known and the forecast path otherwise."""
    full = pd.concat([hist_log, fc_log])
    full = full[~full.index.duplicated(keep="first")]
    infl = 100 * (np.exp(full - full.shift(12)) - 1)
    return infl.loc[fc_log.index]


def _future_index(last: pd.Timestamp, h: int) -> pd.DatetimeIndex:
    return pd.date_range(last + pd.offsets.MonthBegin(1), periods=h, freq="MS")


# Models are estimated on 100 * log CPI. With raw logs the innovation
# variance is ~1e-5, which makes the optimiser stop early (we found the
# seasonal MA off by 0.1 versus R before this fix). Scaling changes nothing
# else: forecasts are divided by SCALE again.
SCALE = 100.0


def _fit(y: pd.Series, order, seasonal_order, exog=None):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # Stationarity/invertibility are enforced so that every candidate model
        # uses the same number of observations in its likelihood; otherwise
        # AIC values are not comparable across seasonal orders.
        model = SARIMAX(y, order=order, seasonal_order=seasonal_order, exog=exog,
                        enforce_stationarity=True, enforce_invertibility=True)
        return model.fit(disp=False, maxiter=200)


def select_order(log_cpi: pd.Series, d: int = 1, D: int = 1, max_p: int = 2, max_q: int = 2,
                 max_P: int = 1, max_Q: int = 1) -> pd.DataFrame:
    """Grid search over (p,q)(P,Q) with fixed differencing.

    Differencing orders are fixed beforehand from the stationarity tests,
    because AIC values are not comparable across different d and D.
    """
    y = SCALE * log_cpi.dropna().asfreq("MS")
    rows = []
    for p, q, P, Q in itertools.product(range(max_p + 1), range(max_q + 1),
                                        range(max_P + 1), range(max_Q + 1)):
        try:
            res = _fit(y, (p, d, q), (P, D, Q, C.SARIMA_SEASONAL_PERIOD))
            lb_p = acorr_ljungbox(res.resid.iloc[13:], lags=[12], model_df=p + q + P + Q, return_df=True)["lb_pvalue"].iloc[0]
            # An AR root at ~1 means the series needs another difference, and
            # such models give unstable, trend-extrapolating forecasts.
            ar_roots = np.abs(res.arroots) if (p or P) else np.array([np.inf])  # includes seasonal AR
            rows.append({"order": (p, d, q), "seasonal_order": (P, D, Q, C.SARIMA_SEASONAL_PERIOD),
                         "aic": res.aic, "bic": res.bic, "ljung_box_p12": lb_p,
                         "near_unit_ar_root": bool(ar_roots.min() < 1.05)})
        except Exception:
            continue
    # Box-Jenkins rule: keep models whose residuals pass the Ljung-Box test
    # (no autocorrelation left at lag 12) and that have no AR root near 1,
    # then take the most parsimonious
    # one by BIC. BIC is preferred to AIC here because over-parameterised
    # ARMA terms tend to cancel out and forecast worse.
    out = pd.DataFrame(rows)
    out["white_residuals"] = out["ljung_box_p12"] > 0.05
    return out.sort_values(["white_residuals", "near_unit_ar_root", "bic"],
                           ascending=[False, True, True]).reset_index(drop=True)


class SarimaForecaster:
    name = "SARIMA"

    def __init__(self, order=(1, 1, 1), seasonal_order=(0, 1, 1, 12)):
        self.order = tuple(order)
        self.seasonal_order = tuple(seasonal_order)

    def fit_full(self, panel: pd.DataFrame):
        """Fitted model on 100 * log CPI (so sigma2 is in squared % units)."""
        return _fit(SCALE * panel["log_cpi"].dropna().asfreq("MS"), self.order, self.seasonal_order)

    def forecast(self, panel: pd.DataFrame, h: int, alpha: float | None = None):
        """Return y-o-y inflation forecasts for the next h months. With alpha,
        also return (lower, upper) prediction bounds."""
        y = panel["log_cpi"].dropna().asfreq("MS")
        res = _fit(SCALE * y, self.order, self.seasonal_order)
        fc = res.get_forecast(h)
        mean = pd.Series(np.asarray(fc.predicted_mean) / SCALE, index=_future_index(y.index[-1], h))
        infl = yoy_from_log_path(y, mean)
        if alpha is None:
            return infl
        ci = fc.conf_int(alpha=alpha)
        lo = pd.Series(np.asarray(ci.iloc[:, 0]) / SCALE, index=mean.index)
        hi = pd.Series(np.asarray(ci.iloc[:, 1]) / SCALE, index=mean.index)
        # For h <= 12 the base month is observed, so the interval maps exactly.
        return infl, yoy_from_log_path(y, lo), yoy_from_log_path(y, hi)


class RandomWalk:
    """No-change forecast: inflation stays where it is today."""
    name = "Random walk"

    def forecast(self, panel: pd.DataFrame, h: int) -> pd.Series:
        last = panel["inflation"].dropna()
        return pd.Series(last.iloc[-1], index=_future_index(last.index[-1], h))


class SeasonalAverage:
    """Each future month gets the average m-o-m price change of that calendar
    month over the last `years` years; the implied index path gives y-o-y
    inflation.

    (A pure seasonal-naive forecast, repeating last year's m-o-m changes, is
    mathematically identical to the random walk for y-o-y inflation at
    horizons up to 12 months, so it would add nothing.)
    """
    name = "Seasonal average"

    def __init__(self, years: int = 5):
        self.years = years

    def forecast(self, panel: pd.DataFrame, h: int) -> pd.Series:
        y = panel["log_cpi"].dropna()
        mom = y.diff().dropna().iloc[-12 * self.years:]
        by_month = mom.groupby(mom.index.month).mean()
        idx = _future_index(y.index[-1], h)
        path = y.iloc[-1] + by_month.reindex(idx.month).cumsum().to_numpy()
        return yoy_from_log_path(y, pd.Series(path, index=idx))
