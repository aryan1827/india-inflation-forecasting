"""Machine-learning baselines: random forest and gradient boosting.

Design choices that keep the comparison with SARIMA/VAR honest:
  * Direct strategy: one model per horizon h, predicting inflation at t+h
    from information available at t. No recursive feeding of predictions.
  * The target is the change in inflation (pi_{t+h} - pi_t), because tree
    models cannot extrapolate beyond the range of levels seen in training.
  * A "base effect" feature: the m-o-m price changes that will drop out of
    the y-o-y rate over the next h months. RBI analysts track this explicitly.
  * Training rows only use targets observed by the forecast origin, so there
    is no look-ahead leakage.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor

from . import config as C


def make_features(panel: pd.DataFrame) -> pd.DataFrame:
    p = panel
    X = pd.DataFrame(index=p.index)
    for k in (0, 1, 2, 3, 6, 12):
        X[f"infl_l{k}"] = p["inflation"].shift(k)
    for k in range(12):
        X[f"mom_l{k}"] = p["cpi_mom"].shift(k)
    # The HP-filter output gap is deliberately left out: it is a two-sided
    # filter, so in a backtest it would leak future information.
    for col in ("oil_yoy", "inr_yoy", "iip_yoy", "repo", "real_repo", "credit_yoy"):
        if col in p:
            X[col] = p[col]
    X["repo_chg3"] = p["repo"] - p["repo"].shift(3)
    X["oil_mom"] = 100 * np.log(p["oil"]).diff()
    return X


def add_horizon_features(X: pd.DataFrame, h: int) -> pd.DataFrame:
    Xh = X.copy()
    Xh["base_effect"] = sum(X[f"mom_l{k}"] for k in range(12 - h, 12))
    target_month = (X.index + pd.DateOffset(months=h)).month
    Xh["tm_sin"] = np.sin(2 * np.pi * target_month / 12)
    Xh["tm_cos"] = np.cos(2 * np.pi * target_month / 12)
    return Xh


def _rf():
    return RandomForestRegressor(n_estimators=300, min_samples_leaf=3, max_features=0.5,
                                 random_state=C.RANDOM_STATE, n_jobs=-1)


def _gbm():
    return HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_depth=3,
                                         min_samples_leaf=8, l2_regularization=1.0,
                                         random_state=C.RANDOM_STATE)


class MLForecaster:
    def __init__(self, kind: str = "gbm"):
        self.kind = kind
        self.name = {"gbm": "Gradient boosting", "rf": "Random forest"}[kind]
        self.last_importance: dict[int, pd.Series] = {}

    def _model(self):
        return _gbm() if self.kind == "gbm" else _rf()

    def forecast(self, panel: pd.DataFrame, h: int, horizons=None) -> pd.Series:
        """Forecast y-o-y inflation at the requested horizons (default 1..h)."""
        horizons = list(horizons) if horizons is not None else list(range(1, h + 1))
        infl = panel["inflation"]
        origin = infl.dropna().index[-1]
        X_all = make_features(panel.loc[:origin])
        out = {}
        for hh in horizons:
            Xh = add_horizon_features(X_all, hh)
            x_now = Xh.loc[[origin]]
            # Drop features missing at the origin (e.g. credit ending early).
            keep = x_now.columns[x_now.notna().all()]
            y = infl.shift(-hh) - infl
            train = pd.concat([Xh[keep], y.rename("y")], axis=1).loc[:origin].dropna()
            model = self._model().fit(train[keep], train["y"])
            if hasattr(model, "feature_importances_"):
                self.last_importance[hh] = pd.Series(model.feature_importances_, index=keep)
            out[origin + pd.DateOffset(months=hh)] = infl.loc[origin] + model.predict(x_now[keep])[0]
        return pd.Series(out)
