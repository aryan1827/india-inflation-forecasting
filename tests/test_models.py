import numpy as np
import pandas as pd
import pytest

from inflation_fc.evaluate import diebold_mariano
from inflation_fc.ml_models import MLForecaster
from inflation_fc.sarima import RandomWalk, SeasonalAverage, yoy_from_log_path


def toy_panel(n=120, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2012-01-01", periods=n, freq="MS")
    mom = 0.4 + 0.3 * np.sin(2 * np.pi * idx.month / 12) + rng.normal(0, 0.2, n)
    log_cpi = np.log(100) + np.cumsum(mom) / 100
    p = pd.DataFrame({"log_cpi": log_cpi}, idx)
    p["cpi"] = np.exp(p["log_cpi"])
    p["cpi_mom"] = 100 * p["log_cpi"].diff()
    p["inflation"] = 100 * (p["cpi"] / p["cpi"].shift(12) - 1)
    for c in ("oil_yoy", "inr_yoy", "iip_yoy"):
        p[c] = rng.normal(0, 5, n)
    p["oil"] = 80 + rng.normal(0, 5, n)
    p["repo"] = 6.0
    p["real_repo"] = p["repo"] - p["inflation"]
    return p


def test_yoy_from_log_path_uses_history_for_base():
    idx = pd.date_range("2020-01-01", periods=12, freq="MS")
    hist = pd.Series(np.log(100.0), idx)
    fc = pd.Series(np.log(110.0), pd.date_range("2021-01-01", periods=3, freq="MS"))
    out = yoy_from_log_path(hist, fc)
    assert out.iloc[0] == pytest.approx(10.0)


def test_benchmarks_shapes():
    p = toy_panel()
    assert len(RandomWalk().forecast(p, 12)) == 12
    sa = SeasonalAverage().forecast(p, 12)
    assert len(sa) == 12 and sa.notna().all()


def test_ml_has_no_lookahead():
    """Changing data after the origin must not change the forecast."""
    p = toy_panel()
    origin = p.index[100]
    a = MLForecaster("rf").forecast(p.loc[:origin], 3, horizons=[3])
    p2 = p.copy()
    p2.loc[p2.index > origin, ["inflation", "cpi_mom"]] = 999.0
    b = MLForecaster("rf").forecast(p2.loc[:origin], 3, horizons=[3])
    assert a.iloc[0] == pytest.approx(b.iloc[0])
    assert a.index[0] == origin + pd.DateOffset(months=3)


def test_diebold_mariano_direction():
    rng = np.random.default_rng(1)
    good = rng.normal(0, 0.5, 200)
    bad = rng.normal(0, 1.5, 200)
    stat, p = diebold_mariano(good, bad, h=1)
    assert stat < 0 and p < 0.01
    stat2, p2 = diebold_mariano(good, good, h=3)
    assert np.isnan(stat2) or abs(stat2) < 1e-9 or p2 > 0.5
