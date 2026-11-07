import numpy as np
import pandas as pd
import pytest

from inflation_fc.data import _parse_dates, read_raw_series, splice_index, yoy


def test_parse_dates_iso_and_dayfirst():
    s = pd.Series(["2024-01", "2024-02-05", "05-03-2024", "31/12/2023"])
    out = _parse_dates(s)
    assert list(out.dt.month) == [1, 2, 3, 12]
    assert out.iloc[1].day == 5  # ISO stays year-month-day


def test_repo_step_series_forward_fills(tmp_path):
    f = tmp_path / "repo.csv"
    f.write_text("date,value\n2020-01-15,5.15\n2020-03-27,4.40\n2020-05-22,4.00\n")
    s = read_raw_series(f, how="last", step=True)
    assert s.loc["2020-02-01"] == 5.15
    assert s.loc["2020-04-01"] == 4.40
    assert s.loc["2020-05-01"] == 4.00


def test_fortnightly_to_month_end(tmp_path):
    f = tmp_path / "credit.csv"
    f.write_text("date,value\n2024-01-12,100\n2024-01-26,\"1,050\"\n2024-02-09,110\n")
    s = read_raw_series(f, how="last")
    assert s.loc["2024-01-01"] == 1050  # last fortnight, comma stripped


def test_yoy():
    idx = pd.date_range("2020-01-01", periods=24, freq="MS")
    s = pd.Series(np.r_[np.full(12, 100.0), np.full(12, 105.0)], idx)
    assert yoy(s).iloc[-1] == pytest.approx(5.0)


def test_splice_preserves_growth():
    idx = pd.date_range("2020-01-01", periods=24, freq="MS")
    old = pd.Series(np.linspace(150, 170, 24), idx)
    new = old.iloc[12:] / 1.6  # rebased series, 12 months overlap
    spliced = splice_index(old.iloc[:18], new)
    assert np.allclose(spliced.values, old.values)
