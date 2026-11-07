"""Load raw series, download what can be downloaded, and build one tidy
monthly panel that every model uses.

Raw files live in data/raw/ as two-column CSVs (date,value). Files you export
by hand from RBI DBIE or MoSPI always take priority; FRED is only used to
fill gaps (oil, exchange rate, and fallbacks for CPI/IIP).
"""
from __future__ import annotations

import io
import json
import re
import urllib.request
import warnings
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from statsmodels.tsa.filters.hp_filter import hpfilter
from statsmodels.tsa.seasonal import STL

from . import config as C

FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}"
_ISO = re.compile(r"^\d{4}-\d{2}(-\d{2})?$")


# --------------------------------------------------------------------------
# Reading raw files
# --------------------------------------------------------------------------
def _parse_dates(raw: pd.Series) -> pd.Series:
    """Parse dates without day/month confusion.

    ISO strings (2024-01 or 2024-01-31) are parsed as ISO. Anything else is
    read day-first, which matches how DBIE and MoSPI write dates (31-01-2024).
    """
    s = raw.astype(str).str.strip()
    iso = s.str.match(_ISO)
    out = pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns]")
    if iso.any():
        out[iso] = pd.to_datetime(s[iso], format="ISO8601")
    if (~iso).any():
        out[~iso] = pd.to_datetime(s[~iso], dayfirst=True, format="mixed")
    return out


def read_raw_series(path, how: str = "last", step: bool = False) -> pd.Series:
    """Read a date,value CSV and return a month-start indexed series.

    how  -- 'last' (stock / index values) or 'mean' (prices averaged over month)
    step -- True for policy rates given as decision dates: carry the last
            value forward into months with no decision.
    """
    df = pd.read_csv(path, comment="#")
    if df.shape[1] < 2:
        raise ValueError(f"{path} needs two columns: date,value")
    dates = _parse_dates(df.iloc[:, 0])
    values = pd.to_numeric(
        df.iloc[:, 1].astype(str).str.replace(",", "").str.strip(), errors="coerce"
    )
    s = pd.Series(values.values, index=dates).dropna().sort_index()
    s = s[~s.index.duplicated(keep="last")]
    monthly = s.resample("MS").last() if how == "last" else s.resample("MS").mean()
    if step:
        monthly = monthly.ffill()
    return monthly


def fetch_fred(series_id: str, timeout: int = 30) -> pd.Series:
    """Download one FRED series as a monthly series (no API key needed)."""
    req = urllib.request.Request(FRED_URL.format(sid=series_id),
                                 headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        text = resp.read().decode("utf-8")
    df = pd.read_csv(io.StringIO(text))
    df.columns = ["date", "value"]
    df["date"] = pd.to_datetime(df["date"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df.dropna().set_index("date")["value"].resample("MS").mean()


def download_missing(force: bool = False, verbose: bool = True) -> dict:
    """Fill data/raw/ from FRED for every series that has a FRED id and no
    hand-made file. Returns {series_name: status}."""
    C.DATA_RAW.mkdir(parents=True, exist_ok=True)
    status = {}
    for name, spec in C.SERIES.items():
        path = C.DATA_RAW / spec["file"]
        if path.exists() and not force:
            status[name] = "local file"
            continue
        if not spec["fred"]:
            status[name] = "missing (manual download needed)" if not path.exists() else "local file"
            continue
        try:
            s = fetch_fred(spec["fred"])
            out = s.rename("value").to_frame()
            out.index.name = "date"
            header = f"# source: FRED {spec['fred']}, downloaded {datetime.now(timezone.utc):%Y-%m-%d}\n"
            path.write_text(header + out.to_csv(date_format="%Y-%m-%d"))
            status[name] = f"downloaded from FRED ({spec['fred']})"
        except Exception as exc:  # network errors, renamed series, etc.
            status[name] = f"FRED download failed: {exc}"
        if verbose:
            print(f"  {name:7s} -> {status[name]}")
    return status


def splice_index(old: pd.Series, new: pd.Series) -> pd.Series:
    """Link an index that changed base year (e.g. CPI 2012=100 -> 2024=100).

    The new-base series is rescaled to the old base using the average ratio
    over the months where both exist, so growth rates are preserved.
    """
    overlap = old.index.intersection(new.index)
    if len(overlap) == 0:
        raise ValueError("Cannot splice: the old and new series need at least one common month.")
    factor = (old[overlap] / new[overlap]).mean()
    return pd.concat([old[old.index < new.index.min()], new * factor]).sort_index()


def load_raw(verbose: bool = True) -> tuple[pd.DataFrame, dict]:
    """Read every available raw series into one monthly DataFrame."""
    frames, notes = {}, {}
    for name, spec in C.SERIES.items():
        path = C.DATA_RAW / spec["file"]
        if not path.exists():
            if spec["required"]:
                raise FileNotFoundError(
                    f"Missing {path.name} ({spec['label']}). Run `python scripts/fetch_data.py` "
                    f"or export it from {spec['source']} - see data/SOURCES.md."
                )
            notes[name] = "not available"
            continue
        s = read_raw_series(path, how=spec["how"], step=(name == "repo"))
        # Index revised to a new base year (CPI 2024=100, IIP 2022-23=100)?
        # Save the new series as <file>_newbase.csv and it is linked here.
        new_path = path.with_name(path.stem + "_newbase.csv")
        if new_path.exists():
            s = splice_index(s, read_raw_series(new_path, how=spec["how"]))
            notes[f"{name}_splice"] = f"spliced {path.name} with {new_path.name}"
        first_line = path.read_text().splitlines()[0]
        notes[name] = first_line.lstrip("# ").strip() if first_line.startswith("#") else "local file"
        frames[name] = s
        if verbose:
            print(f"  {name:7s} {s.index.min():%Y-%m} to {s.index.max():%Y-%m}  ({len(s)} months)")
    return pd.DataFrame(frames), notes


# --------------------------------------------------------------------------
# Transformations
# --------------------------------------------------------------------------
def yoy(s: pd.Series) -> pd.Series:
    """Year-on-year % change of a monthly series."""
    return 100 * (s / s.shift(12) - 1)


def output_gap(iip: pd.Series) -> pd.Series:
    """Output gap proxy from IIP: seasonally adjust log IIP with robust STL,
    then take the deviation from an HP trend (lambda=129600 for monthly)."""
    log_iip = np.log(iip.dropna())
    stl = STL(log_iip, period=12, robust=True).fit()
    sa = log_iip - stl.seasonal
    cycle, _trend = hpfilter(sa, lamb=129600)
    return 100 * cycle


def build_panel(raw: pd.DataFrame) -> pd.DataFrame:
    """Create model-ready variables from raw levels."""
    df = raw.copy()
    df = df[df.index >= pd.Timestamp(C.START)]
    # Repo is a step series: carry it to the last month of the other data.
    df["repo"] = df["repo"].ffill()

    df["log_cpi"] = np.log(df["cpi"])
    df["inflation"] = yoy(df["cpi"])
    df["cpi_mom"] = 100 * df["log_cpi"].diff()
    df["iip_yoy"] = yoy(df["iip"])
    df["oil_yoy"] = yoy(df["oil"])
    df["inr_yoy"] = yoy(df["inr"])
    if "credit" in df:
        df["credit_yoy"] = yoy(df["credit"])
    df["output_gap"] = output_gap(df["iip"])
    df["real_repo"] = df["repo"] - df["inflation"]
    df["post_fit"] = (df.index >= pd.Timestamp(C.FIT_BREAK_DATE)).astype(int)
    df["covid"] = df.index.isin(pd.to_datetime(C.COVID_MONTHS)).astype(int)

    # Keep months where all required series exist.
    required = [k for k, v in C.SERIES.items() if v["required"]]
    last_common = min(df[k].last_valid_index() for k in required)
    df = df[df.index <= last_common]
    df.index.name = "date"
    return df


def make_panel(verbose: bool = True, source_label: str = "real") -> pd.DataFrame:
    """Read raw files, build the panel and save it to data/processed/."""
    raw, notes = load_raw(verbose=verbose)
    panel = build_panel(raw)
    C.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    panel.to_csv(C.PANEL_PATH, date_format="%Y-%m-%d")
    meta = {
        "data": source_label,
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "start": f"{panel['inflation'].first_valid_index():%Y-%m}",
        "end": f"{panel.index.max():%Y-%m}",
        "sources": notes,
    }
    C.PANEL_META_PATH.write_text(json.dumps(meta, indent=2))
    if verbose:
        print(f"Saved panel {meta['start']} to {meta['end']} -> {C.PANEL_PATH.relative_to(C.ROOT)}")
    return panel


def load_panel() -> pd.DataFrame:
    """Load the processed panel (built by make_panel)."""
    if not C.PANEL_PATH.exists():
        raise FileNotFoundError("No processed panel yet. Run `python scripts/run_all.py` first.")
    return pd.read_csv(C.PANEL_PATH, index_col="date", parse_dates=True).asfreq("MS")


def load_meta() -> dict:
    if C.PANEL_META_PATH.exists():
        return json.loads(C.PANEL_META_PATH.read_text())
    return {}


def var_columns(panel: pd.DataFrame) -> list[str]:
    """VAR variables that exist in this panel, in Cholesky order."""
    cols = [c for c in C.VAR_ORDER if c in panel and panel[c].notna().sum() > 60]
    return cols


warnings.filterwarnings("ignore", category=FutureWarning)
