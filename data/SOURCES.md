# Data sources and how to get each file

Every raw series is a CSV in `data/raw/` with two columns:

```
date,value
2012-01-01,100.0
2012-02-01,100.6
```

Dates can be `2012-01`, `2012-01-01` or `01-01-2012` (day first). Commas in
numbers are fine. Lines starting with `#` are comments. Files you put here
yourself are never overwritten by `fetch_data.py`.

## Step 1: automatic downloads

```
python scripts/fetch_data.py
```

This pulls from FRED (no API key needed):

| File | FRED id | What it is |
|---|---|---|
| `brent_usd.csv` | POILBREUSDM | Brent crude, USD/barrel, monthly (IMF) |
| `usd_inr.csv` | EXINUS | INR per USD, monthly average (Federal Reserve H.10) |
| `iip_general_index.csv` | INDPROINDMISMEI | India industrial production (OECD), **fallback only** |
| `cpi_combined_index.csv` | INDCPIALLMINMEI | India CPI (OECD), **fallback only**, historically based on CPI-IW |
| `wacr.csv` | IRSTCI01INM156N | Call money rate (OECD), optional |

If a FRED id has been discontinued, the script says so; get that series from
the Indian source below instead.

## Step 2: replace the fallbacks with official Indian data (do this)

The CPI, IIP and credit series should come from the official Indian
sources. Website menus change, so search for the table names
given here.

### CPI Combined, all-India general index → `cpi_combined_index.csv` (required)

* **Where:** MoSPI CPI data portal (`cpi.mospi.gov.in`, or the eSankhyiki
  portal `esankhyiki.mospi.gov.in`), or RBI DBIE (`data.rbi.org.in`) →
  Prices → "Consumer Price Index (Base: 2012=100)".
* **What to pick:** All India, Combined (Rural + Urban), General Index, monthly,
  January 2011 onward. Use the **index**, not the inflation rate.
* **Base-year change:** MoSPI is moving CPI to a new base year (2024=100). If
  recent months are only available on the new base, save them as
  `cpi_combined_index_newbase.csv`, with at least one month that overlaps the old
  series. The pipeline links the two using the overlap, which keeps growth rates
  unchanged. If there is no overlap, multiply the new series by MoSPI's published
  linking factor yourself and append it to the old file.

### Policy repo rate → `repo_rate.csv` (required, already included)

* This file is **already filled** with every RBI repo rate decision from 2010
  to December 2025 (one row per change, effective date).
* **Do this:** check it against DBIE → Interest Rates → "Policy Rates", or RBI
  monetary policy press releases, and add a row for any decision after
  December 2025.

### IIP general index → `iip_general_index.csv` (required)

* **Where:** MoSPI → Index of Industrial Production (or eSankhyiki), or DBIE →
  Real Sector → "Index of Industrial Production".
* **What to pick:** General index, base 2011-12=100, monthly, April 2012
  onward (older months come from the 2004-05 series; link them with the same
  `_newbase.csv` method, or start from 2012). If MoSPI has moved IIP to a newer
  base year, use the same `iip_general_index_newbase.csv` approach as CPI.

### Non-food bank credit → `scb_nonfood_credit.csv` (optional, strongly recommended)

* **Where:** DBIE → Money and Banking → "Scheduled Commercial Banks - Business
  in India" (also in the RBI Weekly Statistical Supplement, table 4).
* **What to pick:** Non-food credit, Rs crore, fortnightly. Keep all
  fortnights; the pipeline takes the last one in each month. Note the jump in
  July 2023 when HDFC merged into HDFC Bank: RBI publishes growth rates
  "excluding the impact of the merger". If you use the raw level, mention this
  break in your write-up.
* Without this file the project still runs; the VAR just drops credit.

### INR/USD → `usd_inr.csv` (optional upgrade)

FRED's EXINUS is fine. To use RBI's own data instead: DBIE → Financial Markets
→ "Reference Rate of INR" (USD), and average it by month.

### Weighted average call rate → `wacr.csv` (optional)

DBIE → Financial Markets → "Money Market" → weighted average call rate. Not
used by the default models, but useful if you want to swap it in for the repo
rate in the VAR.

## Step 3: build

```
python scripts/run_all.py
```

`data/processed/panel_meta.json` records where every series came from, and the
app shows it under Data → "Data table and sources".
