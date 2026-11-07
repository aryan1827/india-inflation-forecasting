"""Write reports/results_summary.md from the files in results/.

Run automatically at the end of run_all.py, or on its own:
    python scripts/make_summary.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from inflation_fc import config as C  # noqa: E402
from inflation_fc.data import load_meta  # noqa: E402
from inflation_fc.style import LABELS  # noqa: E402


def _read(name):
    p = C.RESULTS / name
    if p.suffix == ".json":
        return json.loads(p.read_text()) if p.exists() else None
    return pd.read_csv(p) if p.exists() else None


def irf_peak(irf: pd.DataFrame, response: str, max_h: int = 24):
    g = irf[(irf["response"] == response) & (irf["h"] >= 1) & (irf["h"] <= max_h)]
    row = g.loc[g["irf"].abs().idxmax()]
    significant = (row["lower"] > 0) or (row["upper"] < 0)
    return float(row["irf"]), int(row["h"]), bool(significant)


def write_summary() -> Path:
    meta = load_meta()
    cfg = _read("model_config.json") or {}
    st = _read("stationarity.csv")
    pc = _read("regression_phillips.json")
    pcc = _read("regression_phillips_coef.csv").set_index("term")
    tr = _read("regression_taylor.json")
    var = _read("var_info.json")
    irf = _read("var_irf_repo.csv")
    met = _read("metrics.csv")
    met_x = _read("metrics_excl_covid.csv")
    dm = _read("dm_vs_random_walk.csv")
    ci = _read("sarima_future_ci.csv")

    L = []
    if meta.get("data") == "demo":
        L += ["> **WARNING: these results come from SYNTHETIC demo data. Do not quote them.**", ""]
    L += [f"# Results summary", "",
          f"Sample: {meta.get('start')} to {meta.get('end')} (monthly). Built {meta.get('built_at', '')[:10]}.", ""]

    L += ["## Stationarity", ""]
    for _, r in st.iterrows():
        L.append(f"- `{r['variable']}`: {r['verdict']} (ADF p={r['adf_p']:.3f}, KPSS p={r['kpss_p']:.3f}"
                 + (f", Zivot-Andrews break {r['za_break']}" if isinstance(r["za_break"], str) and r["za_break"] else "") + ")")

    d = pc["derived"]
    L += ["", "## Phillips curve (OLS, Newey-West errors)", "",
          f"- R² = {pc['diagnostics']['r2']:.3f}, n = {pc['diagnostics']['n_obs']}",
          f"- Inflation persistence (lag coefficient) = {pcc.loc['inflation_lag1', 'coef']:.3f}",
          f"- Output gap effect: {pcc.loc['output_gap_lag1', 'coef']:.3f} (p = {pcc.loc['output_gap_lag1', 'p_value']:.3f}); long run {d['long_run_gap_effect']:.3f}",
          f"- Oil pass-through: {pcc.loc['oil_yoy_lag1', 'coef']:.4f} (p = {pcc.loc['oil_yoy_lag1', 'p_value']:.3f}); long run {d['long_run_oil_passthrough']:.3f} pp per 1% oil",
          f"- Exchange-rate pass-through: long run {d['long_run_inr_passthrough']:.3f} pp per 1% depreciation",
          f"- Chow test at {pc['chow']['break_date'][:7]}: F = {pc['chow']['F']:.2f}, p = {pc['chow']['p_value']:.3f}",
          f"- Breusch-Godfrey p = {pc['diagnostics']['breusch_godfrey_p (12 lags)']:.3f}, Breusch-Pagan p = {pc['diagnostics']['breusch_pagan_p']:.3f}, max VIF = {pc['diagnostics']['max_vif']:.1f}"]

    t = tr["derived"]
    L += ["", "## Taylor rule", "",
          f"- Smoothing ρ = {t['smoothing_rho']:.3f}",
          f"- Long-run response to inflation gap = {t['long_run_inflation_response']:.2f} "
          f"({'satisfies' if t['long_run_inflation_response'] > 1 else 'does not satisfy'} the Taylor principle, >1)",
          f"- Long-run response to output gap = {t['long_run_gap_response']:.2f}",
          f"- Chow test at {tr['chow']['break_date'][:7]}: p = {tr['chow']['p_value']:.3f}"]

    L += ["", "## Models", "",
          f"- SARIMA{tuple(cfg.get('sarima_order', []))}x{tuple(cfg.get('sarima_seasonal_order', []))} on log CPI, chosen by BIC among models with white-noise residuals, on pre-{C.BACKTEST_START[:4]} data",
          f"- VAR({var['lags']}) on {', '.join(LABELS.get(v, v) for v in var['variables'])}; stable = {var['is_stable']}"]

    if irf is not None:
        L += ["", "## Monetary transmission (VAR, 100 bp repo shock)", ""]
        for resp in [v for v in var["variables"] if v != "repo"]:
            val, h, sig = irf_peak(irf, resp)
            L.append(f"- {LABELS.get(resp, resp)}: peak response {val:+.2f} pp after {h} months"
                     f" ({'outside' if sig else 'inside'} the 90% band around zero)")

    best_lines = []
    if met is not None:
        L += ["", "## Forecast accuracy (rolling origin from " + C.BACKTEST_START[:7] + ")", "",
              "| h | model | RMSE | MAE | RMSE vs random walk |", "|---|---|---|---|---|"]
        for _, r in met.iterrows():
            L.append(f"| {r['h']} | {r['model']} | {r['rmse']:.2f} | {r['mae']:.2f} | {r['rel_rmse_vs_rw']:.2f} |")
        L += ["", "Best model per horizon (all months / excluding COVID):", ""]
        for h in sorted(met["h"].unique()):
            b = met[met["h"] == h].iloc[0]
            bx = met_x[met_x["h"] == h].iloc[0]
            dmr = dm[(dm["h"] == h) & (dm["model"] == b["model"])]
            p = f", DM p = {dmr['p_value'].iloc[0]:.3f}" if len(dmr) else ""
            L.append(f"- h={h}: {b['model']} (RMSE {b['rmse']:.2f}, {100 * (1 - b['rel_rmse_vs_rw']):.0f}% below random walk{p}) / {bx['model']}")
            best_lines.append((h, b))

    if ci is not None:
        last = ci.iloc[-1]
        L += ["", "## Outlook", "",
              f"- SARIMA forecast for {str(last['date'])[:7]}: {last['forecast']:.1f}% (80% interval {last['lower']:.1f} to {last['upper']:.1f})"]

    out = ROOT / "reports" / "results_summary.md"
    out.write_text("\n".join(L) + "\n")
    print(f"  wrote {out.relative_to(ROOT)}")
    return out


if __name__ == "__main__":
    write_summary()
