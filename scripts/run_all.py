"""Run the whole project end to end and save every result.

    python scripts/run_all.py            # real data from data/raw/
    python scripts/run_all.py --quick    # backtest every 3rd month (faster)
    python scripts/run_all.py --demo     # synthetic data, only to test the code

Outputs:
    data/processed/india_macro_monthly.csv   model-ready panel
    results/*.csv, results/*.json            tables used by the Streamlit app
    reports/figures/*.png                    charts for the README / note
    reports/results_summary.md               key numbers in one place
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from inflation_fc import config as C  # noqa: E402
from inflation_fc import evaluate, plots, regression, stationarity, var_model  # noqa: E402
from inflation_fc.data import make_panel  # noqa: E402
from inflation_fc.ml_models import MLForecaster  # noqa: E402
from inflation_fc.sarima import RandomWalk, SarimaForecaster, SeasonalAverage, select_order  # noqa: E402


def step(msg):
    print(f"\n=== {msg}")


def save_csv(df: pd.DataFrame, name: str, index: bool = False):
    C.RESULTS.mkdir(parents=True, exist_ok=True)
    df.to_csv(C.RESULTS / name, index=index, float_format="%.6g")


def save_json(obj, name: str):
    C.RESULTS.mkdir(parents=True, exist_ok=True)
    (C.RESULTS / name).write_text(json.dumps(obj, indent=2, default=str))


def regression_outputs(res: regression.RegressionResult, tag: str):
    save_csv(res.coef.reset_index(names="term"), f"regression_{tag}_coef.csv")
    extra = {k: v for k, v in res.extra.items() if not isinstance(v, pd.Series)}
    save_json({"equation": res.equation, "diagnostics": res.diagnostics, "chow": res.chow,
               "derived": extra}, f"regression_{tag}.json")
    (C.RESULTS / f"regression_{tag}_summary.txt").write_text(res.summary_text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true", help="use synthetic demo data")
    ap.add_argument("--quick", action="store_true", help="backtest every 3rd origin")
    ap.add_argument("--skip-backtest", action="store_true")
    args = ap.parse_args()
    t_start = time.time()

    step("1. Building the monthly panel")
    if args.demo:
        sys.path.insert(0, str(ROOT / "scripts"))
        import make_demo_data
        make_demo_data.main()
        C.DATA_RAW = make_demo_data.DEMO_DIR
    panel = make_panel(source_label="demo" if args.demo else "real")

    step("2. Stationarity tests (ADF, KPSS, Zivot-Andrews)")
    st = stationarity.stationarity_table(panel)
    save_csv(st, "stationarity.csv")
    print(st[["variable", "adf_p", "kpss_p", "verdict"]].to_string(index=False))

    step("3. Regressions: Phillips curve and Taylor rule")
    pc = regression.phillips_curve(panel)
    tr = regression.taylor_rule(panel)
    regression_outputs(pc, "phillips")
    regression_outputs(tr, "taylor")
    save_csv(regression.subsample_comparison(panel, regression.phillips_curve).reset_index(names="term"),
             "regression_phillips_subsamples.csv")
    save_csv(regression.subsample_comparison(panel, regression.taylor_rule).reset_index(names="term"),
             "regression_taylor_subsamples.csv")
    fitted = pd.DataFrame({"actual_inflation": panel["inflation"], "phillips_fitted": pc.extra["fitted"],
                           "actual_repo": panel["repo"], "taylor_fitted": tr.extra["fitted"]})
    save_csv(fitted.reset_index(), "regression_fitted.csv")
    print(f"  Phillips curve: R2={pc.diagnostics['r2']:.3f}, Chow p={pc.chow['p_value']:.3f}")
    print(f"  Taylor rule: long-run inflation response={tr.extra['long_run_inflation_response']:.2f}")

    step("4. SARIMA order selection (on data before the backtest starts)")
    pre = panel.loc[: pd.Timestamp(C.BACKTEST_START) - pd.offsets.MonthBegin(1)]
    grid = select_order(pre["log_cpi"])
    save_csv(grid.astype({"order": str, "seasonal_order": str}), "sarima_order_search.csv")
    best = grid.iloc[0]
    print(f"  chosen (white residuals, lowest BIC): SARIMA{best['order']}x{best['seasonal_order']}  (Ljung-Box p={best['ljung_box_p12']:.2f})")
    sarima = SarimaForecaster(best["order"], best["seasonal_order"])
    full_fit = sarima.fit_full(panel)
    save_csv(pd.DataFrame({"param": full_fit.params.index, "estimate": full_fit.params.values,
                           "std_err": full_fit.bse.values}), "sarima_params.csv")

    step("5. VAR: lag selection, Granger causality, impulse responses, FEVD")
    var_cols = var_model.var_columns(panel)  # decided once, on the full sample
    lag_tab = var_model.lag_selection(pre, cols=var_cols)
    # BIC, not AIC: with only ~4 years before the backtest, AIC picks the
    # maximum lag length and the VAR overfits (forecasts get much worse).
    var_lags = max(1, int(lag_tab.attrs["selected"]["bic"]))
    save_csv(lag_tab.reset_index(), "var_lag_selection.csv")
    res_var = var_model.fit_var(panel, lags=var_lags, cols=var_cols)
    save_csv(var_model.granger_table(res_var).reset_index(), "var_granger.csv")
    irf_repo = var_model.impulse_responses(res_var, "repo", scale_to=1.0)
    irf_oil = var_model.impulse_responses(res_var, "oil_yoy", scale_to=10.0)  # 10% oil price shock
    save_csv(irf_repo, "var_irf_repo.csv")
    save_csv(irf_oil, "var_irf_oil.csv")
    save_csv(var_model.fevd_table(res_var).reset_index(), "var_fevd_inflation.csv")
    stab = var_model.stability(res_var)
    save_json({"lags": var_lags, "variables": res_var.names, "nobs": res_var.nobs, **stab}, "var_info.json")
    print(f"  VAR({var_lags}) on {res_var.names}, stable={stab['is_stable']}")

    models = [RandomWalk(), SeasonalAverage(), sarima, var_model.VarForecaster(var_lags, var_cols),
              MLForecaster("gbm"), MLForecaster("rf")]
    save_json({"sarima_order": list(sarima.order), "sarima_seasonal_order": list(sarima.seasonal_order),
               "var_lags": var_lags, "horizons": C.HORIZONS, "backtest_start": C.BACKTEST_START,
               "models": [m.name for m in models]}, "model_config.json")

    if not args.skip_backtest:
        step("6. Rolling-origin backtest (this is the slow part)")
        bt = evaluate.rolling_backtest(panel, models, step=3 if args.quick else 1)
        save_csv(bt, "backtest_forecasts.csv")
        met = evaluate.metrics_table(bt)
        met_x = evaluate.metrics_table(bt, exclude_covid=True)
        save_csv(met, "metrics.csv")
        save_csv(met_x, "metrics_excl_covid.csv")
        save_csv(evaluate.dm_table(bt, "Random walk"), "dm_vs_random_walk.csv")
        save_csv(evaluate.dm_table(bt, "SARIMA"), "dm_vs_sarima.csv")
        save_csv(evaluate.dm_table(bt, "Random walk", exclude_covid=True), "dm_vs_random_walk_excl_covid.csv")
        print(met.round(3).to_string(index=False))
        plots.rmse_bars(met)
        plots.rmse_bars(met_x, "03b_rmse_excl_covid.png", "Out-of-sample RMSE, excluding COVID months")

    step("7. Forecasts for the next 12 months")
    future = evaluate.future_forecasts(panel, [m for m in models if m.name != "Seasonal average"], h=12)
    save_csv(future, "future_forecasts.csv")
    mean, lo, hi = sarima.forecast(panel, 12, alpha=0.2)
    ci = pd.DataFrame({"forecast": mean, "lower": lo, "upper": hi})
    save_csv(ci.reset_index(names="date"), "sarima_future_ci.csv")
    gbm = models[4]
    imp = pd.DataFrame({f"h{h}": s for h, s in gbm.last_importance.items() if h in (1, 3, 12)})
    if imp.empty:  # HistGradientBoosting has no impurity importances; use the RF
        imp = pd.DataFrame({f"h{h}": s for h, s in models[5].last_importance.items() if h in (1, 3, 12)})
    save_csv(imp.reset_index(names="feature"), "ml_feature_importance.csv")

    step("8. Figures and summary")
    plots.inflation_vs_target(panel)
    plots.forecast_fan(panel, future, ci)
    plots.irf_grid(irf_repo)
    plots.phillips_fit(panel, pc.extra["fitted"])
    from make_summary import write_summary  # noqa: E402
    write_summary()
    print(f"\nDone in {time.time() - t_start:.0f}s. Launch the app with: streamlit run streamlit_app.py")


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT / "scripts"))
    main()
