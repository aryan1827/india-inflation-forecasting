"""India CPI inflation: forecasting and monetary transmission dashboard.

    streamlit run streamlit_app.py

Reads the panel in data/processed/ and the tables in results/ that
scripts/run_all.py produces. Nothing heavy is re-estimated here, so the app
loads fast on Streamlit Community Cloud.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from inflation_fc import config as C  # noqa: E402
from inflation_fc.style import (BAND, GRID, LABELS, MODEL_COLORS, NEUTRAL, TEXT,  # noqa: E402
                                TEXT_2, VAR_COLORS)

st.set_page_config(page_title="India Inflation Forecasting", page_icon="📈", layout="wide")


# ---------------------------------------------------------------- loading
@st.cache_data
def load_csv(name: str, **kw) -> pd.DataFrame | None:
    p = C.RESULTS / name
    return pd.read_csv(p, **kw) if p.exists() else None


@st.cache_data
def load_json(name: str) -> dict | None:
    p = C.RESULTS / name
    return json.loads(p.read_text()) if p.exists() else None


@st.cache_data
def load_panel() -> pd.DataFrame | None:
    if not C.PANEL_PATH.exists():
        return None
    return pd.read_csv(C.PANEL_PATH, index_col="date", parse_dates=True)


def meta() -> dict:
    return json.loads(C.PANEL_META_PATH.read_text()) if C.PANEL_META_PATH.exists() else {}


def base_layout(fig: go.Figure, title: str = "", y_title: str = "", height: int = 420) -> go.Figure:
    fig.update_layout(
        title=dict(text=title, x=0, xanchor="left", font=dict(size=16, color=TEXT)),
        height=height, margin=dict(l=10, r=10, t=50, b=10), hovermode="x unified",
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", y=-0.15, x=0), font=dict(color=TEXT_2),
    )
    fig.update_xaxes(showgrid=False, linecolor=GRID)
    fig.update_yaxes(title=y_title, gridcolor=GRID, zerolinecolor=GRID)
    return fig


def add_band(fig: go.Figure):
    lo, hi = C.TOLERANCE_BAND
    fig.add_hrect(y0=lo, y1=hi, fillcolor=BAND, opacity=0.6, line_width=0, layer="below")
    fig.add_hline(y=C.INFLATION_TARGET, line=dict(color=NEUTRAL, dash="dash", width=1))


def fmt_p(p: float) -> str:
    return "<0.001" if p < 0.001 else f"{p:.3f}"


panel = load_panel()
m = meta()

# ---------------------------------------------------------------- header
st.title("Forecasting India's CPI inflation")
st.caption("SARIMA, VAR and machine-learning forecasts, and how a repo rate change passes through "
           "to inflation, credit and output. Data: RBI DBIE, MoSPI, FRED.")

if panel is None:
    st.error("No data yet. Run `python scripts/fetch_data.py` and then `python scripts/run_all.py`, "
             "then reload this page.")
    st.stop()

if m.get("data") == "demo":
    st.warning("**Demo mode:** these charts use synthetic data generated to test the code. "
               "They are not real Indian statistics.", icon="⚠️")

last = panel["inflation"].dropna()
k1, k2, k3, k4 = st.columns(4)
k1.metric(f"CPI inflation, {last.index[-1]:%b %Y}", f"{last.iloc[-1]:.2f}%",
          f"{last.iloc[-1] - last.iloc[-2]:+.2f} pp vs previous month", delta_color="inverse")
k2.metric("Policy repo rate", f"{panel['repo'].dropna().iloc[-1]:.2f}%")
k3.metric("Real repo rate", f"{panel['real_repo'].dropna().iloc[-1]:.2f}%")
ci = load_csv("sarima_future_ci.csv", parse_dates=["date"])
if ci is not None:
    k4.metric(f"SARIMA forecast, {ci['date'].iloc[-1]:%b %Y}", f"{ci['forecast'].iloc[-1]:.2f}%",
              f"80% range {ci['lower'].iloc[-1]:.1f} to {ci['upper'].iloc[-1]:.1f}", delta_color="off")

tabs = st.tabs(["Data", "Stationarity", "Regression", "Forecasts", "Model comparison",
                "Monetary transmission (VAR)", "About"])

# ---------------------------------------------------------------- data
with tabs[0]:
    fig = go.Figure()
    add_band(fig)
    fig.add_trace(go.Scatter(x=panel.index, y=panel["inflation"], name="CPI inflation (y-o-y)",
                             line=dict(color=VAR_COLORS["inflation"], width=2)))
    fig.add_trace(go.Scatter(x=panel.index, y=panel["repo"], name="Repo rate",
                             line=dict(color=VAR_COLORS["repo"], width=2, shape="hv")))
    fig.add_vline(x=pd.Timestamp(C.FIT_BREAK_DATE), line=dict(color=TEXT_2, width=1))
    fig.add_annotation(x=pd.Timestamp(C.FIT_BREAK_DATE), y=1, yref="paper", text="Inflation targeting",
                       showarrow=False, xanchor="left", font=dict(size=11, color=TEXT_2))
    st.plotly_chart(base_layout(fig, "CPI inflation, the 2–6% band and the repo rate", "%"), width="stretch")

    options = {LABELS.get(c, c): c for c in ["iip_yoy", "credit_yoy", "oil_yoy", "inr_yoy", "output_gap", "real_repo"]
               if c in panel}
    pick = st.selectbox("Another series", list(options))
    col = options[pick]
    fig2 = go.Figure(go.Scatter(x=panel.index, y=panel[col], name=pick,
                                line=dict(color=VAR_COLORS.get(col, MODEL_COLORS["SARIMA"]), width=2)))
    fig2.add_hline(y=0, line=dict(color=GRID))
    st.plotly_chart(base_layout(fig2, pick, "%", 340), width="stretch")

    with st.expander("Data table and sources"):
        st.dataframe(panel.drop(columns=["post_fit", "covid"]).round(3).iloc[::-1], width="stretch")
        st.json(m.get("sources", {}))
        st.download_button("Download panel (CSV)", C.PANEL_PATH.read_bytes(), "india_macro_monthly.csv")

# ---------------------------------------------------------------- stationarity
with tabs[1]:
    st.markdown("**Why this matters:** regressions on non-stationary series can show strong relationships "
                "that are not real (spurious regression), and ARIMA needs the right amount of differencing. "
                "ADF's null is a unit root; KPSS's null is stationarity, so the two are read together.")
    stt = load_csv("stationarity.csv")
    if stt is not None:
        show = stt[["variable", "description", "adf_stat", "adf_p", "kpss_stat", "kpss_p", "za_p", "za_break", "verdict"]]
        st.dataframe(show.round(4), width="stretch", hide_index=True)
        st.caption("Zivot-Andrews allows one structural break chosen by the data; za_break is the estimated break month.")

# ---------------------------------------------------------------- regression
with tabs[2]:
    which = st.radio("Equation", ["Phillips curve", "Taylor rule"], horizontal=True)
    tag = "phillips" if which == "Phillips curve" else "taylor"
    info = load_json(f"regression_{tag}.json")
    coef = load_csv(f"regression_{tag}_coef.csv")
    if info and coef is not None:
        st.latex(info["equation"].replace("π", r"\pi ").replace("α", r"\alpha ").replace("β", r"\beta ")
                 .replace("γ", r"\gamma ").replace("δ", r"\delta ").replace("ε", r"\varepsilon ")
                 .replace("ρ", r"\rho ").replace("·", r"\cdot ").replace("−", "-")
                 .replace("₁", "_1").replace("₂", "_2").replace("₃", "_3").replace("₄", "_4"))
        c1, c2 = st.columns([3, 2])
        with c1:
            st.markdown("**Coefficients (Newey-West HAC standard errors, 12 lags)**")
            st.dataframe(coef.round(4), width="stretch", hide_index=True)
            sub = load_csv(f"regression_{tag}_subsamples.csv")
            if sub is not None:
                st.markdown("**Before vs after inflation targeting (Sept 2016)**")
                st.dataframe(sub.round(3), width="stretch", hide_index=True)
        with c2:
            d = info["diagnostics"]
            st.markdown("**Diagnostics**")
            diag = pd.DataFrame({
                "test": ["Observations", "R²", "Adjusted R²", "Durbin-Watson", "Breusch-Godfrey (12) p",
                         "Breusch-Pagan p", "Jarque-Bera p", "Max VIF", f"Chow test at {info['chow']['break_date'][:7]} p"],
                "value": [d["n_obs"], round(d["r2"], 3), round(d["adj_r2"], 3), round(d["durbin_watson"], 2),
                          fmt_p(d["breusch_godfrey_p (12 lags)"]), fmt_p(d["breusch_pagan_p"]),
                          fmt_p(d["jarque_bera_p"]), round(d["max_vif"], 1), fmt_p(info["chow"]["p_value"])],
            })
            st.dataframe(diag.astype(str), width="stretch", hide_index=True)
            st.markdown("**Derived quantities**")
            for k, v in info["derived"].items():
                st.write(f"{k.replace('_', ' ').capitalize()}: **{v:.3f}**")
        fitted = load_csv("regression_fitted.csv", parse_dates=["date"])
        if fitted is not None:
            a, f = ("actual_inflation", "phillips_fitted") if tag == "phillips" else ("actual_repo", "taylor_fitted")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=fitted["date"], y=fitted[a], name="Actual", line=dict(color=TEXT, width=2)))
            fig.add_trace(go.Scatter(x=fitted["date"], y=fitted[f], name="Fitted",
                                     line=dict(color=MODEL_COLORS["SARIMA"], width=2)))
            st.plotly_chart(base_layout(fig, f"{which}: actual vs fitted", "%", 340), width="stretch")
        st.caption("Breusch-Godfrey p < 0.05 means autocorrelated residuals, which is why HAC standard errors are used. "
                   "A long-run Taylor-rule inflation response above 1 is the Taylor principle.")

# ---------------------------------------------------------------- forecasts
with tabs[3]:
    fut = load_csv("future_forecasts.csv", parse_dates=["date"])
    if fut is not None:
        n_hist = st.slider("Months of history to show", 12, 120, 48, step=12)
        hist = last.iloc[-n_hist:]
        models = st.multiselect("Models", list(fut["model"].unique()), default=list(fut["model"].unique()))
        fig = go.Figure()
        add_band(fig)
        if ci is not None and "SARIMA" in models:
            fig.add_trace(go.Scatter(x=list(ci["date"]) + list(ci["date"][::-1]),
                                     y=list(ci["upper"]) + list(ci["lower"][::-1]), fill="toself",
                                     fillcolor="rgba(42,120,214,0.15)", line=dict(width=0),
                                     name="SARIMA 80% interval", hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=hist.index, y=hist, name="Actual", line=dict(color=TEXT, width=2)))
        for model in models:  # filtering never repaints: colour is fixed per model
            g = fut[fut["model"] == model].sort_values("date")
            fig.add_trace(go.Scatter(x=[hist.index[-1], *g["date"]], y=[hist.iloc[-1], *g["forecast"]], name=model,
                                     line=dict(color=MODEL_COLORS.get(model, NEUTRAL), width=2,
                                               dash="dash" if model == "Random walk" else "solid")))
        st.plotly_chart(base_layout(fig, "CPI inflation forecasts, next 12 months", "% y-o-y", 460),
                        width="stretch")
        table = fut.pivot_table(index="date", columns="model", values="forecast").round(2)
        table.index = table.index.strftime("%b %Y")
        st.dataframe(table, width="stretch")
        imp = load_csv("ml_feature_importance.csv")
        if imp is not None and not imp.empty:
            with st.expander("What drives the ML forecast? (random forest feature importance)"):
                hcol = st.selectbox("Horizon", [c for c in imp.columns if c != "feature"])
                top = imp.sort_values(hcol, ascending=True).tail(12)
                f2 = go.Figure(go.Bar(x=top[hcol], y=top["feature"], orientation="h",
                                      marker_color=MODEL_COLORS["Random forest"]))
                st.plotly_chart(base_layout(f2, f"Top features, {hcol}", "", 380), width="stretch")
                st.caption("base_effect = the past m-o-m price changes that drop out of the y-o-y rate over the horizon.")

# ---------------------------------------------------------------- model comparison
with tabs[4]:
    met = load_csv("metrics.csv")
    if met is None:
        st.info("Run the backtest (`python scripts/run_all.py`) to see this tab.")
    else:
        excl = st.toggle("Exclude COVID months (Mar 2020 to Sep 2021) from scoring", value=False)
        met = load_csv("metrics_excl_covid.csv") if excl else met
        dm = load_csv("dm_vs_random_walk_excl_covid.csv" if excl else "dm_vs_random_walk.csv")
        cfg = load_json("model_config.json") or {}
        st.markdown(f"Each model is re-estimated every month from **{cfg.get('backtest_start', '')[:7]}** using "
                    "only data available up to that month, then scored on what actually happened.")
        fig = go.Figure()
        for model in [mm for mm in MODEL_COLORS if mm in set(met["model"])]:
            g = met[met["model"] == model]
            fig.add_trace(go.Bar(x=[f"{h} m ahead" for h in g["h"]], y=g["rmse"], name=model,
                                 marker_color=MODEL_COLORS[model]))
        fig.update_layout(barmode="group", bargap=0.25, bargroupgap=0.05)
        st.plotly_chart(base_layout(fig, "Out-of-sample RMSE (lower is better)", "pp", 400), width="stretch")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Accuracy by horizon**")
            st.dataframe(met.round(3), width="stretch", hide_index=True)
        with c2:
            st.markdown("**Diebold-Mariano tests vs random walk**")
            if dm is not None:
                st.dataframe(dm.round(3), width="stretch", hide_index=True)
            st.caption("Negative DM statistic = the model is more accurate than the random walk. "
                       "Harvey-Leybourne-Newbold small-sample correction applied.")
        bt = load_csv("backtest_forecasts.csv", parse_dates=["origin", "target"])
        if bt is not None:
            h = st.selectbox("Show backtest forecasts at horizon", sorted(bt["h"].unique()), index=1)
            g = bt[bt["h"] == h]
            fig = go.Figure()
            act = g.drop_duplicates("target").sort_values("target")
            fig.add_trace(go.Scatter(x=act["target"], y=act["actual"], name="Actual", line=dict(color=TEXT, width=2)))
            for model in [mm for mm in MODEL_COLORS if mm in set(g["model"])]:
                gm = g[g["model"] == model].sort_values("target")
                fig.add_trace(go.Scatter(x=gm["target"], y=gm["forecast"], name=model,
                                         line=dict(color=MODEL_COLORS[model], width=1.5)))
            st.plotly_chart(base_layout(fig, f"{h}-month-ahead forecasts vs outcome", "% y-o-y", 420),
                            width="stretch")

# ---------------------------------------------------------------- VAR
with tabs[5]:
    vinfo = load_json("var_info.json")
    if vinfo:
        st.markdown(f"VAR({vinfo['lags']}) on {', '.join(LABELS.get(v, v) for v in vinfo['variables'])}, "
                    f"with COVID dummies. Shocks identified by Cholesky ordering in that order "
                    f"(oil most exogenous). Stable: **{vinfo['is_stable']}**.")
        shock = st.radio("Shock", ["100 bp repo rate increase", "10% rise in Brent crude"], horizontal=True)
        irf = load_csv("var_irf_repo.csv" if shock.startswith("100") else "var_irf_oil.csv")
        responses = list(dict.fromkeys(irf["response"]))
        cols = st.columns(3)
        for i, r in enumerate(responses):
            g = irf[irf["response"] == r]
            c = VAR_COLORS.get(r, NEUTRAL)
            rgb = tuple(int(c[k:k + 2], 16) for k in (1, 3, 5))
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=list(g["h"]) + list(g["h"][::-1]), y=list(g["upper"]) + list(g["lower"][::-1]),
                                     fill="toself", fillcolor=f"rgba{rgb + (0.15,)}", line=dict(width=0),
                                     hoverinfo="skip", showlegend=False))
            fig.add_trace(go.Scatter(x=g["h"], y=g["irf"], line=dict(color=c, width=2), name=LABELS.get(r, r),
                                     showlegend=False))
            fig.add_hline(y=0, line=dict(color=TEXT_2, width=1))
            base_layout(fig, LABELS.get(r, r), "pp", 260)
            fig.update_layout(hovermode="x")
            fig.update_xaxes(title="months")
            cols[i % 3].plotly_chart(fig, width="stretch")
        st.caption("Shaded: 90% Monte Carlo confidence band. Responses in percentage points.")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**What explains inflation's forecast errors? (FEVD, %)**")
            fevd = load_csv("var_fevd_inflation.csv")
            st.dataframe(fevd.rename(columns=LABELS).round(1), width="stretch", hide_index=True)
        with c2:
            st.markdown("**Granger causality p-values (row helps predict column)**")
            gr = load_csv("var_granger.csv")
            gr = gr.set_index(gr.columns[0]).rename(index=LABELS, columns=LABELS)
            st.dataframe(gr.round(3), width="stretch")
        st.caption("Granger causality is about predictive content, not structural cause.")

# ---------------------------------------------------------------- about
with tabs[6]:
    readme = ROOT / "reports" / "results_summary.md"
    st.markdown(f"""
**Question.** How well can India's CPI inflation be forecast 1, 3 and 12 months ahead, and how does monetary
policy pass through to inflation, credit and output?

**Data.** {m.get('start', '?')} to {m.get('end', '?')}, monthly. CPI-Combined (MoSPI), policy repo rate and
scheduled commercial bank non-food credit (RBI DBIE), IIP (MoSPI), INR/USD and Brent crude (FRED).

**Methods.** ADF/KPSS/Zivot-Andrews stationarity tests; OLS Phillips curve and Taylor rule with Newey-West errors and
a Chow test at the 2016 move to inflation targeting; SARIMA on log CPI chosen by AIC; a VAR with Cholesky
impulse responses; random forest and gradient boosting with a direct multi-horizon setup; a rolling-origin
backtest scored with RMSE, MAE and Diebold-Mariano tests. Key results are replicated in R.

**Caveats.** Uses the latest data vintage, not real-time data. The output gap is an HP-filter estimate from IIP.
Cholesky identification depends on the variable ordering.
""")
    if readme.exists():
        with st.expander("Full results summary"):
            st.markdown(readme.read_text())
