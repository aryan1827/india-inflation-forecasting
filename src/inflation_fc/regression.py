"""Single-equation econometrics: a Phillips curve and a Taylor-type rule.

Both are estimated by OLS with Newey-West (HAC) standard errors, because
monthly macro residuals are almost always autocorrelated and heteroskedastic.
A full set of residual diagnostics and a Chow test for a structural break at
the 2016 adoption of inflation targeting are reported alongside.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.diagnostic import acorr_breusch_godfrey, het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson, jarque_bera

from . import config as C

HAC_LAGS = 12


@dataclass
class RegressionResult:
    name: str
    equation: str
    coef: pd.DataFrame
    diagnostics: dict
    chow: dict
    extra: dict = field(default_factory=dict)
    summary_text: str = ""


def _design(panel: pd.DataFrame, y: str, X: dict[str, pd.Series]) -> tuple[pd.Series, pd.DataFrame]:
    data = pd.concat({y: panel[y], **X}, axis=1).dropna()
    Xd = data.drop(columns=y)
    # Drop regressors that never vary in this sample (e.g. the COVID dummy
    # before 2020), otherwise the design matrix is singular.
    Xd = Xd.loc[:, Xd.nunique() > 1]
    return data[y], sm.add_constant(Xd)


def _diagnostics(res_ols, X: pd.DataFrame) -> dict:
    bg = acorr_breusch_godfrey(res_ols, nlags=12)
    bp = het_breuschpagan(res_ols.resid, X)
    jb = jarque_bera(res_ols.resid)
    vif = {c: variance_inflation_factor(X.values, i) for i, c in enumerate(X.columns) if c != "const"}
    return {
        "n_obs": int(res_ols.nobs),
        "r2": res_ols.rsquared,
        "adj_r2": res_ols.rsquared_adj,
        "durbin_watson": durbin_watson(res_ols.resid),
        "breusch_godfrey_p (12 lags)": bg[1],
        "breusch_pagan_p": bp[1],
        "jarque_bera_p": jb[1],
        "max_vif": max(vif.values()) if vif else float("nan"),
        "vif": vif,
    }


def chow_test(y: pd.Series, X: pd.DataFrame, break_date: str) -> dict:
    """Classic Chow F-test for a break in all coefficients at break_date."""
    mask = y.index >= pd.Timestamp(break_date)
    if mask.sum() <= X.shape[1] or (~mask).sum() <= X.shape[1]:
        return {"break_date": break_date, "F": np.nan, "p_value": np.nan}
    def ssr(yy, XX):
        XX = XX.loc[:, (XX.nunique() > 1) | (XX.columns == "const")]  # e.g. COVID dummy pre-2020
        return sm.OLS(yy, XX).fit().ssr

    rss_p = ssr(y, X)
    rss_1 = ssr(y[~mask], X[~mask])
    rss_2 = ssr(y[mask], X[mask])
    k, n = X.shape[1], len(y)
    F = ((rss_p - rss_1 - rss_2) / k) / ((rss_1 + rss_2) / (n - 2 * k))
    return {"break_date": break_date, "F": F, "p_value": 1 - stats.f.cdf(F, k, n - 2 * k),
            "n_before": int((~mask).sum()), "n_after": int(mask.sum())}


def _fit(name: str, equation: str, y: pd.Series, X: pd.DataFrame) -> tuple[RegressionResult, object]:
    ols = sm.OLS(y, X).fit()
    hac = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": HAC_LAGS})
    coef = pd.DataFrame({
        "coef": hac.params, "se_hac": hac.bse, "t": hac.tvalues, "p_value": hac.pvalues,
        "ci_low": hac.conf_int()[0], "ci_high": hac.conf_int()[1],
    })
    res = RegressionResult(
        name=name, equation=equation, coef=coef,
        diagnostics=_diagnostics(ols, X),
        chow=chow_test(y, X, C.FIT_BREAK_DATE),
        summary_text=hac.summary().as_text(),
    )
    return res, hac


def phillips_curve(panel: pd.DataFrame, sample_start: str | None = None,
                   sample_end: str | None = None) -> RegressionResult:
    """Backward-looking open-economy Phillips curve.

    inflation_t = a + b1*inflation_{t-1} + b2*gap_{t-1} + b3*oil_yoy_{t-1}
                  + b4*inr_yoy_{t-1} + d*covid_t + e_t
    """
    p = panel.loc[sample_start:sample_end]
    X = {
        "inflation_lag1": p["inflation"].shift(1),
        "output_gap_lag1": p["output_gap"].shift(1),
        "oil_yoy_lag1": p["oil_yoy"].shift(1),
        "inr_yoy_lag1": p["inr_yoy"].shift(1),
        "covid": p["covid"],
    }
    y, Xd = _design(p, "inflation", X)
    res, hac = _fit("Phillips curve",
                    "π_t = α + β₁π_{t-1} + β₂gap_{t-1} + β₃oil_{t-1} + β₄inr_{t-1} + δ·covid_t + ε_t",
                    y, Xd)
    rho = hac.params["inflation_lag1"]
    res.extra["long_run_oil_passthrough"] = hac.params["oil_yoy_lag1"] / (1 - rho)
    res.extra["long_run_inr_passthrough"] = hac.params["inr_yoy_lag1"] / (1 - rho)
    res.extra["long_run_gap_effect"] = hac.params["output_gap_lag1"] / (1 - rho)
    res.extra["fitted"] = hac.fittedvalues
    res.extra["resid"] = hac.resid
    return res


def taylor_rule(panel: pd.DataFrame, sample_start: str | None = None,
                sample_end: str | None = None) -> RegressionResult:
    """Taylor rule with interest-rate smoothing.

    repo_t = a + rho*repo_{t-1} + b*(inflation_t - 4) + c*gap_t + e_t
    Long-run responses are b/(1-rho) and c/(1-rho). A long-run inflation
    response above 1 is the 'Taylor principle'.
    """
    p = panel.loc[sample_start:sample_end]
    X = {
        "repo_lag1": p["repo"].shift(1),
        "inflation_gap": p["inflation"] - C.INFLATION_TARGET,
        "output_gap": p["output_gap"],
    }
    y, Xd = _design(p, "repo", X)
    res, hac = _fit("Taylor rule",
                    "i_t = α + ρ·i_{t-1} + β(π_t − 4) + γ·gap_t + ε_t", y, Xd)
    rho = hac.params["repo_lag1"]
    res.extra["smoothing_rho"] = rho
    res.extra["long_run_inflation_response"] = hac.params["inflation_gap"] / (1 - rho)
    res.extra["long_run_gap_response"] = hac.params["output_gap"] / (1 - rho)
    res.extra["implied_neutral_rate"] = hac.params["const"] / (1 - rho)
    res.extra["fitted"] = hac.fittedvalues
    res.extra["resid"] = hac.resid
    return res


def subsample_comparison(panel: pd.DataFrame, fn) -> pd.DataFrame:
    """Re-estimate before and after inflation targeting and put the
    coefficients side by side."""
    brk = pd.Timestamp(C.FIT_BREAK_DATE)
    pre = fn(panel, sample_end=brk - pd.offsets.MonthBegin(1))
    post = fn(panel, sample_start=brk)
    full = fn(panel)
    return pd.DataFrame({
        "full_sample": full.coef["coef"],
        "pre_FIT": pre.coef["coef"],
        "post_FIT": post.coef["coef"],
        "pre_FIT_p": pre.coef["p_value"],
        "post_FIT_p": post.coef["p_value"],
    })
