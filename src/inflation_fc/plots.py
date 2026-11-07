"""Static figures (PNG) for the README and the policy note."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from . import config as C  # noqa: E402
from .style import BAND, GRID, LABELS, MODEL_COLORS, NEUTRAL, SURFACE, TEXT, TEXT_2, VAR_COLORS  # noqa: E402

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": TEXT_2, "xtick.color": TEXT_2, "ytick.color": TEXT_2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
    "axes.spines.right": False, "font.size": 10, "axes.titlesize": 12, "axes.titleweight": "bold",
    "axes.titlecolor": TEXT, "axes.titlelocation": "left", "lines.linewidth": 2, "legend.frameon": False,
})


def _save(fig, name):
    C.FIGURES.mkdir(parents=True, exist_ok=True)
    path = C.FIGURES / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def inflation_vs_target(panel: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(9, 4))
    lo, hi = C.TOLERANCE_BAND
    ax.axhspan(lo, hi, color=BAND, zorder=0, label=f"Tolerance band ({lo:.0f}–{hi:.0f}%)")
    ax.axhline(C.INFLATION_TARGET, color=NEUTRAL, lw=1, ls="--", label="4% target")
    ax.plot(panel.index, panel["inflation"], color=VAR_COLORS["inflation"], label="CPI inflation (y-o-y)")
    ax.plot(panel.index, panel["repo"], color=VAR_COLORS["repo"], label="Repo rate")
    ax.axvline(pd.Timestamp(C.FIT_BREAK_DATE), color=TEXT_2, lw=0.8)
    ax.text(pd.Timestamp(C.FIT_BREAK_DATE), ax.get_ylim()[1], " Inflation targeting",
            va="top", fontsize=8, color=TEXT_2)
    ax.set_ylabel("%")
    ax.set_title("CPI inflation and the policy repo rate")
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    return _save(fig, "01_inflation_repo.png")


def forecast_fan(panel: pd.DataFrame, future: pd.DataFrame, sarima_ci: pd.DataFrame | None = None):
    fig, ax = plt.subplots(figsize=(9, 4))
    hist = panel["inflation"].dropna().iloc[-48:]
    ax.plot(hist.index, hist, color=TEXT, label="Actual")
    if sarima_ci is not None:
        ax.fill_between(sarima_ci.index, sarima_ci["lower"], sarima_ci["upper"],
                        color=MODEL_COLORS["SARIMA"], alpha=0.15, lw=0, label="SARIMA 80% interval")
    for model, g in future.groupby("model"):
        g = g.sort_values("date")
        x = [hist.index[-1]] + list(g["date"])
        y = [hist.iloc[-1]] + list(g["forecast"])
        ax.plot(x, y, color=MODEL_COLORS.get(model, NEUTRAL), label=model,
                ls="--" if model in ("Random walk", "Seasonal average") else "-",
                marker="o" if len(g) < 6 else None, ms=4)
    lo, hi = C.TOLERANCE_BAND
    ax.axhspan(lo, hi, color=BAND, zorder=0)
    ax.set_ylabel("% y-o-y")
    ax.set_title("CPI inflation: next 12 months by model")
    ax.legend(fontsize=8, ncol=3, loc="upper left")
    return _save(fig, "02_forecasts.png")


def rmse_bars(metrics: pd.DataFrame, name: str = "03_rmse.png", title: str = "Out-of-sample RMSE by horizon"):
    models = [m for m in MODEL_COLORS if m in set(metrics["model"])]
    hs = sorted(metrics["h"].unique())
    fig, ax = plt.subplots(figsize=(9, 4))
    width = 0.8 / len(models)
    for i, m in enumerate(models):
        vals = metrics[metrics["model"] == m].set_index("h").reindex(hs)["rmse"]
        ax.bar([k + i * width for k in range(len(hs))], vals, width=width * 0.92,
               color=MODEL_COLORS[m], label=m)
    ax.set_xticks([k + width * (len(models) - 1) / 2 for k in range(len(hs))])
    ax.set_xticklabels([f"{h} month{'s' if h > 1 else ''} ahead" for h in hs])
    ax.set_ylabel("RMSE (percentage points)")
    ax.set_title(title)
    ax.legend(fontsize=8, ncol=3)
    ax.grid(axis="x", visible=False)
    return _save(fig, name)


def irf_grid(irf: pd.DataFrame, name: str = "04_irf_repo.png"):
    responses = list(dict.fromkeys(irf["response"]))
    n = len(responses)
    cols = 3
    rows = -(-n // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(10, 3 * rows), sharex=True)
    for ax, r in zip(axes.flat, responses):
        g = irf[irf["response"] == r]
        c = VAR_COLORS.get(r, NEUTRAL)
        ax.fill_between(g["h"], g["lower"], g["upper"], color=c, alpha=0.15, lw=0)
        ax.plot(g["h"], g["irf"], color=c)
        ax.axhline(0, color=TEXT_2, lw=0.8)
        ax.set_title(LABELS.get(r, r), fontsize=10)
    for ax in list(axes.flat)[n:]:
        ax.set_visible(False)
    fig.suptitle("Responses to a 100 bp repo rate shock (pp, 90% bands)", x=0.01, ha="left",
                 fontweight="bold", color=TEXT)
    fig.supxlabel("Months after shock", color=TEXT_2, fontsize=9)
    fig.tight_layout()
    return _save(fig, name)


def phillips_fit(panel: pd.DataFrame, fitted: pd.Series):
    fig, ax = plt.subplots(figsize=(9, 3.6))
    ax.plot(panel.index, panel["inflation"], color=TEXT, label="Actual")
    ax.plot(fitted.index, fitted, color=MODEL_COLORS["SARIMA"], label="Phillips curve fitted")
    ax.set_ylabel("% y-o-y")
    ax.set_title("Phillips curve: in-sample fit")
    ax.legend(fontsize=8)
    return _save(fig, "05_phillips_fit.png")
