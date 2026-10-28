"""Static matplotlib figures (light surface). Categorical colours follow a fixed per-model order."""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd

SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
# fixed colour per entity (never re-assigned by rank): slots 1-5 of the reference categorical palette
COLORS = {"Persistence": "#2a78d6", "Historical mean": "#eb6834", "Linear Regression": "#1baf7a",
          "Random Forest": "#eda100", "Gradient Boosting": "#e87ba4"}
REGIME_ORDER = ["decline", "steady", "boom"]


def _style(ax, xlabel="", ylabel="", title=""):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.set_xlabel(xlabel, color=INK2, fontsize=10)
    ax.set_ylabel(ylabel, color=INK2, fontsize=10)
    if title:
        ax.set_title(title, color=INK, fontsize=11, loc="left", fontweight="bold")


def _fig(w=9, h=5):
    fig, ax = plt.subplots(figsize=(w, h), facecolor=SURFACE)
    return fig, ax


def _legend(ax, ncol=3, **_):
    # legend sits below the plot so it never covers data
    leg = ax.legend(frameon=False, fontsize=9, ncol=ncol, loc="upper center", bbox_to_anchor=(0.5, -0.16))
    for t in leg.get_texts():
        t.set_color(INK2)


def _save(fig, out_dir, name):
    path = os.path.join(out_dir, name)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return path


def pred_vs_actual(res, models, out_dir):
    n = len(models)
    cols = 3
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.3 * cols, 4.0 * rows), facecolor=SURFACE, squeeze=False)
    lim = np.nanpercentile(np.r_[res["actual"], *(res[m] for m in models)], [0.5, 99.5])
    for ax, m in zip(axes.flat, models):
        ax.scatter(res["actual"], res[m], s=6, alpha=0.35, color=COLORS[m], linewidths=0)
        ax.plot(lim, lim, color=MUTED, linewidth=1, linestyle="--")
        ax.set_xlim(lim); ax.set_ylim(lim)
        _style(ax, "Actual next-year growth (%)", "Predicted (%)", m)
        ax.grid(axis="x", color=GRID, linewidth=0.8)
    for ax in list(axes.flat)[n:]:
        ax.axis("off")
    return _save(fig, out_dir, "pred_vs_actual.png")


def national_timeseries(res, models, out_dir):
    g = res.groupby("t")[["actual"] + models].mean()
    x = g.index / 4
    fig, ax = _fig(10, 5)
    ax.plot(x, g["actual"], color=INK, linewidth=2.4, label="Actual")
    for m in models:
        ax.plot(x, g[m], color=COLORS[m], linewidth=1.6, label=m)
    ax.axhline(0, color=AXIS, linewidth=1)
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    _style(ax, "Forecast origin (year)", "Next-year HPI growth (%), mean across places",
           "Average predicted vs actual next-year growth")
    _legend(ax, ncol=3, loc="upper right")
    return _save(fig, out_dir, "national_timeseries.png")


def mae_by_year(by_year, models, out_dir):
    piv = by_year.pivot(index="group", columns="model", values="MAE")
    fig, ax = _fig(10, 5)
    for m in models:
        ax.plot(piv.index, piv[m], color=COLORS[m], linewidth=2, label=m)
    _style(ax, "Test (origin) year", "MAE (percentage points)", "Forecast error by test year")
    _legend(ax, ncol=3, loc="upper left")
    return _save(fig, out_dir, "mae_by_year.png")


def mae_by_regime(by_regime, models, out_dir):
    piv = by_regime.pivot(index="group", columns="model", values="MAE").reindex(REGIME_ORDER).dropna(how="all")
    fig, ax = _fig(9, 5)
    k = len(models)
    w = 0.8 / k
    x = np.arange(len(piv))
    for i, m in enumerate(models):
        ax.bar(x + (i - (k - 1) / 2) * w, piv[m], width=w - 0.03, color=COLORS[m], label=m)
    ax.set_xticks(x, [f"{r}\n(n={int(by_regime[by_regime['group'] == r]['n'].iloc[0]):,})" for r in piv.index])
    _style(ax, "Realised market regime (national next-year growth)", "MAE (percentage points)",
           "Forecast error by market regime")
    _legend(ax, ncol=3, loc="upper left")
    return _save(fig, out_dir, "mae_by_regime.png")


def worst_places_plot(tbl, models, out_dir):
    tbl = tbl.iloc[::-1]
    fig, ax = _fig(9, 6)
    y = np.arange(len(tbl))
    k = len(models)
    h = 0.8 / k
    for i, m in enumerate(models):
        ax.barh(y + (i - (k - 1) / 2) * h, tbl[m], height=h - 0.03, color=COLORS[m], label=m)
    ax.set_yticks(y, tbl.index)
    _style(ax, "MAE (percentage points)", "", "Places with the largest forecast error")
    ax.grid(axis="x", color=GRID, linewidth=0.8); ax.grid(axis="y", visible=False)
    _legend(ax, loc="lower right")
    return _save(fig, out_dir, "worst_places.png")
