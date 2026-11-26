"""The figures of the Regression pages, drawn with the library itself.

*_3d: the daily returns of a stock index (percent, one stream of 300 days) and a forecast that is the return plus a
    noise of size sigma (the same noise draws for every sigma); the surfaces are the running metric for 25 values
    of sigma from 0.1 to 1.4.
*_finance: a stock price over 250 days (a random walk) and the forecast "tomorrow equals today"; the lines are the
    running metrics of that forecast.

Run from the repo root:
    python docs/assets/metrics/make_regression_figures.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from dense_armor.core.evaluation.metrics import (
    MeanAbsoluteError,
    MeanSquaredError,
    R2Score,
    RootMeanSquaredError,
)

HERE = Path(__file__).resolve().parent
BG, PANEL, GRID = "#0b1118", "#111a24", "#26364a"
INK, MUTED = "#e8fbff", "#8fa3b8"
CYAN, ORANGE, PINK = "#3ec6e8", "#ffa21f", "#ff2d87"
CMAP = LinearSegmentedColormap.from_list("s", [CYAN, "#6a5acd", PINK])
N3, SIGMAS = 300, np.linspace(0.1, 1.4, 25)
RNG = np.random.default_rng(31)
RET, NOISE = RNG.normal(0, 1, N3), RNG.normal(0, 1, N3)


def run(factory, truth, pred):
    m = factory()
    out = np.zeros(len(truth))
    for i, (a, b) in enumerate(zip(truth, pred)):
        m.update(float(a), float(b))
        out[i] = m.get()
    return out


def surface(series, name, title, subtitle, zl, zlim):
    T, Y = np.meshgrid(np.arange(20, N3 + 1), SIGMAS)
    fig = plt.figure(figsize=(12, 8.2))
    ax = fig.add_subplot(111, projection="3d")
    fig.subplots_adjust(top=0.93, bottom=0.0, left=-0.05, right=1.05)
    ax.set_facecolor(BG)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_pane_color((0.07, 0.10, 0.14, 1.0))
        axis._axinfo["grid"]["color"] = GRID
    ax.view_init(elev=24, azim=-58)
    ax.set_box_aspect((1.35, 1.2, 0.7), zoom=1.1)
    ax.set_xlabel("days seen", color=CYAN, labelpad=8)
    ax.set_ylabel("size of the forecast noise", color=CYAN, labelpad=8)
    ax.set_zlabel(zl, color=CYAN, labelpad=6)
    ax.tick_params(colors=MUTED, labelsize=8)
    fig.text(0.05, 0.955, title, fontsize=21, fontweight="bold", color=INK)
    fig.text(0.05, 0.915, subtitle, fontsize=11.5, color=CYAN)
    if len(series) == 1:
        Z = series[0]
        ax.plot_surface(T, Y, Z, facecolors=CMAP(np.clip((Z - zlim[0]) / (zlim[1] - zlim[0]), 0, 1)), rstride=1, cstride=4, linewidth=0, antialiased=False, alpha=0.93)
    else:
        for Z, col in zip(series, (CYAN, PINK)):
            ax.plot_surface(T, Y, Z, color=col, rstride=1, cstride=6, linewidth=0, antialiased=False, alpha=0.8)
    ax.set_zlim(*zlim)
    fig.savefig(HERE / f"{name}_3d.png", dpi=150, facecolor=BG)
    plt.close(fig)


def sweep(factory):
    return np.array([run(factory, RET, RET + s * NOISE)[19:] for s in SIGMAS])


def lines(items, name, title, ylabel, loc="upper right"):
    n = 250
    walk = 100 + np.cumsum(np.random.default_rng(6).normal(0, 1, n + 1))
    truth, fc = walk[1:], walk[:-1]
    fig, ax = plt.subplots(figsize=(9, 3.6), facecolor=BG)
    ax.set_facecolor(PANEL)
    ax.set_xlabel("trading days seen", color=CYAN)
    ax.set_ylabel(ylabel, color=CYAN)
    ax.tick_params(colors=MUTED)
    for s in ax.spines.values():
        s.set_color(GRID)
    ax.grid(color=GRID, lw=0.5)
    ax.set_title(title, color=INK, loc="left", fontsize=13, fontweight="bold")
    out = {}
    for factory, col, lab in items:
        v = run(factory, truth, fc)
        out[lab] = [round(float(v[k]), 3) for k in (9, 49, 99, n - 1)]
        ax.plot(np.arange(1, n + 1), v, color=col, lw=2.0, label=lab)
    ax.legend(facecolor=PANEL, edgecolor=GRID, labelcolor=INK, fontsize=9, loc=loc)
    fig.tight_layout()
    fig.savefig(HERE / f"{name}_finance.png", dpi=150, facecolor=BG)
    plt.close(fig)
    print(name, "after 10, 50, 100, 250 days:", out)


if __name__ == "__main__":
    mae, mse, rmse, r2 = (sweep(f) for f in (MeanAbsoluteError, MeanSquaredError, RootMeanSquaredError, R2Score))
    k = lambda s: int(np.argmin(np.abs(SIGMAS - s)))
    surface([mae], "regression_mae", "MAE GROWS WITH THE SIZE OF THE FORECAST ERROR",
            "daily returns of a stock index, same 300 days: running MAE for 25 sizes of forecast noise", "running MAE", (0, 1.4))
    surface([mse, rmse], "regression_mse_rmse", "MSE AND RMSE FOR THE SAME FORECASTS",
            "same 300 days: running MSE (cyan) and RMSE (pink) for 25 sizes of forecast noise", "score", (0, 2.6))
    surface([r2], "regression_r2", "R2 BELOW ZERO: WORSE THAN THE MEAN",
            "same 300 days: running R2 for 25 sizes of forecast noise", "running R2", (-2.0, 1.0))
    for nm, z in (("MAE", mae), ("MSE", mse), ("RMSE", rmse), ("R2", r2)):
        print(nm, "after 300 days, sigma 0.1 / 0.5 / 1.0 / 1.4:", [round(float(z[k(s), -1]), 3) for s in (0.1, 0.5, 1.0, 1.4)], "(sigma", [round(float(SIGMAS[k(s)]), 3) for s in (0.1, 0.5, 1.0, 1.4)], ")")
    lines([(MeanAbsoluteError, CYAN, "MAE")], "regression_mae", "A stock price: running MAE of the forecast tomorrow = today", "MAE (price units)")
    lines([(MeanSquaredError, CYAN, "MSE"), (RootMeanSquaredError, PINK, "RMSE")], "regression_mse_rmse", "A stock price: running MSE and RMSE of the same forecast", "score")
    lines([(R2Score, CYAN, "R2")], "regression_r2", "A stock price: running R2 of the same forecast", "R2", "lower right")
