"""Figure for the first example of the time series index: GARCH11 following the volatility of a stock.

Generates garch_volatility.png in the same folder, with the library itself.

Run from the repo root:
    python docs/assets/timeseries/make_garch_index.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from dense_armor.core.online.timeseries import GARCH11

HERE = Path(__file__).resolve().parent
BG, PANEL, GRID = "#0b1118", "#111a24", "#26364a"
INK, MUTED = "#e8fbff", "#8fa3b8"
CYAN, ORANGE, PINK = "#3ec6e8", "#ffa21f", "#ff2d87"

plt.rcParams.update(
    {
        "figure.facecolor": BG,
        "axes.facecolor": PANEL,
        "axes.edgecolor": GRID,
        "axes.labelcolor": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "text.color": INK,
        "grid.color": GRID,
        "grid.alpha": 0.6,
        "font.size": 10,
    }
)


def main():
    rng = np.random.default_rng(0)
    returns = np.concatenate(
        [
            rng.normal(0.0, 0.01, 250),
            rng.normal(0.0, 0.03, 50),
            rng.normal(0.0, 0.01, 100),
        ]
    )
    true_vol = np.concatenate([np.full(250, 1.0), np.full(50, 3.0), np.full(100, 1.0)])
    g = GARCH11(omega=1e-6, alpha=0.08, beta=0.90)
    before, after = [], []
    for r in returns:
        before.append(g.sigma2_**0.5)
        g.learn_one({"x": float(r)})
        after.append(g.sigma2_**0.5)
    before, after = np.array(before) * 100, np.array(after) * 100
    day = np.arange(1, len(returns) + 1)

    fig, (a1, a2) = plt.subplots(
        2,
        1,
        figsize=(16, 7.4),
        sharex=True,
        gridspec_kw={"height_ratios": [2.2, 1.4], "hspace": 0.1},
    )
    fig.subplots_adjust(top=0.82, bottom=0.09, left=0.06, right=0.985)
    fig.text(
        0.04,
        0.945,
        "GARCH FOLLOWS THE VOLATILITY OF A STOCK",
        fontsize=20,
        fontweight="bold",
        color=INK,
    )
    fig.text(
        0.04,
        0.885,
        "250 calm days, 50 days of turbulence, 100 calm days: the estimate rises with the storm and falls back after it",
        fontsize=11.5,
        color=CYAN,
    )
    for ax in (a1, a2):
        ax.axvspan(251, 300, color=ORANGE, alpha=0.10, lw=0)
        ax.grid(True)
    a1.vlines(day, 0, returns * 100, color=MUTED, lw=0.9, alpha=0.8)
    a1.fill_between(
        day,
        -2 * before,
        2 * before,
        color=CYAN,
        alpha=0.28,
        lw=0,
        label="±2 σ expected by the model before the day",
    )
    a1.plot(day, 2 * before, color=CYAN, lw=1.2)
    a1.plot(day, -2 * before, color=CYAN, lw=1.2)
    a1.set_ylabel("daily return [%]")
    a1.legend(loc="upper left", facecolor=PANEL, edgecolor=GRID, labelcolor=INK)
    a1.text(256, a1.get_ylim()[0] * 0.85, "turbulence", color=ORANGE, fontsize=11)
    a2.plot(day, after, color=PINK, lw=2.4, label="volatility estimated by GARCH11")
    a2.plot(day, true_vol, color=INK, lw=1.4, ls="--", label="true volatility")
    a2.set_ylabel("volatility [% per day]")
    a2.set_xlabel("day")
    a2.legend(loc="upper left", facecolor=PANEL, edgecolor=GRID, labelcolor=INK)
    a2.set_xlim(1, len(day))
    fig.savefig(HERE / "garch_volatility.png", dpi=150, facecolor=BG)
    plt.close(fig)
    print(
        "outside the band:",
        int(np.sum(np.abs(returns * 100) > 2 * before)),
        "of",
        len(returns),
    )


if __name__ == "__main__":
    main()
