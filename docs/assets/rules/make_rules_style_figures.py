"""The figures of the Rules pages (3D at the start and finance figure), drawn with the library itself.

A desk predicts the slippage of an order, in basis points, from the size of the order, the spread and the volatility
(three numbers between -1 and 1). The slippage is 5 when the size is above 0 and 0 otherwise, plus a noise of 0.1; the
spread and the volatility say nothing. On the page of the change, the rule of the desk is swapped after 2000 orders:
the slippage is 5 when the size is not above 0. The score is the mean absolute error of the prediction made before each
order is learned, averaged over blocks of 100 orders, as the median of 3 seeds.

Run from the repo root:
    python docs/assets/rules/make_rules_style_figures.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from dense_armor.core.online.rules import AMRulesRegressor

HERE = Path(__file__).resolve().parent / "pages"
HERE.mkdir(exist_ok=True)
BG, PANEL, GRID = "#0b1118", "#111a24", "#26364a"
INK, MUTED = "#e8fbff", "#8fa3b8"
CYAN, ORANGE, PINK, GREEN = "#3ec6e8", "#ffa21f", "#ff2d87", "#46d37f"
CMAP = LinearSegmentedColormap.from_list("s", [CYAN, "#6a5acd", PINK])
SEEDS = 3
BLOCK = 100


def stream(n, seed, swap_at=None):
    r = np.random.default_rng(seed)
    for t in range(n):
        size, spread, vol = (float(r.uniform(-1, 1)) for _ in range(3))
        up = size > 0.0
        if swap_at is not None and t >= swap_at:
            up = not up
        yield {"size": size, "spread": spread, "vol": vol}, 5.0 * up + 0.1 * float(r.normal())


def curve(make, n, seed, swap_at=None):
    """Mean absolute error of each block of 100 orders, predict first and learn after."""
    m = make()
    err = np.zeros(n)
    for t, (x, y) in enumerate(stream(n, seed, swap_at)):
        err[t] = abs(m.predict_one(x) - y)
        m.learn_one(x, y)
    return err.reshape(-1, BLOCK).mean(axis=1)


def med(make, n, swap_at=None):
    return np.median([curve(make, n, k, swap_at) for k in range(SEEDS)], axis=0)


def surface(Z, xs, ys, name, title, subtitle, xl, yl, zl, zmax=None, ticks=None):
    X, Y = np.meshgrid(xs, ys)
    fig = plt.figure(figsize=(12, 8.2))
    ax = fig.add_subplot(111, projection="3d")
    fig.subplots_adjust(top=0.93, bottom=0.0, left=-0.05, right=1.05)
    ax.set_facecolor(BG)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_pane_color((0.07, 0.10, 0.14, 1.0))
        axis._axinfo["grid"]["color"] = GRID
    ax.view_init(elev=24, azim=-58)
    ax.set_box_aspect((1.35, 1.2, 0.7), zoom=1.1)
    ax.set_xlabel(xl, color=CYAN, labelpad=8)
    ax.set_ylabel(yl, color=CYAN, labelpad=8)
    ax.set_zlabel(zl, color=CYAN, labelpad=6)
    ax.tick_params(colors=MUTED, labelsize=8)
    if ticks is not None:
        ax.set_yticks(ticks)
    fig.text(0.05, 0.955, title, fontsize=21, fontweight="bold", color=INK)
    fig.text(0.05, 0.915, subtitle, fontsize=11.5, color=CYAN)
    top = zmax if zmax else max(Z.max(), 1e-9)
    Zc = np.minimum(Z, top)
    ax.plot_surface(X, Y, Zc, facecolors=CMAP(np.clip(Zc / top, 0, 1)), rstride=1, cstride=1, linewidth=0, antialiased=False, alpha=0.93)
    ax.set_zlim(0, top)
    fig.savefig(HERE / f"{name}_3d.png", dpi=150, facecolor=BG)
    plt.close(fig)


def lines(curves, xs, name, title, ylim=None, loc="upper right", vline=None):
    fig, ax = plt.subplots(figsize=(9, 3.6), facecolor=BG)
    ax.set_facecolor(PANEL)
    ax.set_xlabel("orders seen", color=CYAN)
    ax.set_ylabel("mean absolute error (bp)", color=CYAN)
    ax.tick_params(colors=MUTED)
    for s in ax.spines.values():
        s.set_color(GRID)
    ax.grid(color=GRID, lw=0.5)
    ax.set_title(title, color=INK, loc="left", fontsize=13, fontweight="bold")
    for (lab, v), col in zip(curves, (CYAN, ORANGE, PINK, GREEN)):
        ax.plot(xs, v, color=col, lw=2.0, label=lab)
    if vline:
        ax.axvline(vline, color=MUTED, lw=1.2, ls="--", label="the rule is swapped")
    if ylim:
        ax.set_ylim(*ylim)
    ax.legend(facecolor=PANEL, edgecolor=GRID, labelcolor=INK, fontsize=9, loc=loc)
    fig.tight_layout()
    fig.savefig(HERE / f"{name}_finance.png", dpi=150, facecolor=BG)
    plt.close(fig)


def at(v, pts):
    return [round(float(v[p]), 3) for p in pts]


def main():
    n = 3000
    xs = np.arange(1, n // BLOCK + 1) * BLOCK
    nm = np.unique(np.geomspace(20, 400, 25).astype(int))
    Z = np.array([med(lambda v=v: AMRulesRegressor(n_min=int(v), seed=0), n) for v in nm])
    surface(Z, xs, nm.astype(float), "rules_learn", "A RULE WAITS FOR ENOUGH ORDERS BEFORE IT CUTS", "mean absolute error of the slippage over 3000 orders, for 25 values of n_min, the orders a rule sees before it tries a cut", "orders seen", "n_min", "mean absolute error (bp)", zmax=3.0)
    blocks = (0, 4, 9, 29)
    for v in (20, 100, 400):
        k = int(np.argmin(np.abs(nm - v)))
        print("learn n_min", int(nm[k]), "MAE in the blocks ending at 100, 500, 1000, 3000:", at(Z[k], blocks))
    lines([("n_min " + str(v), med(lambda v=v: AMRulesRegressor(n_min=v, seed=0), n)) for v in (20, 100, 400)], xs, "rules_learn", "Error of the rules, for three values of n_min", ylim=(0, 3.0))

    n2 = 4000
    xs2 = np.arange(1, n2 // BLOCK + 1) * BLOCK
    lam = np.geomspace(2, 300, 25)
    Zc = np.array([med(lambda v=v: AMRulesRegressor(n_min=100, ph_lambda=float(v), seed=0), n2, swap_at=2000) for v in lam])
    surface(Zc, xs2, np.log10(lam), "rules_change", "A LOW ALARM THRESHOLD RECOVERS SOONER", "mean absolute error when the rule of the desk is swapped after 2000 orders, for 25 values of the Page-Hinkley threshold ph_lambda", "orders seen", "log10 of ph_lambda", "mean absolute error (bp)", zmax=5.0, ticks=[0.5, 1.0, 1.5, 2.0, 2.5])
    for v in (2, 35, 300):
        k = int(np.argmin(np.abs(lam - v)))
        print("change ph_lambda", round(float(lam[k]), 1), "MAE in the blocks ending at 1900, 2100, 2500, 4000:", at(Zc[k], (18, 20, 24, 39)))
    lines([("ph_lambda " + str(v), med(lambda v=v: AMRulesRegressor(n_min=100, ph_lambda=v, seed=0), n2, swap_at=2000)) for v in (2.0, 35.0, 300.0)], xs2, "rules_change", "Error when the rule is swapped, for three alarm thresholds", ylim=(0, 5.0), vline=2000)

    mr = np.arange(1, 21)
    Zm = np.array([med(lambda v=v: AMRulesRegressor(n_min=100, max_rules=int(v), seed=0), n) for v in mr])
    surface(Zm, xs, mr.astype(float), "rules_options", "TOO FEW RULES KEEP THE ERROR UP", "mean absolute error over 3000 orders, for the largest number of rules from 1 to 20", "orders seen", "max_rules", "mean absolute error (bp)", zmax=3.0, ticks=[1, 5, 10, 15, 20])
    for v in (1, 5, 20):
        k = int(np.argmin(np.abs(mr - v)))
        print("options max_rules", int(mr[k]), "MAE in the blocks ending at 100, 500, 1000, 3000:", at(Zm[k], blocks))
    lines([(p, med(lambda p=p: AMRulesRegressor(n_min=100, prediction=p, seed=0), n)) for p in ("adaptive", "mean", "perceptron")], xs, "rules_options", "Error of the rules, for the three ways a rule predicts", ylim=(0, 3.0))
    for p in ("adaptive", "mean", "perceptron"):
        print("options prediction", p, at(med(lambda p=p: AMRulesRegressor(n_min=100, prediction=p, seed=0), n), blocks))


if __name__ == "__main__":
    main()
