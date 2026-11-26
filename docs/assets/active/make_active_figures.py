"""The figures of the Active learning pages (3D at the start and finance figure), drawn with the library itself.

A compliance desk receives a stream of trades. Each trade is normal (0) or suspicious (1), and a human review gives the
true label at a cost, so only a fraction (the budget) can be reviewed. The score of a trade is Normal(0, 1) for a normal
one and Normal(2, 1) for a suspicious one, 30 % of the trades are suspicious, and the classifier is OnlineGaussianNB. It is
warmed up on the first two trades of each class, which are not counted by the strategy. The surface of each page is the labelling rate
so far (or the accuracy so far, on the random page) over 1000 trades, for 25 values of one parameter of the strategy.

Run from the repo root:
    python docs/assets/active/make_active_figures.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from dense_armor.core.online.active import (
    FixedUncertainty,
    RandomStrategy,
    VariableRandomizedUncertainty,
    VariableUncertainty,
)
from dense_armor.core.online.learn.online_classifiers import OnlineGaussianNB

HERE = Path(__file__).resolve().parent
BG, PANEL, GRID = "#0b1118", "#111a24", "#26364a"
INK, MUTED = "#e8fbff", "#8fa3b8"
CYAN, ORANGE, PINK, GREEN = "#3ec6e8", "#ffa21f", "#ff2d87", "#46d37f"
CMAP = LinearSegmentedColormap.from_list("s", [CYAN, "#6a5acd", PINK])
T = 1000
WARM = 2
SEEDS = 10


def stream(n=T, seed=0):
    r = np.random.default_rng(seed)
    y = (r.random(n) < 0.3).astype(int)
    return r.normal(2.0 * y, 1.0), y


def run(make, seed=0, n=T):
    """Labelling rate so far and accuracy so far, per trade."""
    s, y = stream(n, seed)
    model = OnlineGaussianNB()
    for c in (0, 1):
        for t in np.flatnonzero(y == c)[:WARM]:
            model.learn_one({"score": float(s[t])}, int(y[t]))
    st = make()
    rate = np.zeros(n)
    acc = np.zeros(n)
    ok = 0
    for t in range(n):
        x = {"score": float(s[t])}
        p = model.predict_proba_one(x)
        ok += int((max(p, key=p.get) if p else 0) == y[t])
        if st.should_label(x, model):
            model.learn_one(x, int(y[t]))
        rate[t] = st.n_labelled_ / (t + 1)
        acc[t] = ok / (t + 1)
    return rate, acc


def med(make, which):
    return np.median([run(make, seed=k)[which] for k in range(SEEDS)], axis=0)


def surface(Z, ys, name, title, subtitle, yl, zl, zlim=None):
    X, Y = np.meshgrid(np.arange(20, T + 1), ys)
    fig = plt.figure(figsize=(12, 8.2))
    ax = fig.add_subplot(111, projection="3d")
    fig.subplots_adjust(top=0.93, bottom=0.0, left=-0.05, right=1.05)
    ax.set_facecolor(BG)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_pane_color((0.07, 0.10, 0.14, 1.0))
        axis._axinfo["grid"]["color"] = GRID
    ax.view_init(elev=24, azim=-58)
    ax.set_box_aspect((1.35, 1.2, 0.7), zoom=1.1)
    ax.set_xlabel("trades seen", color=CYAN, labelpad=8)
    ax.set_ylabel(yl, color=CYAN, labelpad=8)
    ax.set_zlabel(zl, color=CYAN, labelpad=6)
    ax.tick_params(colors=MUTED, labelsize=8)
    fig.text(0.05, 0.955, title, fontsize=21, fontweight="bold", color=INK)
    fig.text(0.05, 0.915, subtitle, fontsize=11.5, color=CYAN)
    top = zlim[1] if zlim else max(Z.max(), 1e-9)
    ax.plot_surface(X, Y, Z, facecolors=CMAP(np.clip(Z / top, 0, 1)), rstride=1, cstride=3, linewidth=0, antialiased=False, alpha=0.93)
    if zlim:
        ax.set_zlim(*zlim)
    fig.savefig(HERE / f"{name}_3d.png", dpi=150, facecolor=BG)
    plt.close(fig)


def lines(curves, name, title, ylabel, ylim=None, loc="upper right", hline=None):
    fig, ax = plt.subplots(figsize=(9, 3.6), facecolor=BG)
    ax.set_facecolor(PANEL)
    ax.set_xlabel("trades seen", color=CYAN)
    ax.set_ylabel(ylabel, color=CYAN)
    ax.tick_params(colors=MUTED)
    for s in ax.spines.values():
        s.set_color(GRID)
    ax.grid(color=GRID, lw=0.5)
    ax.set_title(title, color=INK, loc="left", fontsize=13, fontweight="bold")
    for (lab, v), col in zip(curves, (CYAN, ORANGE, PINK, GREEN)):
        ax.plot(np.arange(1, len(v) + 1), v, color=col, lw=2.0, label=lab)
    if hline is not None:
        ax.axhline(hline, color=MUTED, lw=1.2, ls="--", label=f"budget {hline}")
    if ylim:
        ax.set_ylim(*ylim)
    ax.legend(facecolor=PANEL, edgecolor=GRID, labelcolor=INK, fontsize=9, loc=loc)
    fig.tight_layout()
    fig.savefig(HERE / f"{name}_finance.png", dpi=150, facecolor=BG)
    plt.close(fig)


def at(v, days=(50, 200, 1000)):
    return [round(float(v[d - 1]), 3) for d in days]


def sweep(make_for, params, which):
    return np.array([med(lambda v=v: make_for(v), which) for v in params])


def main():
    bud = np.linspace(0.02, 1.0, 25)
    Z = sweep(lambda b: RandomStrategy(budget=float(b), seed=0), bud, 1)
    surface(Z[:, 19:], bud, "active_random", "MORE LABELS HELP LITTLE ON AN EASY STREAM", "a compliance desk reviews a random fraction of 1000 trades: accuracy so far of the classifier for 25 budgets", "budget", "accuracy so far", zlim=(0.6, 1.0))
    for b in (0.02, 0.1, 1.0):
        k = int(np.argmin(np.abs(bud - b)))
        print("random budget", round(float(bud[k]), 3), "accuracy at 50, 200, 1000:", at(Z[k]))
    lines([("budget " + str(b), med(lambda b=b: RandomStrategy(budget=b, seed=0), 1)) for b in (0.05, 0.1, 0.5)], "active_random", "Random labelling: accuracy so far for three budgets", "accuracy so far", ylim=(0.6, 1.0), loc="lower right")

    th = np.linspace(0.5, 0.99, 25)
    Zf = sweep(lambda t: FixedUncertainty(theta=float(t), seed=0), th, 0)
    surface(Zf[:, 19:], th, "active_fixed", "A FIXED THRESHOLD IGNORES THE BUDGET", "labelling rate so far of FixedUncertainty for 25 values of theta, on 1000 trades", "theta", "labelling rate so far")
    for t in (0.5, 0.7, 0.99):
        k = int(np.argmin(np.abs(th - t)))
        print("fixed theta", round(float(th[k]), 3), "rate at 50, 200, 1000:", at(Zf[k]))
    lines([("theta " + str(t), med(lambda t=t: FixedUncertainty(theta=t, seed=0), 0)) for t in (0.6, 0.8, 0.95)], "active_fixed", "Fixed threshold: labelling rate so far for three thresholds", "labelling rate so far", hline=0.1)

    ss = np.linspace(0.0005, 0.05, 25)
    Zv = sweep(lambda s: VariableUncertainty(budget=0.1, s=float(s), seed=0), ss, 0)
    surface(Zv[:, 19:], ss, "active_variable", "THE VARIABLE THRESHOLD FINDS THE BUDGET", "labelling rate so far of VariableUncertainty with budget 0.1 for 25 values of the step s, on 1000 trades", "step s", "labelling rate so far")
    for s in (0.0005, 0.01, 0.05):
        k = int(np.argmin(np.abs(ss - s)))
        print("variable s", round(float(ss[k]), 4), "rate at 50, 200, 1000:", at(Zv[k]))
    lines([("s " + str(s), med(lambda s=s: VariableUncertainty(budget=0.1, s=s, seed=0), 0)) for s in (0.001, 0.01, 0.05)], "active_variable", "Variable threshold: labelling rate so far for three steps", "labelling rate so far", hline=0.1)

    de = np.linspace(0.001, 1.0, 25)
    Zr = sweep(lambda d: VariableRandomizedUncertainty(budget=0.1, s=0.01, delta=float(d), seed=0), de, 0)
    surface(Zr[:, 19:], de, "active_randomized", "A RANDOM THRESHOLD STILL KEEPS THE BUDGET", "labelling rate so far of VariableRandomizedUncertainty with budget 0.1 for 25 values of delta, on 1000 trades", "delta", "labelling rate so far")
    for d in (0.001, 0.05, 1.0):
        k = int(np.argmin(np.abs(de - d)))
        print("randomized delta", round(float(de[k]), 3), "rate at 50, 200, 1000:", at(Zr[k]))
    lines([("delta " + str(d), med(lambda d=d: VariableRandomizedUncertainty(budget=0.1, s=0.01, delta=d, seed=0), 0)) for d in (0.01, 0.05, 1.0)], "active_randomized", "Randomised threshold: labelling rate so far for three values of delta", "labelling rate so far", hline=0.1)

    bb = np.linspace(0.02, 0.5, 25)
    Zs = sweep(lambda b: VariableUncertainty(budget=float(b), s=0.01, seed=0), bb, 0)
    surface(Zs[:, 19:], bb, "active_strategy", "ONE BUDGET, ONE LABELLING RATE", "labelling rate so far of VariableUncertainty for 25 budgets, on 1000 trades", "budget", "labelling rate so far")
    for b in (0.02, 0.1, 0.5):
        k = int(np.argmin(np.abs(bb - b)))
        print("strategy budget", round(float(bb[k]), 3), "rate at 50, 200, 1000:", at(Zs[k]))
    lines([("RandomStrategy", med(lambda: RandomStrategy(budget=0.1, seed=0), 0)), ("FixedUncertainty 0.7", med(lambda: FixedUncertainty(theta=0.7, seed=0), 0)), ("VariableUncertainty", med(lambda: VariableUncertainty(budget=0.1, s=0.01, seed=0), 0)), ("VariableRandomized", med(lambda: VariableRandomizedUncertainty(budget=0.1, s=0.01, delta=0.05, seed=0), 0))], "active_strategy", "The four strategies on the same 1000 trades, budget 0.1", "labelling rate so far", hline=0.1)


if __name__ == "__main__":
    main()
