"""The figures of the Factorization machines pages (3D at the start and finance figure), drawn with the library itself.

A desk sends orders and wants to know if a fill will be good. An order has a venue (A or B), an algorithm (twap or pov)
and a time slot (open or close), each written as a one-hot feature. On the pair pages the fill is good when the venue
and the algorithm are a matching pair (A with twap, B with pov), that is a pure interaction: no single feature says
anything alone. On the higher-order page the fill is good when the three bits (venue, algorithm, slot) add up to an odd
number, a pure interaction of three. The score of each page is the accuracy so far (predict, then learn), as the
median of 5 seeds over 1500 orders.

Run from the repo root:
    python docs/assets/facto/make_facto_style_figures.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from dense_armor.core.online.facto import (
    FFMClassifier,
    FMClassifier,
    FwFMClassifier,
    HOFMClassifier,
)
from dense_armor.core.online.linear.glm import LogisticRegression
from dense_armor.core.online.optim import SGD

HERE = Path(__file__).resolve().parent / "pages"
HERE.mkdir(exist_ok=True)
BG, PANEL, GRID = "#0b1118", "#111a24", "#26364a"
INK, MUTED = "#e8fbff", "#8fa3b8"
CYAN, ORANGE, PINK, GREEN = "#3ec6e8", "#ffa21f", "#ff2d87", "#46d37f"
CMAP = LinearSegmentedColormap.from_list("s", [CYAN, "#6a5acd", PINK])
T = 1500
SEEDS = 5
FIELDS = {"venue=A": "venue", "venue=B": "venue", "algo=twap": "algo", "algo=pov": "algo", "slot=open": "slot", "slot=close": "slot"}


def stream(seed, triple, n=T):
    r = np.random.default_rng(seed)
    for _ in range(n):
        v, a, s = (int(r.integers(2)) for _ in range(3))
        x = {f"venue={'AB'[v]}": 1.0, f"algo={['twap', 'pov'][a]}": 1.0, f"slot={['open', 'close'][s]}": 1.0}
        yield x, ((v ^ a ^ s) if triple else (v ^ a))


def run(make, triple=False, seed=0):
    model = make()
    acc = np.zeros(T)
    ok = 0
    for t, (x, y) in enumerate(stream(seed, triple)):
        p = model.predict_proba_one(x)
        ok += int((max(p, key=p.get) if p else 0) == y)
        model.learn_one(x, y)
        acc[t] = ok / (t + 1)
    return acc


def med(make, triple=False):
    return np.median([run(make, triple, seed=k) for k in range(SEEDS)], axis=0)


def surface(Z, ys, name, title, subtitle, yl, zl, ticks=None):
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
    ax.set_xlabel("orders seen", color=CYAN, labelpad=8)
    ax.set_ylabel(yl, color=CYAN, labelpad=8)
    ax.set_zlabel(zl, color=CYAN, labelpad=6)
    ax.tick_params(colors=MUTED, labelsize=8)
    if ticks is not None:
        ax.set_yticks(ticks)
    fig.text(0.05, 0.955, title, fontsize=21, fontweight="bold", color=INK)
    fig.text(0.05, 0.915, subtitle, fontsize=11.5, color=CYAN)
    ax.plot_surface(X, Y, Z, facecolors=CMAP(np.clip((Z - 0.4) / 0.6, 0, 1)), rstride=1, cstride=3, linewidth=0, antialiased=False, alpha=0.93)
    ax.set_zlim(0.4, 1.0)
    fig.savefig(HERE / f"{name}_3d.png", dpi=150, facecolor=BG)
    plt.close(fig)


def lines(curves, name, title):
    fig, ax = plt.subplots(figsize=(9, 3.6), facecolor=BG)
    ax.set_facecolor(PANEL)
    ax.set_xlabel("orders seen", color=CYAN)
    ax.set_ylabel("accuracy so far", color=CYAN)
    ax.tick_params(colors=MUTED)
    for s in ax.spines.values():
        s.set_color(GRID)
    ax.grid(color=GRID, lw=0.5)
    ax.set_title(title, color=INK, loc="left", fontsize=13, fontweight="bold")
    for (lab, v), col in zip(curves, (CYAN, ORANGE, PINK, GREEN)):
        ax.plot(np.arange(1, len(v) + 1), v, color=col, lw=2.0, label=lab)
    ax.set_ylim(0.4, 1.0)
    ax.legend(facecolor=PANEL, edgecolor=GRID, labelcolor=INK, fontsize=9, loc="lower right")
    fig.tight_layout()
    fig.savefig(HERE / f"{name}_finance.png", dpi=150, facecolor=BG)
    plt.close(fig)


def at(v, days=(50, 200, 1500)):
    return [round(float(v[d - 1]), 3) for d in days]


def sweep(make_for, params, triple=False):
    return np.array([med(lambda v=v: make_for(v), triple) for v in params])


def fm(lr=0.2, k=4, init=0.3):
    return FMClassifier(optimizer=SGD(lr=float(lr)), k=int(k), init_scale=init, seed=0)


def main():
    lrs = np.linspace(0.01, 0.4, 25)
    Z = sweep(lambda v: fm(lr=v), lrs)
    surface(Z[:, 19:], lrs, "facto_fm", "A PLAIN FM LEARNS A PURE INTERACTION", "accuracy so far of FMClassifier on 1500 orders where only the pair venue + algorithm decides, for 25 learning rates", "learning rate", "accuracy so far")
    for v in (0.01, 0.2, 0.4):
        k = int(np.argmin(np.abs(lrs - v)))
        print("fm lr", round(float(lrs[k]), 3), "accuracy at 50, 200, 1500:", at(Z[k]))
    lines([("LogisticRegression", med(lambda: LogisticRegression(optimizer=SGD(lr=0.05)))), ("FMClassifier, init_scale 0.3", med(lambda: fm())), ("FMClassifier, init_scale 0.01", med(lambda: fm(lr=0.05, init=0.01)))], "facto_fm", "Pair of venue and algorithm: a linear model never gets it")
    print("lr", at(med(lambda: LogisticRegression(optimizer=SGD(lr=0.05)))), "fm init.01", at(med(lambda: fm(lr=0.05, init=0.01))), "fm", at(med(lambda: fm())))

    lrf = np.linspace(0.02, 1.0, 25)
    Zf = sweep(lambda v: FFMClassifier(fields=FIELDS, k=2, lr=float(v), seed=0), lrf)
    surface(Zf[:, 19:], lrf, "facto_ffm", "A FIELD-AWARE FM, WITHOUT LINEAR TERMS", "accuracy so far of FFMClassifier on 1500 orders where only the pair venue + algorithm decides, for 25 learning rates", "learning rate", "accuracy so far")
    for v in (0.02, 0.1, 1.0):
        k = int(np.argmin(np.abs(lrf - v)))
        print("ffm lr", round(float(lrf[k]), 3), at(Zf[k]))
    lines([("FFM lr 0.1", med(lambda: FFMClassifier(fields=FIELDS, k=2, lr=0.1, seed=0))), ("FFM lr 0.5", med(lambda: FFMClassifier(fields=FIELDS, k=2, lr=0.5, seed=0))), ("FFM lr 1.0", med(lambda: FFMClassifier(fields=FIELDS, k=2, lr=1.0, seed=0)))], "facto_ffm", "Field-aware FM on the same orders, three learning rates")

    ks = np.arange(1, 13)
    Zw = sweep(lambda v: FwFMClassifier(optimizer=SGD(lr=0.2), fields=FIELDS, k=int(v), init_scale=0.3, seed=0), ks)
    surface(Zw[:, 19:], ks.astype(float), "facto_fwfm", "A WEIGHT PER PAIR OF FIELDS", "accuracy so far of FwFMClassifier on 1500 orders where only the pair venue + algorithm decides, for k from 1 to 12", "k", "accuracy so far", ticks=[1, 4, 8, 12])
    for v in (1, 4, 12):
        k = int(np.argmin(np.abs(ks - v)))
        print("fwfm k", int(ks[k]), at(Zw[k]))
    lines([("FwFM k=2", med(lambda: FwFMClassifier(optimizer=SGD(lr=0.2), fields=FIELDS, k=2, init_scale=0.3, seed=0))), ("FM k=4", med(lambda: fm())), ("FFM lr 0.1", med(lambda: FFMClassifier(fields=FIELDS, k=2, lr=0.1, seed=0)))], "facto_fwfm", "Field-weighted FM against FM and FFM on the same orders")

    ms = np.arange(2, 9)
    Zh = sweep(lambda v: HOFMClassifier(optimizer=SGD(lr=0.2), m=int(v), k=4, init_scale=0.3, seed=0), ms, triple=True)
    surface(Zh[:, 19:], ms.astype(float), "facto_hofm", "ONLY A DEGREE OF THREE SEES A TRIPLE", "accuracy so far of HOFMClassifier on 1500 orders where venue + algorithm + slot decide together, for the degree m from 2 to 8", "degree m", "accuracy so far", ticks=[2, 3, 4, 6, 8])
    for v in (2, 3, 8):
        k = int(np.argmin(np.abs(ms - v)))
        print("hofm m", int(ms[k]), at(Zh[k]))
    lines([("FM (degree 2)", med(lambda: fm(), triple=True)), ("HOFM m=3", med(lambda: HOFMClassifier(optimizer=SGD(lr=0.2), m=3, k=4, init_scale=0.3, seed=0), triple=True)), ("LogisticRegression", med(lambda: LogisticRegression(optimizer=SGD(lr=0.05)), triple=True))], "facto_hofm", "Venue, algorithm and slot together: degree 3 against degree 2")
    print("triple fm", at(med(lambda: fm(), triple=True)), "hofm3", at(med(lambda: HOFMClassifier(optimizer=SGD(lr=0.2), m=3, k=4, init_scale=0.3, seed=0), triple=True)))


if __name__ == "__main__":
    main()
