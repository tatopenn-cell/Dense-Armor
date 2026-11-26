"""The figures of the Recommenders pages (3D at the start and finance figure), drawn with the library itself.

A bank has 20 desks and 15 execution algorithms. A desk rates an algorithm with a score (the fill quality, around 3.5),
and the scores come from a planted structure of rank 3 plus a noise of 0.1: each desk and each algorithm has three hidden
traits (for example how much it trades illiquid names, how urgent it is, how large the orders are), and the score is the
mean plus the product of the traits. 60 % of the pairs are given to the model, one pass per epoch, in the same order, and
the score of a page is the root mean squared error on the 40 % of the pairs that the model has never seen (median of 5
seeds). The ranking page scores the top 3 algorithms of each desk. The last page uses a stream of random points.

Run from the repo root:
    python docs/assets/reco/make_reco_style_figures.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from dense_armor.core.online.reco import Baseline, BiasedMF, FunkMF
from dense_armor.core.signals.stats.skyline import Skyline

HERE = Path(__file__).resolve().parent / "pages"
HERE.mkdir(exist_ok=True)
BG, PANEL, GRID = "#0b1118", "#111a24", "#26364a"
INK, MUTED = "#e8fbff", "#8fa3b8"
CYAN, ORANGE, PINK, GREEN = "#3ec6e8", "#ffa21f", "#ff2d87", "#46d37f"
CMAP = LinearSegmentedColormap.from_list("s", [CYAN, "#6a5acd", PINK])
EP = 100
SEEDS = 5
NU, NI, K = 20, 15, 3


def world(seed, frac=0.6):
    rng = np.random.default_rng(seed)
    P = rng.normal(size=(NU, K))
    Q = rng.normal(size=(NI, K))
    R = 3.5 + P @ Q.T + rng.normal(0, 0.1, size=(NU, NI))
    pairs = [(u, i) for u in range(NU) for i in range(NI)]
    rng.shuffle(pairs)
    cut = int(frac * len(pairs))
    return R, pairs[:cut], pairs[cut:]


def rmse_curve(make, seed=0):
    R, tr, te = world(seed)
    m = make()
    out = np.zeros(EP)
    for ep in range(EP):
        for u, i in tr:
            m.learn_one(u, i, float(R[u, i]))
        out[ep] = np.sqrt(np.mean([(m.predict_one(u, i) - R[u, i]) ** 2 for u, i in te]))
    return out


def prec_curve(frac, seed=0, top=3):
    R, tr, te = world(seed, frac)
    m = BiasedMF(k=3, lr=0.02, l2=0.001, seed=0)
    out = np.zeros(EP)
    items = list(range(NI))
    for ep in range(EP):
        for u, i in tr:
            m.learn_one(u, i, float(R[u, i]))
        hits = 0
        for u in range(NU):
            true = set(np.argsort(-R[u])[:top].tolist())
            hits += len(true & set(m.rank(u, items, top_k=top)))
        out[ep] = hits / (NU * top)
    return out


def med(fn, *a):
    return np.median([fn(*a, seed=k) for k in range(SEEDS)], axis=0)


def sky_curve(w, seed=0, n=EP * 10):
    rng = np.random.default_rng(seed)
    s = Skyline(window=int(w))
    out = np.zeros(n)
    for t in range(n):
        s.update({"x": float(rng.random()), "y": float(rng.random())})
        out[t] = len(s.skyline())
    return out


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
    ax.plot_surface(X, Y, Zc, facecolors=CMAP(np.clip(Zc / top, 0, 1)), rstride=1, cstride=2, linewidth=0, antialiased=False, alpha=0.93)
    ax.set_zlim(0, top)
    fig.savefig(HERE / f"{name}_3d.png", dpi=150, facecolor=BG)
    plt.close(fig)


def lines(curves, name, title, xl, yl, ylim=None, loc="upper right"):
    fig, ax = plt.subplots(figsize=(9, 3.6), facecolor=BG)
    ax.set_facecolor(PANEL)
    ax.set_xlabel(xl, color=CYAN)
    ax.set_ylabel(yl, color=CYAN)
    ax.tick_params(colors=MUTED)
    for s in ax.spines.values():
        s.set_color(GRID)
    ax.grid(color=GRID, lw=0.5)
    ax.set_title(title, color=INK, loc="left", fontsize=13, fontweight="bold")
    for (lab, v), col in zip(curves, (CYAN, ORANGE, PINK, GREEN)):
        ax.plot(np.arange(1, len(v) + 1), v, color=col, lw=2.0, label=lab)
    if ylim:
        ax.set_ylim(*ylim)
    ax.legend(facecolor=PANEL, edgecolor=GRID, labelcolor=INK, fontsize=9, loc=loc)
    fig.tight_layout()
    fig.savefig(HERE / f"{name}_finance.png", dpi=150, facecolor=BG)
    plt.close(fig)


def at(v, pts=(1, 10, 50, 100)):
    return [round(float(v[p - 1]), 3) for p in pts]


def main():
    eps = np.arange(1, EP + 1)
    lrs = np.linspace(0.005, 0.2, 25)
    Z = np.array([med(rmse_curve, (lambda v=v: Baseline(lr=float(v), l2=0.02, seed=0))) for v in lrs])
    surface(Z, eps, lrs, "reco_baseline", "THE BIASES ALONE CANNOT SEE WHO LIKES WHAT", "held-out error of Baseline over 100 epochs on 20 desks and 15 algorithms, for 25 learning rates", "epochs", "learning rate", "held-out RMSE", zmax=2.5)
    for v in (0.005, 0.05, 0.2):
        k = int(np.argmin(np.abs(lrs - v)))
        print("baseline lr", round(float(lrs[k]), 3), "held-out RMSE at epochs 1, 10, 50, 100:", at(Z[k]))
    lines([("Baseline lr 0.05", med(rmse_curve, lambda: Baseline(lr=0.05, l2=0.02, seed=0))), ("BiasedMF k=3", med(rmse_curve, lambda: BiasedMF(k=3, lr=0.02, l2=0.001, seed=0)))], "reco_baseline", "Held-out error: the biases against the factors", "epochs", "held-out RMSE", ylim=(0, 2.5))
    print("baseline curve", at(med(rmse_curve, lambda: Baseline(lr=0.05, l2=0.02, seed=0))), "biased", at(med(rmse_curve, lambda: BiasedMF(k=3, lr=0.02, l2=0.001, seed=0))))

    ks = np.arange(1, 13)
    Zb = np.array([med(rmse_curve, (lambda v=v: BiasedMF(k=int(v), lr=0.02, l2=0.001, seed=0))) for v in ks])
    surface(Zb, eps, ks.astype(float), "reco_biased_mf", "THE RIGHT NUMBER OF FACTORS IS THE HIDDEN RANK", "held-out error of BiasedMF over 100 epochs, for k from 1 to 12, when the true structure has rank 3", "epochs", "k", "held-out RMSE", zmax=2.5, ticks=[1, 3, 6, 9, 12])
    for v in (1, 3, 12):
        k = int(np.argmin(np.abs(ks - v)))
        print("biased k", int(ks[k]), at(Zb[k]))
    lines([("BiasedMF k=1", med(rmse_curve, lambda: BiasedMF(k=1, lr=0.02, l2=0.001, seed=0))), ("BiasedMF k=3", med(rmse_curve, lambda: BiasedMF(k=3, lr=0.02, l2=0.001, seed=0))), ("BiasedMF k=12", med(rmse_curve, lambda: BiasedMF(k=12, lr=0.02, l2=0.001, seed=0)))], "reco_biased_mf", "Held-out error for three values of k", "epochs", "held-out RMSE", ylim=(0, 2.5))

    l2s = np.linspace(0.0, 0.5, 25)
    Zf = np.array([med(rmse_curve, (lambda v=v: FunkMF(k=3, lr=0.02, l2=float(v), init_scale=0.3, seed=0))) for v in l2s])
    surface(Zf, eps, l2s, "reco_funk_mf", "A STRONG PENALTY HOLDS THE FACTORS BACK", "held-out error of FunkMF over 100 epochs, for 25 values of the L2 penalty", "epochs", "L2 penalty", "held-out RMSE", zmax=2.5)
    for v in (0.0, 0.1, 0.5):
        k = int(np.argmin(np.abs(l2s - v)))
        print("funk l2", round(float(l2s[k]), 3), at(Zf[k]))
    lines([("FunkMF l2 0.001", med(rmse_curve, lambda: FunkMF(k=3, lr=0.02, l2=0.001, init_scale=0.3, seed=0))), ("FunkMF l2 0.3", med(rmse_curve, lambda: FunkMF(k=3, lr=0.02, l2=0.3, init_scale=0.3, seed=0))), ("BiasedMF l2 0.001", med(rmse_curve, lambda: BiasedMF(k=3, lr=0.02, l2=0.001, seed=0)))], "reco_funk_mf", "Held-out error: factors without biases against with biases", "epochs", "held-out RMSE", ylim=(0, 2.5))

    fr = np.linspace(0.2, 0.9, 8)
    Zr = np.array([med(prec_curve, float(v)) for v in fr])
    surface(Zr, eps, fr, "reco_rank", "MORE RATINGS SEEN, A BETTER TOP 3", "share of the true top 3 algorithms of each desk found by rank, for 8 shares of the ratings given to the model", "epochs", "share of ratings seen", "precision at 3", zmax=1.0)
    for v in (0.2, 0.6, 0.9):
        k = int(np.argmin(np.abs(fr - v)))
        print("rank frac", round(float(fr[k]), 2), at(Zr[k]))
    lines([("30 % of the ratings", med(prec_curve, 0.3)), ("60 % of the ratings", med(prec_curve, 0.6)), ("90 % of the ratings", med(prec_curve, 0.9))], "reco_rank", "Precision at 3 of rank on three shares of the ratings", "epochs", "precision at 3", ylim=(0, 1.0), loc="lower right")

    ws = np.array([5, 10, 20, 30, 40, 50, 60])
    pts = np.arange(1, EP * 10 + 1)
    Zs = np.array([np.median([sky_curve(w, seed=k) for k in range(SEEDS)], axis=0) for w in ws])
    surface(Zs, pts, ws.astype(float), "reco_windows", "A BIGGER WINDOW, A BIGGER SKYLINE", "size of the skyline of a sliding window of random points, for window sizes from 5 to 60", "points seen", "window", "skyline size", ticks=[5, 20, 40, 60])
    for w in (5, 20, 60):
        k = int(np.argmin(np.abs(ws - w)))
        print("skyline w", int(ws[k]), "size at 10, 100, 500, 1000:", [int(Zs[k][p - 1]) for p in (10, 100, 500, 1000)], "mean after the window is full", round(float(Zs[k][ws[k]:].mean()), 2))
    lines([("window 10", Zs[1]), ("window 20", Zs[2]), ("window 50", Zs[5])], "reco_windows", "Skyline size for three window sizes", "points seen", "skyline size")


if __name__ == "__main__":
    main()
