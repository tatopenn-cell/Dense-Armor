"""The figures of the Fit pages (3D at the start and finance figure), drawn with the library itself.

A desk has a table of past orders. The first page fits a linear model of the slippage on 200 orders (size and depth,
slippage = 2 size - depth plus a noise of 0.1) and scores it on 100 orders it has not seen. The second page fits a decision
tree on a table where the fill is good when the size and the depth are on opposite sides of zero, and scores it on 400
fresh orders. The third page teaches the online metric learners OASIS (triplets), LEGO and POLA (pairs) one triplet or pair at a time, on
orders that are good or bad fills, and scores the distance on 300 fresh triplets: the share in which an order is closer to
one of its own kind than to one of the other kind. Median of 5 seeds.

Run from the repo root:
    python docs/assets/fit/make_fit_figures.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from dense_armor.core.online.linear.glm import LinearRegression
from dense_armor.core.online.learn.metric_learning import LEGO, OASIS, POLA
from dense_armor.core.online.optim.sgd import SGD
from dense_armor.core.online.tree.batch import BatchDecisionTreeClassifier

HERE = Path(__file__).resolve().parent / "pages"
HERE.mkdir(exist_ok=True)
BG, PANEL, GRID = "#0b1118", "#111a24", "#26364a"
INK, MUTED = "#e8fbff", "#8fa3b8"
CYAN, ORANGE, PINK, GREEN = "#3ec6e8", "#ffa21f", "#ff2d87", "#46d37f"
CMAP = LinearSegmentedColormap.from_list("s", [CYAN, "#6a5acd", PINK])
SEEDS = 5
EPOCHS = 30


def orders(seed, n):
    r = np.random.default_rng(seed)
    A = r.normal(size=(n, 2))
    X = [{"size": float(a), "depth": float(b)} for a, b in A]
    return X, A


def reg_curve(lr, seed=0):
    X, A = orders(seed, 300)
    r = np.random.default_rng(seed + 100)
    y = list(2 * A[:, 0] - A[:, 1] + r.normal(0, 0.1, size=300))
    m = LinearRegression(optimizer=SGD(lr=float(lr)))
    out = np.zeros(EPOCHS)
    for e in range(EPOCHS):
        m.fit(X[:200], y[:200], epochs=1, shuffle=True, seed=e)
        err = [(m.predict_one(x) - v) ** 2 for x, v in zip(X[200:], y[200:])]
        out[e] = np.sqrt(np.mean(err))
    return out


def tree_acc(depth, n, seed=0):
    X, A = orders(seed, n + 400)
    y = [int((a > 0) != (b > 0)) for a, b in A]
    m = BatchDecisionTreeClassifier(max_depth=int(depth)).fit(X[:n], y[:n])
    return float(np.mean([m.predict_one(x) == v for x, v in zip(X[n:], y[n:])]))


def metric_data(seed, n_train, n_test=300):
    """Good and bad fills: two informative features (size, depth) and two that say nothing."""
    r = np.random.default_rng(seed)
    n = 2 * (n_train + n_test)
    y = r.integers(2, size=n)
    X = np.column_stack([r.normal(1.5 * y, 1.0), r.normal(-1.5 * y, 1.0), r.normal(size=n) * 3, r.normal(size=n) * 3])
    rows = [{"size": float(a), "depth": float(b), "noise1": float(c), "noise2": float(d)} for a, b, c, d in X]
    return rows, y, r


def triplets_from(rows, y, r, n):
    idx = np.arange(len(rows))
    out = []
    while len(out) < n:
        i = int(r.integers(len(rows)))
        pos = idx[(y == y[i]) & (idx != i)]
        neg = idx[y != y[i]]
        out.append((rows[i], rows[int(r.choice(pos))], rows[int(r.choice(neg))]))
    return out


def metric_curve(kind, param, seed=0, n_train=300, points=15):
    """Share of fresh triplets that the learned distance gets right after each batch of learning."""
    rows, y, r = metric_data(seed, n_train)
    train = triplets_from(rows[: 2 * n_train], y[: 2 * n_train], r, n_train)
    test = triplets_from(rows[2 * n_train:], y[2 * n_train:], r, 300)
    if kind == "oasis":
        m = OASIS(C=float(param))
    elif kind == "lego":
        m = LEGO(eta=float(param))
    else:
        m = POLA()
    steps = np.linspace(0, n_train, points + 1).astype(int)
    out = np.zeros(points + 1)
    for s in range(points + 1):
        if s:
            a, b = steps[s - 1], steps[s]
            batch = train[a:b]
            for q, pos, neg in batch:
                if kind == "oasis":
                    m.learn_triplet(q, pos, neg)
                elif kind == "lego":
                    m.learn_pair(q, pos, 0.05)
                    m.learn_pair(q, neg, 4.0)
                else:
                    m.learn_pair(q, pos, 1)
                    m.learn_pair(q, neg, -1)
        out[s] = np.mean([m.distance(q, pos) < m.distance(q, neg) for q, pos, neg in test])
    return out


def euclid_curve(seed=0, n_train=300, points=15):
    rows, y, r = metric_data(seed, n_train)
    test = triplets_from(rows[2 * n_train:], y[2 * n_train:], r, 300)
    keys = ["size", "depth", "noise1", "noise2"]
    d = lambda a, b: sum((a[k] - b[k]) ** 2 for k in keys)
    return np.full(points + 1, float(np.mean([d(q, pos) < d(q, neg) for q, pos, neg in test])))


def med(fn, *a, n=SEEDS):
    return np.median([fn(*a, seed=k) for k in range(n)], axis=0)


def surface(Z, xs, ys, name, title, subtitle, xl, yl, zl, ticks=None, zmax=None):
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


def lines(curves, xs, name, title, xl, yl, ylim=None, loc="upper right"):
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
        ax.plot(xs, v, color=col, lw=2.0, label=lab)
    if ylim:
        ax.set_ylim(*ylim)
    ax.legend(facecolor=PANEL, edgecolor=GRID, labelcolor=INK, fontsize=9, loc=loc)
    fig.tight_layout()
    fig.savefig(HERE / f"{name}_finance.png", dpi=150, facecolor=BG)
    plt.close(fig)


def main():
    eps = np.arange(1, EPOCHS + 1)
    lrs = np.linspace(0.001, 0.1, 25)
    Z = np.array([med(lambda seed, v=v: reg_curve(v, seed)) for v in lrs])
    surface(Z, eps, lrs, "fit_estimators", "A SMALL LEARNING RATE NEEDS MORE PASSES", "held-out error of LinearRegression after each pass of fit over 200 orders, for 25 learning rates", "passes (epochs)", "learning rate", "held-out RMSE", zmax=2.5)
    for v in (0.001, 0.005, 0.02, 0.1):
        k = int(np.argmin(np.abs(lrs - v)))
        print("fit lr", round(float(lrs[k]), 3), "held-out RMSE after 1, 5, 30 passes:", [round(float(Z[k][p - 1]), 3) for p in (1, 5, 30)])
    lines([("lr " + str(v), med(lambda seed, v=v: reg_curve(v, seed))) for v in (0.005, 0.02, 0.1)], eps, "fit_estimators", "Held-out error after each pass of fit, three learning rates", "passes (epochs)", "held-out RMSE", ylim=(0, 2.5))

    ns = np.linspace(20, 800, 25).astype(int)
    ds = np.arange(1, 9)
    Zt = np.array([[med(lambda seed, d=d, n=n: tree_acc(d, n, seed)) for n in ns] for d in ds])
    surface(Zt, ns.astype(float), ds.astype(float), "fit_batch", "A GREEDY TREE NEEDS DEPTH AND ROWS FOR AN XOR", "held-out accuracy of BatchDecisionTreeClassifier on 400 fresh orders, for 8 depths and 25 table sizes", "orders in the table", "max_depth", "held-out accuracy", ticks=[1, 2, 4, 6, 8], zmax=1.0)
    for d, n in ((1, 100), (1, 800), (2, 100), (2, 800), (4, 100), (4, 800), (8, 100), (8, 800)):
        i, j = int(d - 1), int(np.argmin(np.abs(ns - n)))
        print("tree depth", d, "orders", int(ns[j]), "accuracy", round(float(Zt[i][j]), 3))
    lines([("max_depth " + str(d), Zt[d - 1]) for d in (1, 2, 4)], ns, "fit_batch", "Held-out accuracy of the tree against the size of the table", "orders in the table", "held-out accuracy", ylim=(0, 1.0), loc="lower right")

    cs = np.geomspace(0.001, 1.0, 25)
    steps = np.linspace(0, 300, 16)
    Zo = np.array([med(lambda seed, c=c: metric_curve("oasis", c, seed)) for c in cs])
    surface(Zo, steps, np.log10(cs), "fit_metric", "A DISTANCE LEARNED A TRIPLET AT A TIME", "share of 300 fresh triplets that OASIS gets right after 0 to 300 triplets learned, for 25 values of C", "triplets learned", "log10 of C", "triplets right", ticks=[-3, -2, -1, 0], zmax=1.0)
    for c in (0.001, 0.01, 0.1, 1.0):
        k = int(np.argmin(np.abs(cs - c)))
        print("oasis C", round(float(cs[k]), 4), "triplets right after 0, 75, 150, 300 learned:", [round(float(Zo[k][i]), 3) for i in (0, 5, 8, 15)])
    base = float(med(lambda seed: euclid_curve(seed))[0])
    print("euclidean triplets right", round(base, 3))
    curves = [("OASIS C=0.1", med(lambda seed: metric_curve("oasis", 0.1, seed))), ("LEGO eta=0.1", med(lambda seed: metric_curve("lego", 0.1, seed))), ("POLA", med(lambda seed: metric_curve("pola", 0, seed))), ("Euclidean distance", med(lambda seed: euclid_curve(seed)))]
    for lab, v in curves:
        print(lab, "triplets right after 0, 75, 150, 300 learned:", [round(float(v[i]), 3) for i in (0, 5, 8, 15)])
    lines(curves, steps, "fit_metric", "Triplets right against the learning, OASIS, LEGO, POLA and the plain distance", "triplets learned", "triplets right", ylim=(0, 1.0), loc="lower right")


if __name__ == "__main__":
    main()
