"""Figures for the Mondrian page, drawn with the library itself.

Generates in the same folder:
- mondrian_partition.png   one Mondrian tree after 8, 40, 200 and 1000 points
- mondrian_classes.png     class probability of a forest: it fades to 50 % away from the data
- mondrian_surface.png     the standard deviation of a forest as a 3D surface (hero)

Run from the repo root:
    python docs/assets/tree/make_mondrian.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from dense_armor.core.online.tree import (
    MondrianForestClassifier,
    MondrianForestRegressor,
)

HERE = Path(__file__).resolve().parent
BG, PANEL, GRID = "#0b1118", "#111a24", "#26364a"
INK, MUTED = "#e8fbff", "#8fa3b8"
CYAN, ORANGE, PINK = "#3ec6e8", "#ffa21f", "#ff2d87"
DPI = 150

plt.rcParams.update(
    {
        "figure.facecolor": BG,
        "axes.facecolor": PANEL,
        "axes.edgecolor": GRID,
        "axes.labelcolor": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "text.color": INK,
        "font.size": 10,
    }
)


def moons(rng, n):
    t = rng.uniform(0, np.pi, n)
    y = (rng.random(n) < 0.5).astype(int)
    x1 = np.where(y == 0, np.cos(t), 1 - np.cos(t)) + rng.normal(0, 0.09, n)
    x2 = np.where(y == 0, np.sin(t), 0.45 - np.sin(t)) + rng.normal(0, 0.09, n)
    return x1, x2, y


def walk(node, box, out):
    if node.feature is None:
        return
    lo1, hi1, lo2, hi2 = box
    c = node.threshold
    if node.feature == "a":
        out.append(((c, c), (lo2, hi2), node.depth))
        walk(node.left, (lo1, c, lo2, hi2), out)
        walk(node.right, (c, hi1, lo2, hi2), out)
    else:
        out.append(((lo1, hi1), (c, c), node.depth))
        walk(node.left, (lo1, hi1, lo2, c), out)
        walk(node.right, (lo1, hi1, c, hi2), out)


def partition():
    rng = np.random.default_rng(7)
    x1, x2, y = moons(rng, 1000)
    forest = MondrianForestClassifier(n_trees=1, seed=3)
    marks = (8, 40, 200, 1000)
    box = (-1.4, 2.4, -0.9, 1.4)
    fig, axs = plt.subplots(1, 4, figsize=(16, 4.6))
    fig.subplots_adjust(top=0.78, bottom=0.07, left=0.03, right=0.99, wspace=0.08)
    k = 0
    for i in range(1000):
        forest.learn_one({"a": float(x1[i]), "b": float(x2[i])}, int(y[i]))
        if i + 1 not in marks:
            continue
        ax = axs[k]
        k += 1
        segs = []
        walk(forest.trees_[0].root, box, segs)
        deepest = max([s[2] for s in segs] + [1])
        for xs, ys, d in segs:
            ax.plot(
                xs,
                ys,
                color=CYAN,
                lw=2.4 - 1.7 * d / deepest,
                alpha=0.95 - 0.5 * d / deepest,
            )
        for cls, col in ((0, CYAN), (1, ORANGE)):
            m = y[: i + 1] == cls
            ax.scatter(
                x1[: i + 1][m],
                x2[: i + 1][m],
                s=14,
                color=col,
                edgecolor="none",
                alpha=0.9,
                zorder=3,
            )
        ax.set_xlim(box[0], box[1])
        ax.set_ylim(box[2], box[3])
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color(GRID)
        ax.set_title(
            f"{i + 1} points   {forest.trees_[0].n_nodes} nodes",
            color=INK,
            fontsize=12,
            pad=8,
        )
    fig.text(
        0.03,
        0.93,
        "ONE MONDRIAN TREE GROWS WITH THE DATA",
        fontsize=20,
        fontweight="bold",
        color=INK,
    )
    fig.text(
        0.03,
        0.865,
        "every line is a split; thick lines are the first, random cuts, thin ones are the later, finer cuts",
        fontsize=11.5,
        color=CYAN,
    )
    fig.savefig(HERE / "mondrian_partition.png", dpi=DPI, facecolor=BG)
    plt.close(fig)


def classes():
    rng = np.random.default_rng(11)
    x1, x2, y = moons(rng, 800)
    f = MondrianForestClassifier(n_trees=25, seed=1)
    for i in range(800):
        f.learn_one({"a": float(x1[i]), "b": float(x2[i])}, int(y[i]))
    gx = np.linspace(-2.2, 3.2, 150)
    gy = np.linspace(-1.7, 2.2, 120)
    p = np.zeros((len(gy), len(gx)))
    for r, b in enumerate(gy):
        for c, a in enumerate(gx):
            pr = f.predict_proba_one({"a": float(a), "b": float(b)})
            p[r, c] = pr.get(1, 0.0) if pr else 0.5
    cmap = LinearSegmentedColormap.from_list("m", [CYAN, "#1b2a3a", ORANGE])
    fig, ax = plt.subplots(figsize=(10.5, 6.6))
    fig.subplots_adjust(top=0.82, bottom=0.08, left=0.07, right=0.93)
    im = ax.imshow(
        p,
        extent=(gx[0], gx[-1], gy[0], gy[-1]),
        origin="lower",
        cmap=cmap,
        vmin=0,
        vmax=1,
        aspect="auto",
    )
    ax.contour(gx, gy, p, levels=[0.5], colors=[INK], linewidths=1.2, alpha=0.8)
    ax.scatter(x1[y == 0], x2[y == 0], s=9, color=CYAN, edgecolor=BG, linewidth=0.3)
    ax.scatter(x1[y == 1], x2[y == 1], s=9, color=ORANGE, edgecolor=BG, linewidth=0.3)
    ax.set_xlabel("a")
    ax.set_ylabel("b")
    cb = fig.colorbar(im, ax=ax, pad=0.02)
    cb.set_label("probability of class 1", color=INK)
    cb.ax.yaxis.set_tick_params(color=MUTED)
    fig.text(
        0.07,
        0.93,
        "THE FOREST ADMITS WHAT IT DOES NOT KNOW",
        fontsize=19,
        fontweight="bold",
        color=INK,
    )
    fig.text(
        0.07,
        0.875,
        "25 trees, 800 points: sure near the data (cyan, orange), a flat 50 % far from it (dark)",
        fontsize=11,
        color=CYAN,
    )
    fig.savefig(HERE / "mondrian_classes.png", dpi=DPI, facecolor=BG)
    plt.close(fig)


def surface():
    rng = np.random.default_rng(5)
    f = MondrianForestRegressor(n_trees=60, seed=2)
    pts = []
    while len(pts) < 2000:
        a, b = rng.uniform(-1, 1, 2)
        if a * a + b * b <= 1.0:
            pts.append((a, b))
    for a, b in pts:
        f.learn_one(
            {"a": float(a), "b": float(b)},
            float(np.sin(3 * a) * np.cos(3 * b) + rng.normal(0, 0.05)),
        )
    g = np.linspace(-2.2, 2.2, 61)
    sd = np.zeros((len(g), len(g)))
    mu = np.zeros_like(sd)
    for r, b in enumerate(g):
        for c, a in enumerate(g):
            e = f.predict_one({"a": float(a), "b": float(b)}, return_estimate=True)
            sd[r, c] = e.std
            mu[r, c] = e.mean
    A, B = np.meshgrid(g, g)
    cmap = LinearSegmentedColormap.from_list("s", [CYAN, "#6a5acd", PINK])
    norm = plt.Normalize(sd.min(), sd.max())
    fig = plt.figure(figsize=(12, 8.2))
    ax = fig.add_subplot(111, projection="3d")
    fig.subplots_adjust(top=0.93, bottom=0.0, left=-0.05, right=1.05)
    ax.set_box_aspect((1, 1, 0.62), zoom=1.12)
    ax.set_facecolor(BG)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_pane_color((0.07, 0.10, 0.14, 1.0))
        axis._axinfo["grid"]["color"] = GRID
    ax.plot_surface(
        A,
        B,
        sd,
        facecolors=cmap(norm(sd)),
        rstride=1,
        cstride=1,
        linewidth=0,
        antialiased=True,
        alpha=0.92,
    )
    t = np.linspace(0, 2 * np.pi, 200)
    ax.plot(
        np.cos(t), np.sin(t), np.zeros_like(t) + sd.min(), color=INK, lw=1.6, alpha=0.9
    )
    pa = np.array(pts)
    ax.scatter(
        pa[:, 0],
        pa[:, 1],
        np.zeros(len(pa)) + sd.min(),
        s=5,
        color=ORANGE,
        alpha=0.7,
        depthshade=False,
    )
    ax.view_init(elev=27, azim=-58)
    ax.set_xlabel("a", color=CYAN, labelpad=2)
    ax.set_ylabel("b", color=CYAN, labelpad=2)
    ax.set_zlabel("standard deviation", color=CYAN, labelpad=6)
    ax.tick_params(colors=MUTED, labelsize=8)
    fig.text(
        0.05,
        0.945,
        "MONDRIAN FOREST: THE SHAPE OF ITS UNCERTAINTY",
        fontsize=21,
        fontweight="bold",
        color=INK,
    )
    fig.text(
        0.05,
        0.895,
        "trained only inside the unit disc (orange dots, white circle): low there, and it climbs",
        fontsize=11.5,
        color=CYAN,
    )
    fig.text(
        0.05,
        0.865,
        "to the spread of the labels on every side as the query leaves the data",
        fontsize=11.5,
        color=CYAN,
    )
    fig.savefig(HERE / "mondrian_surface.png", dpi=DPI, facecolor=BG)
    plt.close(fig)
    r = np.hypot(A, B)
    print(
        f"mean std inside r<0.8: {sd[r < 0.8].mean():.3f} | at r>1.6: {sd[r > 1.6].mean():.3f}"
    )


if __name__ == "__main__":
    partition()
    classes()
    surface()
