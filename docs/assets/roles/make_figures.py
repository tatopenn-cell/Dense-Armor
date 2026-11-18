import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from dense_armor.core.online.anomaly.hst import HalfSpaceTrees
from dense_armor.core.online.learn.online_classifiers import OnlineGaussianNB
from dense_armor.core.online.tree.hoeffding_regressor import HoeffdingTreeRegressor

BG = "#0d131a"
CMAP = LinearSegmentedColormap.from_list(
    "roles", ["#00b4d8", "#4cc9f0", "#7209b7", "#f72585", "#ff9e00"]
)
OUT = Path(__file__).resolve().parent


def stream(n, seed):
    rng = np.random.default_rng(seed)
    for _ in range(n):
        k = int(rng.integers(2))
        c = 1.4 if k else -1.4
        a, b = rng.normal(c, 0.7, 2)
        yield {"a": float(a), "b": float(b)}, k


def train():
    nb = OnlineGaussianNB()
    rt = HoeffdingTreeRegressor(grace_period=40)
    hst = HalfSpaceTrees(seed=0)
    for x, k in stream(9000, 0):
        nb.learn_one(x, k)
        rt.learn_one(x, float(np.sin(1.3 * x["a"])) + 0.3 * x["b"])
        hst.learn_one(x)
    return nb, rt, hst


def surface(fn, g):
    z = np.zeros((len(g), len(g)))
    for i, b in enumerate(g):
        for j, a in enumerate(g):
            z[i, j] = fn({"a": float(a), "b": float(b)})
    return z


def panel(fig, rect, g, z, zlabel, pts, zpts):
    ax = fig.add_axes(rect, projection="3d")
    ax.set_facecolor(BG)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.fill = False
        axis.pane.set_edgecolor("#1e293b")
        axis._axinfo["grid"]["color"] = (0.20, 0.30, 0.42, 0.45)
        axis._axinfo["grid"]["linewidth"] = 0.7
    a, b = np.meshgrid(g, g)
    ax.plot_surface(
        a,
        b,
        z,
        cmap=CMAP,
        alpha=0.78,
        edgecolor="#48cae4",
        linewidth=0.25,
        rstride=1,
        cstride=1,
    )
    ax.scatter(
        pts[:, 0],
        pts[:, 1],
        zpts,
        color="#ffffff",
        s=9,
        alpha=0.9,
        edgecolors="#90e0ef",
        linewidth=0.3,
    )
    ax.set_xlabel("a", color="#90e0ef", fontsize=8, fontweight="bold")
    ax.set_ylabel("b", color="#90e0ef", fontsize=8, fontweight="bold")
    ax.set_zlabel(zlabel, color="#90e0ef", fontsize=8, fontweight="bold")
    ax.tick_params(colors="#64748b", labelsize=7)
    ax.view_init(elev=24, azim=-58)
    return ax


def overview():
    nb, rt, hst = train()
    g = np.linspace(-3.5, 3.5, 36)
    zc = surface(lambda x: nb.predict_proba_one(x).get(1, 0.0), g)
    zr = surface(lambda x: rt.predict_one(x), g)
    za = surface(lambda x: hst.score_one(x), g)
    pts = np.array([[x["a"], x["b"]] for x, _ in stream(120, 7)])
    fig = plt.figure(figsize=(13.5, 6.0), facecolor=BG)
    fig.text(
        0.5,
        0.94,
        "ONE LOOP, THREE ROLES",
        ha="center",
        va="center",
        fontsize=22,
        fontweight="bold",
        color="#e0fbfc",
    )
    fig.text(
        0.5,
        0.88,
        "The same dict goes in; each role answers with a different number",
        ha="center",
        va="center",
        fontsize=10.5,
        color="#4cc9f0",
    )
    cols = [
        (
            "01",
            "CLASSIFIER",
            "predict_proba_one(x)[1]",
            0.02,
            zc,
            "P(class 1)",
            lambda p: nb.predict_proba_one({"a": float(p[0]), "b": float(p[1])}).get(
                1, 0.0
            ),
        ),
        (
            "02",
            "REGRESSOR",
            "predict_one(x)",
            0.35,
            zr,
            "prediction",
            lambda p: rt.predict_one({"a": float(p[0]), "b": float(p[1])}),
        ),
        (
            "03",
            "ANOMALY DETECTOR",
            "score_one(x)",
            0.68,
            za,
            "score",
            lambda p: hst.score_one({"a": float(p[0]), "b": float(p[1])}),
        ),
    ]
    for num, title, call, x0, z, zl, fz in cols:
        fig.text(x0 + 0.04, 0.78, num, fontsize=13, fontweight="bold", color="#f72585")
        fig.text(
            x0 + 0.075, 0.78, title, fontsize=11, fontweight="bold", color="#ffffff"
        )
        fig.text(
            x0 + 0.04, 0.735, call, fontsize=9.5, color="#cbd5e1", family="monospace"
        )
        zp = np.array([fz(p) for p in pts])
        panel(fig, [x0, 0.0, 0.31, 0.74], g, z, zl, pts, zp)
    fig.savefig(OUT / "overview.png", dpi=200, facecolor=BG)


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("all", "overview"):
        overview()
