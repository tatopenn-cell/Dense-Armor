"""Dark figures for the Hoeffding, Anytime, Adaptive and stochastic gradient tree pages.

Generates in the same folder, with the library itself:
- hoeffding_story.png            the split, the Hoeffding bound and the accuracy of one tree
- hoeffding_regressor_surface.png  a Hoeffding regression tree against the true torque (3D)
- efdt_replace.png               plain and Anytime tree when the useful feature changes
- adaptive_recovery.png          plain and adaptive tree after a sudden change, with the signals
- sgt_3d.png                     a stochastic gradient tree against the true torque (3D)

Run from the repo root:
    python docs/assets/tree/make_trees_figures.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from dense_armor.core.online.drift.adwin import ADWIN
from dense_armor.core.online.tree import (
    HoeffdingAdaptiveTreeClassifier,
    HoeffdingAnytimeTreeClassifier,
    HoeffdingTreeClassifier,
    HoeffdingTreeRegressor,
    SGTRegressor,
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
        "grid.color": GRID,
        "grid.alpha": 0.6,
        "font.size": 10,
    }
)


def heading(fig, title, sub, y=0.95):
    fig.text(0.04, y, title, fontsize=20, fontweight="bold", color=INK)
    fig.text(0.04, y - 0.062, sub, fontsize=11.5, color=CYAN)


def rolling(hits, w):
    c = np.cumsum(np.insert(np.asarray(hits, dtype=float), 0, 0.0))
    out = (c[w:] - c[:-w]) / w
    return np.arange(w, len(hits) + 1), out


def hoeffding_story():
    rng = np.random.default_rng(0)
    ht = HoeffdingTreeClassifier(grace_period=100)
    xs, ys, hits, split_at = [], [], [], None
    for i in range(2000):
        y = int(rng.random() < 0.5)
        x = {"tau": rng.normal(2.0 + 3.0 * y, 0.5), "qd": rng.normal(0.0, 1.0)}
        hits.append(ht.predict_one(x) == y)
        ht.learn_one(x, y)
        xs.append((x["tau"], x["qd"]))
        ys.append(y)
        if split_at is None and ht.n_nodes_ > 1:
            split_at = i + 1
    thr = ht.explain_one({"tau": 5.2, "qd": 0.0})[0]["threshold"]
    xs, ys = np.array(xs), np.array(ys)
    fig = plt.figure(figsize=(16, 5.6))
    fig.subplots_adjust(top=0.72, bottom=0.12, left=0.05, right=0.985, wspace=0.22)
    heading(
        fig,
        "A HOEFFDING TREE DECIDES WHEN IT HAS PROOF",
        f"2000 samples of a robot arm: idle or carrying a load. its first split comes at sample {split_at}, on the torque",
    )
    ax = fig.add_subplot(1, 3, 1)
    for cls, col in ((0, CYAN), (1, ORANGE)):
        m = ys == cls
        ax.scatter(xs[m, 0], xs[m, 1], s=7, color=col, alpha=0.75, edgecolor="none")
    ax.axvline(thr, color=PINK, lw=2.2)
    ax.text(
        thr + 0.08, 2.9, f"tau > {thr:.2f}", color=PINK, fontsize=11, fontweight="bold"
    )
    ax.set_xlabel("joint torque tau [N·m]")
    ax.set_ylabel("joint velocity qd")
    ax.set_title("the question it learned", color=INK, loc="left", fontsize=12)
    ax2 = fig.add_subplot(1, 3, 2)
    n = np.linspace(40, 2000, 400)
    eps = np.sqrt(np.log(1 / 1e-7) / (2 * n))
    ax2.plot(n, eps, color=CYAN, lw=2.4)
    ax2.fill_between(n, eps, 0, color=CYAN, alpha=0.12)
    e200 = np.sqrt(np.log(1 / 1e-7) / 400)
    ax2.scatter([200], [e200], s=70, color=ORANGE, zorder=3)
    ax2.annotate(
        f"n = 200, ε = {e200:.2f} bits",
        (200, e200),
        (520, 0.33),
        color=ORANGE,
        fontsize=11,
        arrowprops={"arrowstyle": "-", "color": ORANGE},
    )
    ax2.set_xlabel("samples at the leaf, n")
    ax2.set_ylabel("Hoeffding bound ε [bits]")
    ax2.set_title(
        "the more it has seen, the smaller the doubt",
        color=INK,
        loc="left",
        fontsize=12,
    )
    ax2.grid(True)
    ax3 = fig.add_subplot(1, 3, 3)
    t, acc = rolling(hits, 100)
    ax3.plot(t, acc, color=CYAN, lw=2.2)
    ax3.axvline(split_at, color=PINK, lw=1.8, ls="--")
    ax3.text(split_at + 25, 0.58, "first split", color=PINK, fontsize=11)
    ax3.set_ylim(0.45, 1.02)
    ax3.set_xlabel("samples")
    ax3.set_ylabel("accuracy over the last 100")
    ax3.set_title(
        "tested before learning each sample", color=INK, loc="left", fontsize=12
    )
    ax3.grid(True)
    fig.savefig(HERE / "hoeffding_story.png", dpi=DPI, facecolor=BG)
    plt.close(fig)
    print(
        "hoeffding split at",
        split_at,
        "threshold",
        round(thr, 3),
        "eps200",
        round(e200, 3),
    )


def torque(q1, q2):
    return 9.81 * (2.0 * np.cos(q1) + np.cos(q1 + q2))


def surface_3d(name, model, title, sub, n_train=20000, seed=3):
    rng = np.random.default_rng(seed)
    for _ in range(n_train):
        q1, q2 = rng.uniform(-np.pi, np.pi, 2)
        model.learn_one({"q1": float(q1), "q2": float(q2)}, float(torque(q1, q2)))
    g = np.linspace(-np.pi, np.pi, 70)
    Q1, Q2 = np.meshgrid(g, g)
    truth = torque(Q1, Q2)
    pred = np.array(
        [[model.predict_one({"q1": float(a), "q2": float(b)}) for a in g] for b in g]
    )
    cmap = LinearSegmentedColormap.from_list("t", [CYAN, "#6a5acd", PINK])
    norm = plt.Normalize(truth.min(), truth.max())
    fig = plt.figure(figsize=(16, 6.4))
    heading(fig, title, sub, y=0.94)
    boxes = ([0.0, 0.04, 0.34, 0.74], [0.33, 0.04, 0.34, 0.74])
    for k, (z, lab) in enumerate(
        ((truth, "true torque"), (pred, "the tree after 20000 samples"))
    ):
        ax = fig.add_axes(boxes[k], projection="3d")
        ax.set_facecolor(BG)
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.set_pane_color((0.07, 0.10, 0.14, 1.0))
            axis._axinfo["grid"]["color"] = GRID
        ax.plot_surface(
            Q1,
            Q2,
            z,
            facecolors=cmap(norm(z)),
            rstride=1,
            cstride=1,
            linewidth=0,
            antialiased=True,
            alpha=0.95,
        )
        ax.view_init(elev=28, azim=-56)
        ax.set_box_aspect((1, 1, 0.7))
        ax.set_title(lab, color=INK, fontsize=12, pad=-4)
        ax.set_xlabel("q1", color=CYAN)
        ax.set_ylabel("q2", color=CYAN)
        ax.tick_params(colors=MUTED, labelsize=7)
    err = np.abs(pred - truth)
    ax = fig.add_axes([0.715, 0.14, 0.21, 0.56])
    im = ax.imshow(
        err,
        extent=(-np.pi, np.pi, -np.pi, np.pi),
        origin="lower",
        cmap=LinearSegmentedColormap.from_list("e", [BG, CYAN, ORANGE, PINK]),
    )
    ax.set_xlabel("q1")
    ax.set_ylabel("q2")
    ax.set_title(f"absolute error (mean {err.mean():.1f} N·m)", color=INK, fontsize=12)
    cb = fig.colorbar(im, cax=fig.add_axes([0.94, 0.14, 0.012, 0.56]))
    cb.ax.tick_params(colors=MUTED)
    fig.savefig(HERE / name, dpi=DPI, facecolor=BG)
    plt.close(fig)
    print(
        name,
        "mean error on the grid",
        round(float(err.mean()), 2),
        "nodes",
        model.n_nodes_,
    )


def efdt_replace():
    rng = np.random.default_rng(2)
    ht = HoeffdingTreeClassifier(grace_period=100)
    ef = HoeffdingAnytimeTreeClassifier(grace_period=100)
    hh, he, rf_ht, rf_ef = [], [], [], []
    for i in range(12000):
        y = int(rng.random() < 0.5)
        a = rng.normal(3.0 * y if i < 3000 else 0.0, 0.5)
        b = rng.normal(3.0 * y if i >= 3000 else 0.0, 0.5)
        x = {"a": a, "b": b}
        hh.append(ht.predict_one(x) == y)
        he.append(ef.predict_one(x) == y)
        ht.learn_one(x, y)
        ef.learn_one(x, y)
        rf_ht.append(ht.root_.split_feature_)
        rf_ef.append(ef.root_.split_feature_)
    switch = next(i for i, f in enumerate(rf_ef) if f == "b") + 1
    fig = plt.figure(figsize=(16, 6.2))
    heading(
        fig,
        "THE ANYTIME TREE CHANGES ITS MIND",
        f"at sample 3000 the useful feature changes from a to b: the Anytime tree moves its root to b at sample {switch}, the plain tree never does",
    )
    gs = fig.add_gridspec(
        3,
        1,
        height_ratios=[1.6, 1.0, 1.0],
        top=0.74,
        bottom=0.11,
        left=0.07,
        right=0.985,
        hspace=0.18,
    )
    ax = fig.add_subplot(gs[0])
    for h, col, lab in (
        (hh, ORANGE, "Hoeffding tree"),
        (he, CYAN, "Hoeffding Anytime tree"),
    ):
        t, acc = rolling(h, 300)
        ax.plot(t, acc, color=col, lw=2.2, label=lab)
    ax.axvline(3000, color=PINK, lw=1.8, ls="--")
    ax.text(3080, 0.915, "the useful feature changes", color=PINK, fontsize=11)
    ax.set_ylim(0.9, 1.005)
    ax.set_xlim(0, 12000)
    ax.set_ylabel("accuracy, last 300")
    ax.legend(loc="lower right", facecolor=PANEL, edgecolor=GRID, labelcolor=INK)
    ax.grid(True)
    ax.tick_params(labelbottom=False)
    for k, (feat, lab) in enumerate(((rf_ht, "Hoeffding"), (rf_ef, "Anytime"))):
        a2 = fig.add_subplot(gs[k + 1])
        col = np.array(
            [[CYAN if f == "a" else (ORANGE if f == "b" else GRID) for f in feat[::20]]]
        )
        from matplotlib.colors import to_rgb

        a2.imshow([[to_rgb(c) for c in col[0]]], aspect="auto", extent=(0, 12000, 0, 1))
        if k == 1:
            a2.axvline(switch, color=INK, lw=1.6)
            a2.text(
                switch + 120,
                0.5,
                f"root moves to b: sample {switch}",
                color=BG,
                fontsize=11,
                fontweight="bold",
                va="center",
            )
        a2.set_yticks([])
        a2.set_ylabel(
            f"root\n{lab}", rotation=0, ha="right", va="center", fontsize=9, color=MUTED
        )
        if k == 0:
            a2.tick_params(labelbottom=False)
        else:
            a2.set_xlabel("samples")
    fig.text(
        0.985,
        0.045,
        "root question:  cyan = a   orange = b",
        ha="right",
        color=MUTED,
        fontsize=10,
    )
    fig.savefig(HERE / "efdt_replace.png", dpi=DPI, facecolor=BG)
    plt.close(fig)
    print(
        "efdt root at end:",
        rf_ht[-1],
        rf_ef[-1],
        "replaced",
        ef.n_replaced_,
        "switch",
        switch,
    )


def adaptive_recovery():
    rng = np.random.default_rng(1)
    ht = HoeffdingTreeClassifier(grace_period=100)
    hat = HoeffdingAdaptiveTreeClassifier(grace_period=100, drift_detector=ADWIN())
    hh, ha, drifts, swaps = [], [], [], []
    d0, s0 = 0, 0
    for i in range(6000):
        y = int(rng.random() < 0.5)
        c = 3.0 * y if i < 3000 else 3.0 * (1 - y)
        x = {"tau": rng.normal(2.0 + c, 0.5), "qd": rng.normal(0.0, 1.0)}
        hh.append(ht.predict_one(x) == y)
        ha.append(hat.predict_one(x) == y)
        ht.learn_one(x, y)
        hat.learn_one(x, y)
        if hat.n_drifts_ > d0:
            drifts.append(i)
            d0 = hat.n_drifts_
        if hat.n_swaps_ > s0:
            swaps.append(i)
            s0 = hat.n_swaps_
    fig, ax = plt.subplots(figsize=(16, 5.8))
    fig.subplots_adjust(top=0.74, bottom=0.12, left=0.06, right=0.985)
    heading(
        fig,
        "THE ADAPTIVE TREE FOLLOWS A SUDDEN CHANGE",
        "at sample 3000 the meaning of the torque flips: the plain tree keeps the old answers, the adaptive tree grows a new subtree and swaps it in",
    )
    for h, col, lab in (
        (hh, ORANGE, "Hoeffding tree"),
        (ha, CYAN, "Hoeffding Adaptive tree (ADWIN)"),
    ):
        t, acc = rolling(h, 200)
        ax.plot(t, acc, color=col, lw=2.4, label=lab)
    ax.axvline(3000, color=INK, lw=1.4, ls=":")
    ax.text(2960, 0.62, "the meaning of tau flips", color=INK, fontsize=11, ha="right")
    for k, d in enumerate(drifts):
        ax.axvline(d, color=PINK, lw=1.3, alpha=0.9, ymax=0.2)
        if k == 0:
            ax.text(2960, 0.1, "drift signals", color=PINK, fontsize=10, ha="right")
    for k, s in enumerate(swaps):
        ax.axvline(s, color=CYAN, lw=2.0, ymax=0.35, ymin=0.2)
        ax.text(s + 40, 0.28, "swap", color=CYAN, fontsize=10, fontweight="bold")
    ax.set_ylim(0.0, 1.03)
    ax.set_xlim(0, 6000)
    ax.set_xlabel("samples")
    ax.set_ylabel("accuracy over the last 200")
    ax.legend(
        loc="lower left",
        bbox_to_anchor=(0.0, 0.42),
        facecolor=PANEL,
        edgecolor=GRID,
        labelcolor=INK,
    )
    ax.grid(True)
    fig.savefig(HERE / "adaptive_recovery.png", dpi=DPI, facecolor=BG)
    plt.close(fig)
    print("adaptive drifts at", drifts, "swaps at", swaps)


if __name__ == "__main__":
    hoeffding_story()
    surface_3d(
        "hoeffding_regressor_surface.png",
        HoeffdingTreeRegressor(grace_period=100),
        "A HOEFFDING REGRESSION TREE LEARNS A TORQUE",
        "gravity torque of the first joint of a two-link arm: the tree cuts the angles into steps that follow the surface",
    )
    efdt_replace()
    adaptive_recovery()
    surface_3d(
        "sgt_3d.png",
        SGTRegressor(),
        "A STOCHASTIC GRADIENT TREE LEARNS THE SAME TORQUE",
        "it cuts where the gradient of the loss says a split lowers it, and a t-test decides when the gain is real",
    )
