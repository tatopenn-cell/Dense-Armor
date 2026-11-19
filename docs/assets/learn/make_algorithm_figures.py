"""Dark figures for the pages of OASIS, LEGO, POLA and Platt scaling, drawn with the library itself, on
the bank customers of the first example of each page.

Generates in the same folder:
- oasis_learning.png      the weight that OASIS gives to each column while it sees triplets, and the matrix W
- oasis_geometry.png      the points at the same distance from a customer, with the Euclidean and the learned distance
- oasis_vs_sym.png        the two ways of making a distance of W, on text (numbers published in Discovery)
- lego_learning.png       the squared distance of two probe pairs moving to their targets, and the smallest eigenvalue of A
- lego_geometry.png       as oasis_geometry.png, for LEGO
- pola_learning.png       the threshold b, the distances of two probe pairs and the corrections of A
- pola_geometry.png       as oasis_geometry.png, for POLA
- platt_surface.png       the map from the raw probability to the calibrated one, while the payments arrive (3D)
- platt_parameters.png    the two parameters a and b of the map, and the final map

Run from the repo root:
    python docs/assets/learn/make_algorithm_figures.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from dense_armor.core.online.learn.calibration import OnlinePlattScaling
from dense_armor.core.online.learn.metric_learning import LEGO, OASIS, POLA
from dense_armor.core.online.learn.online_classifiers import OnlineGaussianNB

HERE = Path(__file__).resolve().parent
BG, PANEL, GRID = "#0b1118", "#111a24", "#26364a"
INK, MUTED = "#e8fbff", "#8fa3b8"
CYAN, ORANGE, PINK = "#3ec6e8", "#ffa21f", "#ff2d87"
COLS = ["age", "balance", "spend", "visits"]
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


def heading(fig, title, sub, y=0.945):
    fig.text(0.04, y, title, fontsize=20, fontweight="bold", color=INK)
    fig.text(0.04, y - 0.06, sub, fontsize=11.5, color=CYAN)


def customers(seed, two=False):
    rng = np.random.default_rng(seed)

    def customer(young):
        d = {"age": rng.normal(-1.0 if young else 1.0, 0.5), "spend": rng.normal(0, 1)}
        if not two:
            d.update({"balance": rng.normal(0, 1), "visits": rng.normal(0, 1)})
        return d

    return rng, customer


def shares(M):
    d = np.diag(M)
    return d / d.sum()


def pools(customer, rng, n=200):
    same, diff = [], []
    for _ in range(n):
        g = rng.random() < 0.5
        same.append((customer(g), customer(g)))
        diff.append((customer(g), customer(not g)))
    return same, diff


def mean_d2(fn, pairs):
    return float(np.mean([fn(a, b) ** 2 for a, b in pairs]))


def across(runs, key):
    """median and quartiles over the seeds of the series runs[seed][key] (same length)."""
    arr = np.array([r[key] for r in runs], float)
    return np.median(arr, 0), np.percentile(arr, 25, 0), np.percentile(arr, 75, 0)


MARKS = np.unique(np.logspace(0, np.log10(2000), 40).astype(int))
SEEDS = range(10)


# ------------------------------------------------------------------------------------------- OASIS
def oasis_learning():
    rng, customer = customers(0)
    oasis = OASIS(C=0.1)
    marks = sorted(
        {int(v) for v in np.unique(np.logspace(0, np.log10(2000), 40).astype(int))}
    )
    hist, snaps = [], {}
    for i in range(1, 2001):
        young = rng.random() < 0.5
        oasis.learn_triplet(customer(young), customer(young), customer(not young))
        if i in marks:
            hist.append((i, shares(oasis.W.T @ oasis.W)))
        if i in (1, 50, 500, 2000):
            snaps[i] = oasis.W.copy()
    fig = plt.figure(figsize=(16, 6.6))
    heading(
        fig,
        "OASIS LEARNS WHAT TO LOOK AT",
        "the share of the distance that OASIS gives to each column of a customer, triplet after triplet: only the age tells the groups apart",
    )
    ax = fig.add_axes([0.05, 0.1, 0.45, 0.62])
    for k, (col, c) in enumerate(zip(COLS, (CYAN, MUTED, ORANGE, PINK))):
        ax.plot(
            [h[0] for h in hist],
            [h[1][k] * 100 for h in hist],
            color=c,
            lw=2.6 if col == "age" else 1.8,
            label=col,
        )
    ax.axhline(25, color=INK, lw=1.0, ls="--")
    ax.text(1.3, 27, "Euclidean: 25 % each", color=INK, fontsize=10)
    ax.set_xscale("log")
    ax.set_xlabel("triplets seen")
    ax.set_ylabel("share of the distance [%]")
    ax.set_ylim(0, 100)
    ax.legend(loc="center left", facecolor=PANEL, edgecolor=GRID, labelcolor=INK)
    ax.grid(True)
    lim = max(np.abs(s).max() for s in snaps.values())
    cmap = LinearSegmentedColormap.from_list("w", [PINK, PANEL, CYAN])
    for k, (i, W) in enumerate(snaps.items()):
        a = fig.add_axes([0.56 + (k % 2) * 0.21, 0.55 - (k // 2) * 0.38, 0.17, 0.2])
        a.imshow(W, cmap=cmap, vmin=-lim, vmax=lim)
        a.set_xticks(range(4))
        a.set_yticks(range(4))
        a.set_xticklabels(COLS, fontsize=7, rotation=45)
        a.set_yticklabels(COLS, fontsize=7)
        a.set_title(
            f"W after {i} triplet" + ("" if i == 1 else "s"), color=INK, fontsize=10
        )
        a.grid(False)
    fig.savefig(HERE / "oasis_learning.png", dpi=DPI, facecolor=BG)
    plt.close(fig)
    print("oasis final shares", np.round(hist[-1][1] * 100).astype(int).tolist())


# ------------------------------------------------------------------------------------------- geometry
def geometry(name, learner, train, title, sub):
    rng, customer = customers(3, two=True)
    train(learner, rng, customer)
    pts = [(customer(g), g) for g in (rng.random(260) < 0.5)]
    xs = np.array([[p["age"], p["spend"]] for p, _ in pts])
    ys = np.array([g for _, g in pts])
    gx = np.linspace(-3.2, 3.2, 160)
    gy = np.linspace(-3.2, 3.2, 160)
    anchor = {"age": 0.0, "spend": 0.0}

    def field(fn):
        return np.array(
            [[fn(anchor, {"age": float(a), "spend": float(b)}) for a in gx] for b in gy]
        )

    euc = field(
        lambda u, v: float(np.hypot(u["age"] - v["age"], u["spend"] - v["spend"]))
    )
    lrn = field(learner.distance)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6.8))
    fig.subplots_adjust(top=0.76, bottom=0.14, left=0.05, right=0.985, wspace=0.12)
    heading(fig, title, sub)
    for ax, f, t, levels in (
        (
            axes[0],
            euc,
            "the Euclidean distance: circles, every column counts the same",
            [1, 2, 3],
        ),
        (axes[1], lrn, "the learned distance: the useless column counts less", None),
    ):
        if levels is None:
            ref = np.interp(2.0, np.sort(euc.ravel()), np.sort(f.ravel()))
            levels = [ref * 0.5, ref, ref * 1.5]
        for cls, col in ((1, CYAN), (0, ORANGE)):
            m = ys == cls
            ax.scatter(xs[m, 0], xs[m, 1], s=13, color=col, alpha=0.8, edgecolor="none")
        cs = ax.contour(
            gx, gy, f, levels=levels, colors=[PINK], linewidths=[1.2, 2.4, 1.2]
        )
        ax.clabel(cs, fmt="%.1f", fontsize=8)
        ax.scatter([0], [0], s=90, color=INK, zorder=5)
        ax.set_xlim(-3.2, 3.2)
        ax.set_ylim(-3.2, 3.2)
        ax.set_xlabel("age (tells the two groups apart)")
        ax.set_ylabel("spend (tells nothing)")
        ax.set_title(t, color=INK, loc="left", fontsize=12)
        ax.set_aspect("equal")
        ax.grid(True)
    fig.text(
        0.5,
        0.012,
        "white dot: a customer; pink lines: points at the same distance from it; cyan and orange: the two kinds of customers",
        ha="center",
        color=MUTED,
        fontsize=10,
    )
    fig.savefig(HERE / name, dpi=DPI, facecolor=BG)
    plt.close(fig)


def oasis_geometry():
    def train(o, rng, customer):
        for _ in range(2000):
            g = rng.random() < 0.5
            o.learn_triplet(customer(g), customer(g), customer(not g))

    geometry(
        "oasis_geometry.png",
        OASIS(C=0.1),
        train,
        "WHERE OASIS DRAWS THE LINE",
        "the points at the same distance from a customer: the learned distance stretches the age and squeezes the useless column",
    )


def oasis_vs_sym():
    labels = [
        "sci.space vs\ncomp.graphics\nC = 0.01",
        "sci.space vs\ncomp.graphics\nC = 0.1",
        "ibm.pc.hardware vs\nsci.electronics\nC = 0.01",
        "ibm.pc.hardware vs\nsci.electronics\nC = 0.1",
    ]
    euc = [0.5255, 0.5255, 0.5051, 0.5051]
    sym = [0.6071, 0.6922, 0.5356, 0.6085]
    wtw = [0.6820, 0.8588, 0.5593, 0.7831]
    fig, ax = plt.subplots(figsize=(16, 5.8))
    fig.subplots_adjust(top=0.74, bottom=0.2, left=0.06, right=0.985)
    heading(
        fig,
        "FROM A SIMILARITY TO A DISTANCE: WᵀW",
        "accuracy of a k-nearest-neighbours classifier on 20 newsgroups (500 word features, 20,000 triplets); numbers published in Dense-Evolution-Discovery",
    )
    x = np.arange(4)
    w = 0.26
    for k, (v, name, col) in enumerate(
        ((euc, "Euclidean", MUTED), (sym, "(W + Wᵀ) / 2", ORANGE), (wtw, "WᵀW", CYAN))
    ):
        bars = ax.bar(x + (k - 1) * w, v, w, color=col, label=name)
        for b, val in zip(bars, v):
            ax.text(
                b.get_x() + b.get_width() / 2,
                val + 0.012,
                f"{val:.2f}",
                ha="center",
                fontsize=9,
                color=INK,
            )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylim(0.4, 0.95)
    ax.set_ylabel("accuracy")
    ax.legend(loc="upper left", facecolor=PANEL, edgecolor=GRID, labelcolor=INK)
    ax.grid(True, axis="y")
    fig.savefig(HERE / "oasis_vs_sym.png", dpi=DPI, facecolor=BG)
    plt.close(fig)


# ------------------------------------------------------------------------------------------- LEGO
def lego_run(seed):
    rng, customer = customers(seed)
    labels = rng.random(200) < 0.5
    X = np.array([list(customer(g).values()) for g in labels])
    near, far = LEGO.percentile_targets(X, labels)
    lego = LEGO(eta=0.1)
    same, diff = pools(customer, rng)
    out = {"near": near, "far": far, "same": [], "diff": [], "emin": [], "emax": []}
    for i in range(1, 2001):
        g = rng.random() < 0.5
        if rng.random() < 0.5:
            lego.learn_pair(customer(g), customer(g), near)
        else:
            lego.learn_pair(customer(g), customer(not g), far)
        if i in MARKS:
            ev = np.linalg.eigvalsh(lego.A)
            out["same"].append(mean_d2(lego.distance, same))
            out["diff"].append(mean_d2(lego.distance, diff))
            out["emin"].append(ev.min())
            out["emax"].append(ev.max())
    return out


def lego_learning():
    runs = [lego_run(sd) for sd in SEEDS]
    near, far = (
        np.median([r["near"] for r in runs]),
        np.median([r["far"] for r in runs]),
    )
    fig = plt.figure(figsize=(16, 6.2))
    heading(
        fig,
        "LEGO MOVES DISTANCES TOWARD THEIR TARGETS",
        "squared distance of 200 fresh pairs, on average, over 10 runs (line: median, band: quartiles) while LEGO sees pairs and their targets",
    )
    a1 = fig.add_axes([0.05, 0.12, 0.43, 0.6])
    for key, col, lab in (
        ("same", CYAN, "pairs of the same kind"),
        ("diff", ORANGE, "pairs of different kinds"),
    ):
        m, lo, hi = across(runs, key)
        a1.plot(MARKS, m, color=col, lw=2.6, label=lab)
        a1.fill_between(MARKS, lo, hi, color=col, alpha=0.2, lw=0)
    a1.axhline(near, color=CYAN, lw=1.0, ls="--")
    a1.axhline(far, color=ORANGE, lw=1.0, ls="--")
    a1.text(1.3, near + 1.0, f"target {near:.2f}", color=CYAN, fontsize=10)
    a1.text(1.3, far + 1.0, f"target {far:.2f}", color=ORANGE, fontsize=10)
    a1.set_xscale("log")
    a1.set_ylim(0, 45)
    a1.set_xlabel("pairs seen")
    a1.set_ylabel("squared distance of a pair")
    a1.legend(loc="center right", facecolor=PANEL, edgecolor=GRID, labelcolor=INK)
    a1.grid(True)
    a2 = fig.add_axes([0.56, 0.12, 0.42, 0.6])
    for key, col, lab in (
        ("emax", ORANGE, "largest eigenvalue of A"),
        ("emin", PINK, "smallest eigenvalue of A"),
    ):
        m, lo, hi = across(runs, key)
        a2.plot(MARKS, m, color=col, lw=2.6, label=lab)
        a2.fill_between(MARKS, lo, hi, color=col, alpha=0.2, lw=0)
    a2.set_xscale("log")
    a2.set_yscale("log")
    a2.set_xlabel("pairs seen")
    a2.set_title(
        "the smallest eigenvalue stays above zero: a proper distance, with no projection",
        color=INK,
        loc="left",
        fontsize=11.5,
    )
    a2.legend(loc="upper left", facecolor=PANEL, edgecolor=GRID, labelcolor=INK)
    a2.grid(True)
    fig.savefig(HERE / "lego_learning.png", dpi=DPI, facecolor=BG)
    plt.close(fig)
    ms, md = across(runs, "same")[0][-1], across(runs, "diff")[0][-1]
    print(
        "lego targets",
        round(near, 2),
        round(far, 2),
        "median final means",
        round(ms, 2),
        round(md, 2),
        "global min eig",
        float(min(min(r["emin"]) for r in runs)),
    )


def lego_geometry():
    def train(lego, rng, customer):
        labels = rng.random(200) < 0.5
        X = np.array([[d["age"], d["spend"]] for d in (customer(g) for g in labels)])
        near, far = LEGO.percentile_targets(X, labels)
        for _ in range(2000):
            g = rng.random() < 0.5
            if rng.random() < 0.5:
                lego.learn_pair(customer(g), customer(g), near)
            else:
                lego.learn_pair(customer(g), customer(not g), far)

    geometry(
        "lego_geometry.png",
        LEGO(eta=0.1),
        train,
        "WHERE LEGO DRAWS THE LINE",
        "the points at the same distance from a customer: the useless column is squeezed, and the matrix stays a distance",
    )


# ------------------------------------------------------------------------------------------- POLA
def pola_run(seed):
    rng, customer = customers(seed)
    pola = POLA()
    same, diff = pools(customer, rng)
    out = {"same": [], "diff": [], "b": [], "corr": []}
    n_corr = 0
    for i in range(1, 2001):
        g = rng.random() < 0.5
        y = +1 if rng.random() < 0.5 else -1
        a = customer(g)
        c = customer(g) if y == 1 else customer(not g)
        z = np.array([a[k] - c[k] for k in sorted(a)])
        if pola.A is not None and y == 1:
            loss = max(0.0, (z @ pola.A @ z - pola.b) + 1.0)
            if loss > 0:
                step = pola.A - (loss / (1.0 + (z @ z) ** 2)) * np.outer(z, z)
                n_corr += int(np.linalg.eigvalsh(step).min() < 0)
        pola.learn_pair(a, c, y)
        if i in MARKS:
            out["same"].append(mean_d2(pola.distance, same))
            out["diff"].append(mean_d2(pola.distance, diff))
            out["b"].append(pola.b)
            out["corr"].append(n_corr)
    return out


def pola_learning():
    runs = [pola_run(sd) for sd in SEEDS]
    fig = plt.figure(figsize=(16, 6.2))
    heading(
        fig,
        "POLA MOVES A THRESHOLD AND KEEPS A VALID DISTANCE",
        "squared distance of 200 fresh pairs, on average, over 10 runs (line: median, band: quartiles) and the threshold b: a pair is called similar below b",
    )
    a1 = fig.add_axes([0.05, 0.12, 0.6, 0.6])
    for key, col, lab in (
        ("same", CYAN, "pairs of the same kind (should be below b)"),
        ("diff", ORANGE, "pairs of different kinds (should be above b)"),
        ("b", PINK, "threshold b"),
    ):
        m, lo, hi = across(runs, key)
        a1.plot(MARKS, m, color=col, lw=2.6, ls="--" if key == "b" else "-", label=lab)
        a1.fill_between(MARKS, lo, hi, color=col, alpha=0.18, lw=0)
    a1.set_xscale("log")
    a1.set_xlabel("pairs seen")
    a1.set_ylabel("squared distance of a pair")
    a1.legend(loc="upper left", facecolor=PANEL, edgecolor=GRID, labelcolor=INK)
    a1.grid(True)
    a2 = fig.add_axes([0.72, 0.12, 0.26, 0.6])
    m, lo, hi = across(runs, "corr")
    a2.plot(MARKS, m, color=PINK, lw=2.6)
    a2.fill_between(MARKS, lo, hi, color=PINK, alpha=0.2, lw=0)
    a2.set_xscale("log")
    a2.set_xlabel("pairs seen")
    a2.set_ylabel("corrections of A (cumulative)")
    a2.set_title(
        "a negative eigenvalue cut back to zero", color=INK, loc="left", fontsize=11
    )
    a2.grid(True)
    fig.savefig(HERE / "pola_learning.png", dpi=DPI, facecolor=BG)
    plt.close(fig)
    fs, fd, fb, fc = (across(runs, k)[0][-1] for k in ("same", "diff", "b", "corr"))
    print(
        "pola median final: same",
        round(fs, 2),
        "diff",
        round(fd, 2),
        "b",
        round(fb, 2),
        "corrections",
        round(fc),
    )


def pola_geometry():
    def train(pola, rng, customer):
        for _ in range(2000):
            g = rng.random() < 0.5
            if rng.random() < 0.5:
                pola.learn_pair(customer(g), customer(g), +1)
            else:
                pola.learn_pair(customer(g), customer(not g), -1)

    geometry(
        "pola_geometry.png",
        POLA(),
        train,
        "WHERE POLA DRAWS THE LINE",
        "the points at the same distance from a customer: after 2000 labelled pairs the age counts and the useless column is almost ignored",
    )


# ------------------------------------------------------------------------------------------- Platt
def platt_run():
    rng = np.random.default_rng(0)
    cal = OnlinePlattScaling(OnlineGaussianNB())
    raw = OnlineGaussianNB()
    marks = sorted(
        {int(v) for v in np.unique(np.logspace(2, np.log10(4000), 40).astype(int))}
    )
    hist = []
    for i in range(1, 4001):
        fraud = int(rng.random() < 0.5)
        shared = rng.normal(0, 1)
        x = {
            f"s{k}": 0.6 * (2 * fraud - 1) + shared + 0.3 * rng.normal()
            for k in range(6)
        }
        cal.predict_proba_one(x)
        raw.learn_one(x, fraud)
        cal.learn_one(x, fraud)
        if i in marks:
            hist.append((i, float(cal._theta[0]), float(cal._theta[1])))
    return hist


def platt_surface(hist):
    p = np.linspace(0.01, 0.99, 99)
    logit = np.log(p / (1 - p))
    t = np.array([h[0] for h in hist], float)
    Z = np.array([1 / (1 + np.exp(-(a * logit + b))) for _, a, b in hist])
    P, T = np.meshgrid(p, np.log10(t))
    cmap = LinearSegmentedColormap.from_list("s", [CYAN, "#6a5acd", PINK])
    fig = plt.figure(figsize=(12, 8.2))
    ax = fig.add_subplot(111, projection="3d")
    fig.subplots_adjust(top=0.93, bottom=0.0, left=-0.05, right=1.05)
    ax.set_facecolor(BG)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_pane_color((0.07, 0.10, 0.14, 1.0))
        axis._axinfo["grid"]["color"] = GRID
    ax.plot_surface(
        P,
        T,
        Z,
        facecolors=cmap(Z),
        rstride=1,
        cstride=1,
        linewidth=0,
        antialiased=True,
        alpha=0.95,
    )
    ax.plot(p, np.full_like(p, np.log10(t[0])), p, color=INK, lw=1.6, ls="--")
    ax.view_init(elev=26, azim=-62)
    ax.set_box_aspect((1, 1.2, 0.75), zoom=1.1)
    ax.set_xlabel("probability announced by the model", color=CYAN, labelpad=8)
    ax.set_ylabel("payments seen (log10)", color=CYAN, labelpad=8)
    ax.set_zlabel("calibrated probability", color=CYAN, labelpad=6)
    ax.tick_params(colors=MUTED, labelsize=8)
    fig.text(
        0.05,
        0.955,
        "PLATT SCALING: THE MAP THAT FIXES THE NUMBERS",
        fontsize=21,
        fontweight="bold",
        color=INK,
    )
    fig.text(
        0.05,
        0.915,
        "at the start the map is the identity (dashed line); as payments arrive it learns to pull the over-confident numbers toward the truth",
        fontsize=11.5,
        color=CYAN,
    )
    fig.savefig(HERE / "platt_surface.png", dpi=DPI, facecolor=BG)
    plt.close(fig)


def platt_parameters(hist):
    t = [h[0] for h in hist]
    fig = plt.figure(figsize=(16, 6.0))
    heading(
        fig,
        "THE TWO NUMBERS THAT FIX THE PROBABILITIES",
        "σ(a · logit(p) + b): a below 1 shrinks over-confident probabilities toward 0.5, b shifts them",
    )
    a1 = fig.add_axes([0.05, 0.12, 0.42, 0.6])
    a1.plot(t, [h[1] for h in hist], color=CYAN, lw=2.6, label="a (the stretch)")
    a1.plot(t, [h[2] for h in hist], color=ORANGE, lw=2.6, label="b (the shift)")
    a1.axhline(1, color=CYAN, lw=0.8, ls=":")
    a1.axhline(0, color=ORANGE, lw=0.8, ls=":")
    a1.set_xscale("log")
    a1.set_xlabel("payments seen")
    a1.legend(loc="center right", facecolor=PANEL, edgecolor=GRID, labelcolor=INK)
    a1.grid(True)
    a2 = fig.add_axes([0.56, 0.12, 0.42, 0.6])
    p = np.linspace(0.01, 0.99, 200)
    lg = np.log(p / (1 - p))
    a, b = hist[-1][1], hist[-1][2]
    a2.plot(p, p, color=INK, lw=1.2, ls="--", label="identity (no calibration)")
    a2.plot(
        p,
        1 / (1 + np.exp(-(a * lg + b))),
        color=PINK,
        lw=2.8,
        label=f"learned map: a = {a:.2f}, b = {b:.2f}",
    )
    a2.set_xlabel("probability announced by the model")
    a2.set_ylabel("calibrated probability")
    a2.legend(loc="upper left", facecolor=PANEL, edgecolor=GRID, labelcolor=INK)
    a2.grid(True)
    fig.savefig(HERE / "platt_parameters.png", dpi=DPI, facecolor=BG)
    plt.close(fig)
    print("platt final a, b", round(a, 3), round(b, 3))


if __name__ == "__main__":
    oasis_learning()
    oasis_geometry()
    oasis_vs_sym()
    lego_learning()
    lego_geometry()
    pola_learning()
    pola_geometry()
    h = platt_run()
    platt_surface(h)
    platt_parameters(h)
