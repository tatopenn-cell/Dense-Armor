"""Dark figures for the pages of the online classifiers, drawn with the library itself, on the card
payments of the first example of each page (half of them fraud; at payment 3000 the fraudsters change
habit and the meaning of the amount flips).

Generates in the same folder:
- nb_learning.png        what OnlineGaussianNB believes about the amount of a fraud, with and without forgetting
- softmax_boundary.png   the probability of fraud given the amount, at five moments of the stream
- drift_swap.png         the drift-adaptive wrapper: when the backup starts and when it takes over

The first and the third use the internal statistics of the classifiers (``_mu``, ``_bg``, ``_main``), which
the pages do not use.

Run from the repo root:
    python docs/assets/learn/make_classifier_figures.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from dense_armor.core.online.drift.detector import CUSUMDriftDetector
from dense_armor.core.online.learn.online_classifiers import (
    DriftAdaptiveClassifier,
    OnlineGaussianNB,
    OnlineSoftmaxRegression,
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


def heading(fig, title, sub, y=0.945):
    fig.text(0.04, y, title, fontsize=20, fontweight="bold", color=INK)
    fig.text(0.04, y - 0.06, sub, fontsize=11.5, color=CYAN)


def stream(seed=1, n=6000):
    rng = np.random.default_rng(seed)
    for i in range(n):
        fraud = int(rng.random() < 0.5)
        big = fraud if i < 3000 else 1 - fraud
        x = {
            "amount": float(rng.normal(400 if big else 60, 60 if big else 20)),
            "hour": float(rng.normal(14, 3)),
        }
        yield i, x, fraud


def rolling(hits, w):
    c = np.cumsum(np.insert(np.asarray(hits, dtype=float), 0, 0.0))
    return np.arange(w, len(hits) + 1), (c[w:] - c[:-w]) / w


def nb_learning():
    runs = {a: OnlineGaussianNB(alpha=a) for a in (0.0, 0.1)}
    mean = {a: [] for a in runs}
    snap = {}
    for i, x, y in stream():
        for a, nb in runs.items():
            nb.learn_one(x, y)
            mu = nb._mu.get(1)
            mean[a].append(float(mu[0]) if mu is not None else float("nan"))
        if i + 1 == 6000:
            grid = np.linspace(0, 600, 200)
            for a, nb in runs.items():
                snap[a] = [
                    nb.predict_proba_one({"amount": float(v), "hour": 14.0}).get(1, 0.0)
                    for v in grid
                ]
    fig = plt.figure(figsize=(16, 6.2))
    heading(
        fig,
        "WHAT THE NAIVE BAYES BELIEVES ABOUT A FRAUD",
        "at payment 3000 a fraud stops being a large payment and becomes a small one; with a forgetting factor the belief follows, without it the belief becomes a blur",
    )
    a1 = fig.add_axes([0.05, 0.12, 0.45, 0.6])
    t = np.arange(1, 6001)
    a1.plot(
        [0, 3000, 3000, 6000],
        [400, 400, 60, 60],
        color=INK,
        lw=1.3,
        ls="--",
        label="true average amount of a fraud",
    )
    a1.plot(t, mean[0.0], color=ORANGE, lw=2.6, label="alpha = 0 (never forgets)")
    a1.plot(t, mean[0.1], color=CYAN, lw=2.6, label="alpha = 0.1")
    a1.set_xlabel("payments")
    a1.set_ylabel("average amount of a fraud, as the classifier sees it")
    a1.legend(loc="center right", facecolor=PANEL, edgecolor=GRID, labelcolor=INK)
    a1.grid(True)
    a2 = fig.add_axes([0.58, 0.12, 0.4, 0.6])
    grid = np.linspace(0, 600, 200)
    a2.plot(grid, snap[0.0], color=ORANGE, lw=2.6, label="alpha = 0")
    a2.plot(grid, snap[0.1], color=CYAN, lw=2.6, label="alpha = 0.1")
    a2.axhline(0.5, color=INK, lw=0.8, ls=":")
    a2.set_xlabel("amount of the payment")
    a2.set_ylabel("probability of fraud, at payment 6000")
    a2.set_ylim(-0.03, 1.03)
    a2.set_title(
        "small payments are the frauds now", color=INK, loc="left", fontsize=12
    )
    a2.legend(loc="center right", facecolor=PANEL, edgecolor=GRID, labelcolor=INK)
    a2.grid(True)
    fig.savefig(HERE / "nb_learning.png", dpi=DPI, facecolor=BG)
    plt.close(fig)
    print(
        "nb final mean of the fraud amount: alpha 0 ->",
        round(mean[0.0][-1], 1),
        "alpha 0.1 ->",
        round(mean[0.1][-1], 1),
    )


def softmax_boundary():
    sm = OnlineSoftmaxRegression()
    marks = (1000, 3000, 3200, 3600, 6000)
    cmap = LinearSegmentedColormap.from_list("m", [CYAN, "#6a5acd", PINK])
    grid = np.linspace(0, 600, 200)
    curves = {}
    for i, x, y in stream():
        sm.learn_one(x, y)
        if i + 1 in marks:
            curves[i + 1] = [
                sm.predict_proba_one({"amount": float(v), "hour": 14.0}).get(1, 0.0)
                for v in grid
            ]
    fig, ax = plt.subplots(figsize=(16, 5.8))
    fig.subplots_adjust(top=0.74, bottom=0.13, left=0.06, right=0.985)
    heading(
        fig,
        "THE BORDER BETWEEN HONEST AND FRAUD, MOVING",
        "OnlineSoftmaxRegression: the probability of fraud given the amount of the payment, at five moments; the fraudsters change habit at payment 3000",
    )
    for k, m in enumerate(marks):
        ax.plot(
            grid,
            curves[m],
            color=cmap(k / (len(marks) - 1)),
            lw=2.8,
            label=f"after {m} payments",
        )
    ax.axhline(0.5, color=INK, lw=0.8, ls=":")
    ax.set_xlabel("amount of the payment")
    ax.set_ylabel("probability of fraud")
    ax.set_ylim(-0.03, 1.03)
    ax.legend(loc="center left", facecolor=PANEL, edgecolor=GRID, labelcolor=INK)
    ax.grid(True)
    fig.savefig(HERE / "softmax_boundary.png", dpi=DPI, facecolor=BG)
    plt.close(fig)


def drift_swap():
    det = CUSUMDriftDetector(reference="fixed", radius=20, ref_mult=5, two_sided=False)
    wrap = DriftAdaptiveClassifier(OnlineGaussianNB(), det, window=50, smooth_window=50)
    plain, fast = OnlineGaussianNB(), OnlineGaussianNB(alpha=0.1)
    hits = {"wrapper": [], "never forgets": [], "alpha=0.1": []}
    start, end, main_id = None, None, id(wrap._main)
    for i, x, y in stream():
        hits["wrapper"].append(wrap.predict_one(x) == y)
        hits["never forgets"].append(plain.predict_one(x) == y)
        hits["alpha=0.1"].append(fast.predict_one(x) == y)
        wrap.learn_one(x, y)
        plain.learn_one(x, y)
        fast.learn_one(x, y)
        if start is None and wrap._bg is not None:
            start = i
        if end is None and id(wrap._main) != main_id:
            end = i
    fig, ax = plt.subplots(figsize=(16, 5.8))
    fig.subplots_adjust(top=0.74, bottom=0.13, left=0.06, right=0.985)
    heading(
        fig,
        "THE SUBSTITUTE ON THE BENCH",
        "the drift-adaptive wrapper: the alarm starts a backup classifier that learns only the new payments, and it takes over when it is better; accuracy over the last 200 payments",
    )
    for name, col, lw in (
        ("never forgets", ORANGE, 1.8),
        ("alpha=0.1", MUTED, 1.8),
        ("wrapper", CYAN, 2.8),
    ):
        t, acc = rolling(hits[name], 200)
        ax.plot(
            t,
            acc,
            color=col,
            lw=lw,
            label=name if name != "wrapper" else "DriftAdaptiveClassifier",
        )
    ax.axvline(3000, color=INK, lw=1.3, ls=":")
    ax.text(
        2960, 0.3, "the fraudsters change habit", color=INK, fontsize=10.5, ha="right"
    )
    if start is not None and end is not None:
        ax.axvspan(start, end, color=PINK, alpha=0.12, lw=0)
        ax.text(
            (start + end) / 2,
            0.55,
            f"the backup learns\n(payments {start} to {end})",
            color=PINK,
            fontsize=10.5,
            ha="center",
        )
        ax.axvline(end, color=PINK, lw=1.6)
    ax.set_xlabel("payments")
    ax.set_ylabel("accuracy")
    ax.set_xlim(0, 6000)
    ax.set_ylim(-0.02, 1.04)
    ax.legend(loc="lower left", facecolor=PANEL, edgecolor=GRID, labelcolor=INK)
    ax.grid(True)
    fig.savefig(HERE / "drift_swap.png", dpi=DPI, facecolor=BG)
    plt.close(fig)
    print("drift: backup starts", start, "takes over", end)


if __name__ == "__main__":
    nb_learning()
    softmax_boundary()
    drift_swap()
