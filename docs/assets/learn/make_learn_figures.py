"""Dark figures for the Adaptive learners pages, drawn with the library itself and the data of the
first example of each page.

Generates in the same folder:
- calibration_reliability.png   a naive Bayes with and without OnlinePlattScaling on a fraud score
- metric_weights.png            the weight that OASIS, LEGO and POLA give to each column of a customer
- classifier_drift.png          three classifiers when the meaning of the amount of a payment flips

Run from the repo root:
    python docs/assets/learn/make_learn_figures.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from dense_armor.core.online.drift.detector import CUSUMDriftDetector
from dense_armor.core.online.learn.calibration import OnlinePlattScaling
from dense_armor.core.online.learn.metric_learning import LEGO, OASIS, POLA
from dense_armor.core.online.learn.online_classifiers import (
    DriftAdaptiveClassifier,
    OnlineGaussianNB,
)

HERE = Path(__file__).resolve().parent
BG, PANEL, GRID = "#0b1118", "#111a24", "#26364a"
INK, MUTED = "#e8fbff", "#8fa3b8"
CYAN, ORANGE, PINK = "#3ec6e8", "#ffa21f", "#ff2d87"

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


def rolling(hits, w):
    c = np.cumsum(np.insert(np.asarray(hits, dtype=float), 0, 0.0))
    return np.arange(w, len(hits) + 1), (c[w:] - c[:-w]) / w


def calibration():
    rng = np.random.default_rng(0)
    raw, cal = OnlineGaussianNB(), OnlinePlattScaling(OnlineGaussianNB())
    pr, pc, ys = [], [], []
    for _ in range(4000):
        fraud = int(rng.random() < 0.5)
        shared = rng.normal(0, 1)
        x = {
            f"s{k}": 0.6 * (2 * fraud - 1) + shared + 0.3 * rng.normal()
            for k in range(6)
        }
        pr.append(raw.predict_proba_one(x).get(1, 0.5))
        pc.append(cal.predict_proba_one(x).get(1, 0.5))
        ys.append(fraud)
        raw.learn_one(x, fraud)
        cal.learn_one(x, fraud)
    pr, pc, ys = np.array(pr), np.array(pc), np.array(ys)

    def ll(p):
        p = np.clip(p, 1e-15, 1 - 1e-15)
        return -(ys * np.log(p) + (1 - ys) * np.log(1 - p))

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(16, 5.8))
    fig.subplots_adjust(top=0.74, bottom=0.12, left=0.06, right=0.985, wspace=0.18)
    heading(
        fig,
        "CALIBRATION MAKES THE NUMBERS HONEST",
        "a fraud score from six copies of the same signal: a naive Bayes is far too sure; OnlinePlattScaling brings its numbers back to the truth",
    )
    edges = np.linspace(0, 1, 11)
    for p, col, lab in (
        (pr, ORANGE, "OnlineGaussianNB"),
        (pc, CYAN, "with OnlinePlattScaling"),
    ):
        idx = np.digitize(p, edges[1:-1])
        xs, fs = [], []
        for b in range(10):
            m = idx == b
            if m.sum() >= 20:
                xs.append(p[m].mean())
                fs.append(ys[m].mean())
        a1.plot(xs, fs, "o-", color=col, lw=2.2, ms=7, label=lab)
    a1.plot(
        [0, 1],
        [0, 1],
        color=INK,
        lw=1.2,
        ls="--",
        label="honest: said 80 %, happens 80 %",
    )
    a1.set_xlabel("probability of fraud announced by the model")
    a1.set_ylabel("how often it really was fraud")
    a1.legend(loc="upper left", facecolor=PANEL, edgecolor=GRID, labelcolor=INK)
    a1.grid(True)
    n = np.arange(1, len(ys) + 1)
    for p, col, lab in (
        (pr, ORANGE, "OnlineGaussianNB"),
        (pc, CYAN, "with OnlinePlattScaling"),
    ):
        run = np.cumsum(ll(p)) / n
        a2.plot(n, run, color=col, lw=2.4, label=f"{lab}: {run[-1]:.3f}")
    a2.set_xscale("log")
    a2.set_xlabel("payments")
    a2.set_ylabel("average log-loss so far (lower is better)")
    a2.legend(loc="upper right", facecolor=PANEL, edgecolor=GRID, labelcolor=INK)
    a2.grid(True)
    fig.savefig(HERE / "calibration_reliability.png", dpi=150, facecolor=BG)
    plt.close(fig)
    print(
        "calibration log-loss",
        round(float(ll(pr).mean()), 3),
        round(float(ll(pc).mean()), 3),
    )


def metric_weights():
    rng = np.random.default_rng(0)

    def customer(young):
        return {
            "age": rng.normal(-1.0 if young else 1.0, 0.5),
            "spend": rng.normal(0, 1),
            "balance": rng.normal(0, 1),
            "visits": rng.normal(0, 1),
        }

    oasis, lego, pola = OASIS(C=0.1), LEGO(eta=0.1), POLA()
    labels = rng.random(200) < 0.5
    X = np.array([list(customer(g).values()) for g in labels])
    near, far = LEGO.percentile_targets(X, labels)
    for _ in range(2000):
        young = rng.random() < 0.5
        oasis.learn_triplet(customer(young), customer(young), customer(not young))
        if rng.random() < 0.5:
            lego.learn_pair(customer(young), customer(young), near)
            pola.learn_pair(customer(young), customer(young), +1)
        else:
            lego.learn_pair(customer(young), customer(not young), far)
            pola.learn_pair(customer(young), customer(not young), -1)
    cols = ["age", "balance", "spend", "visits"]
    w = {
        "OASIS": np.diag(oasis.W.T @ oasis.W),
        "LEGO": np.diag(lego.A),
        "POLA": np.diag(pola.A),
    }
    w = {k: v / v.sum() for k, v in w.items()}
    fig, ax = plt.subplots(figsize=(16, 5.6))
    fig.subplots_adjust(top=0.74, bottom=0.13, left=0.06, right=0.985)
    heading(
        fig,
        "WHAT THREE LEARNERS DECIDE TO LOOK AT",
        "four columns for each customer: only the age tells the groups apart; the share of the distance that each learner gives to each column (Euclidean: 25 % each)",
    )
    x = np.arange(len(cols))
    width = 0.24
    for k, (name, col) in enumerate(
        (("OASIS", CYAN), ("LEGO", ORANGE), ("POLA", PINK))
    ):
        bars = ax.bar(x + (k - 1) * width, w[name] * 100, width, color=col, label=name)
        for b, v in zip(bars, w[name] * 100):
            ax.text(
                b.get_x() + b.get_width() / 2,
                v + 1.2,
                f"{v:.0f}%",
                ha="center",
                color=INK,
                fontsize=10,
            )
    ax.axhline(25, color=INK, lw=1.2, ls="--")
    ax.text(3.45, 26.5, "Euclidean: 25 % each", color=INK, fontsize=10.5, ha="right")
    ax.set_xticks(x)
    ax.set_xticklabels(cols, fontsize=12)
    ax.set_ylabel("share of the distance [%]")
    ax.set_ylim(0, 100)
    ax.legend(loc="upper right", facecolor=PANEL, edgecolor=GRID, labelcolor=INK)
    ax.grid(True, axis="y")
    fig.savefig(HERE / "metric_weights.png", dpi=150, facecolor=BG)
    plt.close(fig)
    print({k: np.round(v * 100).astype(int).tolist() for k, v in w.items()})


def classifier_drift():
    rng = np.random.default_rng(1)
    models = {
        "never forgets": (OnlineGaussianNB(), ORANGE),
        "alpha=0.1": (OnlineGaussianNB(alpha=0.1), MUTED),
        "drift-adaptive": (
            DriftAdaptiveClassifier(
                OnlineGaussianNB(),
                CUSUMDriftDetector(
                    reference="fixed", radius=20, ref_mult=5, two_sided=False
                ),
                window=50,
                smooth_window=50,
            ),
            CYAN,
        ),
    }
    hits = {k: [] for k in models}
    for i in range(6000):
        fraud = int(rng.random() < 0.5)
        big = fraud if i < 3000 else 1 - fraud
        x = {
            "amount": rng.normal(400 if big else 60, 60 if big else 20),
            "hour": rng.normal(14, 3),
        }
        for k, (m, _) in models.items():
            hits[k].append(m.predict_one(x) == fraud)
            m.learn_one(x, fraud)
    fig, ax = plt.subplots(figsize=(16, 5.8))
    fig.subplots_adjust(top=0.74, bottom=0.12, left=0.06, right=0.985)
    heading(
        fig,
        "A CLASSIFIER THAT NOTICES WHEN THE WORLD CHANGES",
        "at payment 3000 the fraudsters change habit and the meaning of the amount flips; accuracy over the last 200 payments",
    )
    for k, (_, col) in models.items():
        t, acc = rolling(hits[k], 200)
        ax.plot(t, acc, color=col, lw=2.6 if col == CYAN else 1.8, label=k)
    ax.axvline(3000, color=INK, lw=1.4, ls=":")
    ax.text(
        2960, 0.25, "the fraudsters change habit", color=INK, fontsize=11, ha="right"
    )
    ax.set_xlabel("payments")
    ax.set_ylabel("accuracy")
    ax.set_xlim(0, 6000)
    ax.set_ylim(0.0, 1.03)
    ax.legend(loc="lower left", facecolor=PANEL, edgecolor=GRID, labelcolor=INK)
    ax.grid(True)
    fig.savefig(HERE / "classifier_drift.png", dpi=150, facecolor=BG)
    plt.close(fig)
    for k in models:
        h = np.array(hits[k], float)
        print(
            "drift",
            k,
            h[500:3000].mean().round(3),
            h[3000:3200].mean().round(3),
            h[5000:5900].mean().round(3),
        )


if __name__ == "__main__":
    calibration()
    metric_weights()
    classifier_drift()
