from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

from dense_armor.usable.catalog import get_catalog

BG = "#0d131a"
BAND = "#0f1822"
CARD = "#101a24"
CYAN = "#4cc9f0"
ORANGE = "#ff9e00"
PINK = "#f72585"
TEXT = "#e0fbfc"
MUTED = "#94a3b8"
LINE = "#1e293b"
OUT = Path(__file__).resolve().parent
REPO = OUT.parents[2]

GROUPS = (
    (
        "01",
        "PREDICT",
        "a label or a number from the features of a sample",
        CYAN,
        (
            ("learn", "Adaptive learners"),
            ("tree", "Trees"),
            ("forest", "Forests"),
            ("ensemble", "Ensembles"),
            ("linear", "Linear models"),
            ("bayes", "Naive Bayes"),
            ("neighbors", "Neighbours"),
            ("multiclass", "Multiclass"),
            ("multioutput", "Multi-output"),
        ),
    ),
    (
        "02",
        "DETECT",
        "what is strange, what has changed, what to keep out of the model",
        PINK,
        (
            ("anomaly", "Anomaly detection"),
            ("drift", "Drift detection"),
            ("protect", "Protection"),
        ),
    ),
    (
        "03",
        "DECIDE AND SUGGEST",
        "choose an action, ask for a label, recommend an item",
        ORANGE,
        (
            ("bandit", "Bandits"),
            ("active", "Active learning"),
            ("reco", "Recommenders"),
            ("facto", "Factorization machines"),
        ),
    ),
    (
        "04",
        "MODEL THE STREAM",
        "time, distributions, the optimizer, the groups and the rules",
        CYAN,
        (
            ("timeseries", "Time series"),
            ("proba", "Distributions"),
            ("optim", "Optimizers"),
            ("cluster", "Clustering"),
            ("rules", "Rules"),
        ),
    ),
)


def counts():
    c = Counter()
    for i in get_catalog(REPO):
        p = i["module"].split(".")
        if len(p) > 3 and p[1] == "core" and p[2] == "online":
            c[p[3]] += 1
    return c


def draw():
    c = counts()
    named = [a for _, _, _, _, cards in GROUPS for a, _ in cards]
    assert sorted(named) == sorted(c), (
        sorted(set(c) - set(named)),
        sorted(set(named) - set(c)),
    )
    total = sum(c.values())
    fig = plt.figure(figsize=(13.5, 8.2), facecolor=BG)
    fig.text(
        0.5,
        0.955,
        "THE MAP OF ONLINE LEARNING",
        ha="center",
        va="center",
        fontsize=22,
        fontweight="bold",
        color=TEXT,
    )
    fig.text(
        0.5,
        0.918,
        f"{len(c)} areas and {total} classes, in four groups: every one learns a sample at a time",
        ha="center",
        va="center",
        fontsize=10.5,
        color=CYAN,
    )
    ax = fig.add_axes([0.0, 0.03, 1.0, 0.875], facecolor=BG)
    ax.set_xlim(0, 135)
    ax.set_ylim(0, 84)
    ax.axis("off")
    ys = (66, 45, 24, 3)
    for (num, title, note, color, cards), y in zip(GROUPS, ys, strict=True):
        ax.add_patch(
            FancyBboxPatch(
                (1, y),
                133,
                17,
                boxstyle="round,pad=0.2,rounding_size=1.4",
                fc=BAND,
                ec=LINE,
                lw=1.2,
            )
        )
        ax.text(
            2.4, y + 14.8, num, color=PINK, fontsize=11, fontweight="bold", va="center"
        )
        ax.text(
            6.0,
            y + 14.8,
            title,
            color="#ffffff",
            fontsize=10.5,
            fontweight="bold",
            va="center",
        )
        ax.text(
            132.6, y + 14.8, note, color=MUTED, fontsize=8.2, ha="right", va="center"
        )
        w, gap = 13.0, 1.2
        for i, (area, name) in enumerate(cards):
            x = 2.4 + i * (w + gap)
            ax.add_patch(
                FancyBboxPatch(
                    (x, y + 1.6),
                    w,
                    10.6,
                    boxstyle="round,pad=0.2,rounding_size=1.0",
                    fc=CARD,
                    ec=color,
                    lw=1.5,
                )
            )
            label = name.replace(" ", chr(10), 1) if len(name) > 13 else name
            ax.text(
                x + 0.9,
                y + 8.6 if chr(10) in label else y + 9.6,
                label,
                color=TEXT,
                fontsize=9,
                fontweight="bold",
                va="center",
                linespacing=1.15,
            )
            ax.text(
                x + 0.9,
                y + 3.9,
                f"{c[area]} class" + ("" if c[area] == 1 else "es"),
                color=color,
                fontsize=9,
                fontweight="bold",
                va="center",
            )
    fig.text(
        0.5,
        0.012,
        "counts: classes in the catalog of the library (dense_armor.usable.catalog), drawn by docs/assets/online/make_overview_figure.py",
        ha="center",
        fontsize=8.6,
        color=MUTED,
    )
    fig.savefig(OUT / "overview.png", dpi=200, facecolor=BG)
    print(total, dict(c))


if __name__ == "__main__":
    draw()
