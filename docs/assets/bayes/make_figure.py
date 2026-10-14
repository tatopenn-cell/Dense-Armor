"""Generate the figures of docs/bayes/.

Run from the repository root:

    python docs/assets/bayes/make_figure.py
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from dense_armor.utility.bayes import MultinomialNB, BernoulliNB

HERE = Path(__file__).resolve().parent

DOCS = [
    ({"a": 2, "b": 1, "c": 0}, "X"),
    ({"a": 1, "b": 0, "c": 2}, "X"),
    ({"a": 0, "b": 2, "c": 1}, "Y"),
    ({"a": 0, "b": 3, "c": 1}, "Y"),
]


def fig_word_probabilities():
    """The class-conditional probability of each word, in the two models."""
    mnb = MultinomialNB(alpha=1.0)
    bnb = BernoulliNB(alpha=1.0)
    for x, y in DOCS:
        mnb.learn_one(x, y)
        bnb.learn_one(x, y)

    words = ["a", "b", "c"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), facecolor="#fafafa")
    for ax, model, title in [
        (axes[0], mnb, "Multinomial: fraction of each word in the class"),
        (axes[1], bnb, "Bernoulli: probability that each word appears at all"),
    ]:
        ax.set_facecolor("#fafafa")
        x = np.arange(len(words))
        w = 0.35
        # class X
        counts_X = model._counts["X"] if hasattr(model, "_counts") else model._present["X"]
        counts_Y = model._counts["Y"] if hasattr(model, "_counts") else model._present["Y"]
        if model is mnb:
            n_X = model._class_totals["X"]
            n_Y = model._class_totals["Y"]
            alpha = 1.0
            theta_X = (counts_X + alpha) / (n_X + alpha * len(words))
            theta_Y = (counts_Y + alpha) / (n_Y + alpha * len(words))
        else:
            n_X = model._class_n["X"]
            n_Y = model._class_n["Y"]
            theta_X = (1.0 + counts_X) / (2.0 + n_X)
            theta_Y = (1.0 + counts_Y) / (2.0 + n_Y)
        ax.bar(x - w/2, theta_X, w, label="class X", color="#2c6fa6")
        ax.bar(x + w/2, theta_Y, w, label="class Y", color="#c0392b")
        ax.set_xticks(x)
        ax.set_xticklabels(words)
        ax.set_ylabel("probability")
        ax.set_title(title, fontsize=11, weight="bold")
        ax.legend(fontsize=10)
        ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(HERE / "word_probabilities.png", dpi=140, facecolor="#fafafa")
    plt.close(fig)
    print("  word_probabilities.png")


def fig_absent_word():
    """Bernoulli counts absence; multinomial does not."""
    fig, ax = plt.subplots(figsize=(9, 5), facecolor="#fafafa")
    ax.set_facecolor("#fafafa")
    mnb = MultinomialNB(alpha=1.0)
    bnb = BernoulliNB(alpha=1.0)
    for x, y in DOCS:
        mnb.learn_one(x, y)
        bnb.learn_one(x, y)

    # Query where the word "a" is absent
    q = {"a": 0, "b": 1, "c": 1}
    pm = mnb.predict_proba_one(q)
    pb = bnb.predict_proba_one(q)
    labels = ["Multinomial", "Bernoulli"]
    xX = [pm.get("X", 0.0), pb.get("X", 0.0)]
    xY = [pm.get("Y", 0.0), pb.get("Y", 0.0)]
    x = np.arange(2)
    w = 0.35
    ax.bar(x - w/2, xX, w, label="class X", color="#2c6fa6")
    ax.bar(x + w/2, xY, w, label="class Y", color="#c0392b")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("P(class | a=0, b=1, c=1)")
    ax.set_title("Same query, two event models. Bernoulli counts absence,\nMultinomial does not.",
                 fontsize=11, weight="bold")
    for i, (vx, vy) in enumerate(zip(xX, xY)):
        ax.text(i - w/2, vx + 0.01, f"{vx:.3f}", ha="center", fontsize=10)
        ax.text(i + w/2, vy + 0.01, f"{vy:.3f}", ha="center", fontsize=10)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(HERE / "absent_word.png", dpi=140, facecolor="#fafafa")
    plt.close(fig)
    print("  absent_word.png")


if __name__ == "__main__":
    print("Generating docs/assets/bayes/")
    fig_word_probabilities()
    fig_absent_word()
    print("Done.")
