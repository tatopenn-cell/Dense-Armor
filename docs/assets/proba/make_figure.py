"""Generate the figures of docs/proba/.

Run from the repository root:

    python docs/assets/proba/make_figure.py
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from dense_armor.utility.proba import Beta, Gaussian, Multinomial

HERE = Path(__file__).resolve().parent


def fig_beta():
    """Beta prior and two posteriors after observations."""
    xs = np.linspace(0.001, 0.999, 400)
    fig, ax = plt.subplots(figsize=(9, 5), facecolor="#fafafa")
    ax.set_facecolor("#fafafa")
    for a, b, label, color in [
        (1, 1, "prior Beta(1, 1)", "#888888"),
        (4, 2, "after 3 heads, 1 tail", "#2c6fa6"),
        (11, 4, "after 10 heads, 3 tails", "#c0392b"),
    ]:
        d = Beta(float(a), float(b))
        ys = [d.pdf(float(x)) for x in xs]
        ax.plot(xs, ys, lw=2.2, color=color, label=label)
    ax.set_xlabel("probability of heads")
    ax.set_ylabel("density")
    ax.set_title("The beta prior becomes the posterior, one toss at a time",
                 fontsize=12, weight="bold")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(HERE / "beta_prior_posterior.png", dpi=140, facecolor="#fafafa")
    plt.close(fig)
    print("  beta_prior_posterior.png")


def fig_gaussian():
    """Welford running mean and variance over 200 samples from N(2, 0.5)."""
    rng = np.random.default_rng(0)
    data = rng.normal(2.0, 0.5, size=200)
    g = Gaussian()
    means, variances = [], []
    for v in data:
        g.update(float(v))
        means.append(g.mean())
        variances.append(g.variance() if g.n() >= 2 else np.nan)
    fig, ax1 = plt.subplots(figsize=(9, 5), facecolor="#fafafa")
    ax1.set_facecolor("#fafafa")
    ax1.plot(range(1, 201), means, color="#2c6fa6", lw=2, label="running mean")
    ax1.axhline(2.0, color="#888888", ls="--", lw=1.4, label="true mean 2.0")
    ax1.set_xlabel("sample")
    ax1.set_ylabel("mean", color="#2c6fa6")
    ax1.tick_params(axis="y", labelcolor="#2c6fa6")
    ax2 = ax1.twinx()
    ax2.plot(range(1, 201), variances, color="#c0392b", lw=2, label="running variance")
    ax2.axhline(0.25, color="#27ae60", ls="--", lw=1.4, label="true variance 0.25")
    ax2.set_ylabel("variance", color="#c0392b")
    ax2.tick_params(axis="y", labelcolor="#c0392b")
    ax1.set_title("Welford: mean and variance of the stream, one sample at a time",
                  fontsize=12, weight="bold")
    fig.tight_layout()
    fig.savefig(HERE / "gaussian_welford.png", dpi=140, facecolor="#fafafa")
    plt.close(fig)
    print("  gaussian_welford.png")


def fig_dirichlet():
    """Dirichlet: the counters are the parameters, the mean follows them."""
    fig, ax = plt.subplots(figsize=(9, 5), facecolor="#fafafa")
    ax.set_facecolor("#fafafa")
    x = np.arange(3)
    w = 0.28
    for j, (label, alpha) in enumerate([
        ("prior (1, 1, 1)", [1, 1, 1]),
        ("after (3, 1, 0)", [4, 2, 1]),
        ("after (8, 3, 1)", [9, 4, 2]),
    ]):
        m = Multinomial(alpha=[float(a) for a in alpha])
        ax.bar(x + (j - 1) * w, m.mean(), w, label=label)
    ax.set_xticks(x)
    ax.set_xticklabels(["A", "B", "C"])
    ax.set_ylabel("posterior mean probability")
    ax.set_title("The counters ARE the parameters: every observation moves one bar",
                 fontsize=12, weight="bold")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(HERE / "dirichlet_counts.png", dpi=140, facecolor="#fafafa")
    plt.close(fig)
    print("  dirichlet_counts.png")


if __name__ == "__main__":
    print("Generating docs/assets/proba/")
    fig_beta()
    fig_gaussian()
    fig_dirichlet()
    print("Done.")
