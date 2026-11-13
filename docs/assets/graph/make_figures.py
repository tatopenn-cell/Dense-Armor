"""Generate the figures of docs/graph/.

Run from the repository root:

    python docs/assets/graph/make_figures.py

Every curve is computed with the functions of ``dense_armor.core.graph``.
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from dense_armor.core.graph import (  # noqa: E402
    Graph,
    apply_filter,
    chebyshev_coefficients,
    chebyshev_filter,
    eigendecomposition,
    graph_anomaly_score,
    heat_kernel,
    low_pass,
    page_rank,
)

OUT = Path(__file__).resolve().parent
C = ["#2a6fbb", "#d9822b", "#3a9a5b", "#b04a8a"]


def path(n):
    return Graph(n, [(i, i + 1, 1.0) for i in range(n - 1)])


def eigenvectors():
    g = path(10)
    w, U = eigendecomposition(g.laplacian)
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    for k in range(4):
        ax.plot(
            range(10),
            U[:, k],
            marker="o",
            color=C[k],
            label=f"eigenvalue {abs(w[k]):.3f}",
        )
    ax.set_xlabel("node")
    ax.set_ylabel("eigenvector")
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(OUT / "eigenvectors.png", dpi=140)
    plt.close(fig)


def filters():
    g = path(10)
    r = np.zeros(10)
    r[4] = 5.0
    fig, (a, b) = plt.subplots(1, 2, figsize=(9.0, 3.2))
    a.plot(r, "k.", label="signal")
    a.plot(
        heat_kernel(g.laplacian, 1.0) @ r,
        marker="o",
        color=C[0],
        label="heat kernel, tau=1",
    )
    a.plot(
        low_pass(g.laplacian, 0.5) @ r,
        marker="s",
        color=C[1],
        label="low-pass, cutoff 0.5",
    )
    a.set_xlabel("node")
    a.legend(frameon=False, fontsize=8)
    exact = apply_filter(r, g.laplacian, lambda lam: np.exp(-lam))
    ks = range(1, 13)
    errs = []
    for k in ks:
        c = chebyshev_coefficients(lambda lam: np.exp(-lam), 4.0, k)
        errs.append(np.abs(chebyshev_filter(r, g.laplacian, c, 4.0) - exact).max())
    b.semilogy(list(ks), errs, marker="o", color=C[2])
    b.set_xlabel("Chebyshev order K")
    b.set_ylabel("max error vs exact filter")
    fig.tight_layout()
    fig.savefig(OUT / "filters.png", dpi=140)
    plt.close(fig)


def anomaly():
    g = path(6)
    x = np.array([0.1, -0.1, 0.0, 6.0, 0.1, -0.1])
    sc = graph_anomaly_score(g, x)
    pr = page_rank(g)
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    idx = np.arange(6)
    ax.bar(idx - 0.2, [sc[i] for i in idx], 0.4, color=C[1], label="anomaly score")
    ax.bar(idx + 0.2, [pr[i] for i in idx], 0.4, color=C[0], label="PageRank")
    ax.set_xlabel("node")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "anomaly.png", dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    eigenvectors()
    filters()
    anomaly()
