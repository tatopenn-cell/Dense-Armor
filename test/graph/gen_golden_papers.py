"""Generate the golden files of the paper primitives.

Run once:
    python test/graph/gen_golden_papers.py

The files are committed. ``test_golden_papers.py`` loads them and checks
that the current code reproduces them on every platform. Every input is
built from a seed written here, and the iterative solvers run a fixed
number of iterations (``tol=0``), so that the count does not depend on the
platform.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from dense_armor.core.graph import (
    AdaCGP,
    Dataflow,
    Graph,
    SpectralWavelet,
    learn_graph,
    recover_graph_signal,
    structural_change,
)

GOLD = Path(__file__).parent / "golden"


def ring(n: int) -> Graph:
    return Graph(n, [(i, (i + 1) % n, 1.0) for i in range(n)])


def path(n: int) -> Graph:
    return Graph(n, [(i, i + 1, 1.0) for i in range(n - 1)])


def wavelet_arrays():
    w = SpectralWavelet(ring(8), [1.0, 0.5], [1.0, 0.3], [1.0, 1.5])
    x = np.arange(8.0)
    return {
        "scaling_response": w.scaling_response,
        "wavelet_responses": w.wavelet_responses,
        "scaling": w.scaling(x),
        "wavelets": w.wavelets(x),
    }


def recovery_array():
    n, p = 12, 10
    i, k = np.arange(n)[:, None], np.arange(p)[None, :]
    x0 = np.sin(0.4 * i) * np.cos(0.2 * k)
    y = x0 + 0.1 * np.random.default_rng(0).standard_normal((n, p))
    return recover_graph_signal(
        y,
        path(n),
        lam=0.05,
        eps=0.9 * 0.1 * np.sqrt(n * p),
        temporal="l2",
        max_iter=500,
        tol=0.0,
    )


def learning_array():
    x = np.random.default_rng(0).standard_normal((6, 10))
    _, w = learn_graph(
        x, alpha=1.0, beta=0.1, max_iter=3000, tol=0.0, return_weights=True
    )
    return w


def adacgp_array():
    w = np.zeros((4, 4))
    for i in range(4):
        w[i, (i + 1) % 4] = w[(i + 1) % 4, i] = 1.0
    h = 0.2 * np.eye(4) + 0.3 * w
    rng = np.random.default_rng(0)
    m = AdaCGP(4, order=2, forgetting=0.99, mu=0.01, gamma=0.1, alpha=0.001)
    x = np.zeros(4)
    for _ in range(500):
        x = h @ x + rng.standard_normal(4)
        m.learn_one(x)
    return np.stack(m.filters)


def change_dict():
    a = Graph(4, [(0, 1, 1.0), (1, 2, 2.0), (2, 3, 1.0)])
    b = Graph(4, [(0, 1, 1.0), (1, 2, 3.0), (1, 3, 1.0)])
    c = structural_change(a, b)
    return {
        "added": [list(e) for e in c["added"]],
        "removed": [list(e) for e in c["removed"]],
        "reweighted": [list(e) for e in c["reweighted"]],
        "topology_change": c["topology_change"],
        "weight_change": c["weight_change"],
    }


class _Inc:
    def learn_one(self, x, y=None):
        return self

    def transform_one(self, x):
        return {k: float(v) + 1.0 for k, v in dict(x).items()}


def dataflow_dict():
    d = Dataflow().source("a", _Inc()).then("b", _Inc(), "a").then("c", _Inc(), "b")
    return {"lineage": d.lineage("c"), "edges": [list(e) for e in d.edges()]}


def main() -> None:
    GOLD.mkdir(exist_ok=True)
    np.savez(GOLD / "papers_wavelet_ring8.npz", **wavelet_arrays())
    np.save(GOLD / "papers_recovery_denoised.npy", recovery_array())
    np.save(GOLD / "papers_learning_weights.npy", learning_array())
    np.save(GOLD / "papers_adacgp_filters.npy", adacgp_array())
    (GOLD / "papers_change.json").write_text(json.dumps(change_dict()))
    (GOLD / "papers_dataflow.json").write_text(json.dumps(dataflow_dict()))


if __name__ == "__main__":
    main()
