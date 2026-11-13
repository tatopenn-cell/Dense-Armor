"""PageRank, random walk with restart, graph anomaly score.

References:
    Page, Brin, Motwani, Winograd (1999), "The PageRank Citation
    Ranking", Stanford InfoLab (bibliographic only).
    Tong, Faloutsos, Pan (2006), "Fast Random Walk with Restart and
    Its Applications", ICDM (bibliographic only).
    Wang (2026), "AgentTrace: Causal Graph Tracing for Root Cause
    Analysis in Deployed Multi-Agent Systems", arXiv:2603.14688
    (bibliographic only: no PDF in ``papers/``, no section verified).
    Akoglu, Tong, Koutra (2015), "Graph based anomaly detection and
    description: a survey", DMKD (bibliographic only).
"""

from __future__ import annotations

import numpy as np


def _nx():
    import networkx as nx  # type: ignore[import-untyped]

    return nx


def page_rank(graph, alpha: float = 0.85, personalization: dict | None = None) -> dict:
    """PageRank of ``graph``; wrapper of ``networkx.pagerank``."""
    if not (0 < alpha < 1):
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    if hasattr(graph, "to_networkx"):
        graph = graph.to_networkx()
    return _nx().pagerank(graph, alpha=alpha, personalization=personalization)


def random_walk_restart(graph, seed, alpha: float = 0.85) -> dict:
    """PageRank with a restart concentrated on ``seed`` (RWR)."""
    if not (0 < alpha < 1):
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    if hasattr(graph, "to_networkx"):
        graph = graph.to_networkx()
    if seed not in graph.nodes:
        raise ValueError(f"unknown seed {seed!r}")
    pers = {n: (1.0 if n == seed else 0.0) for n in graph.nodes}
    return _nx().pagerank(graph, alpha=alpha, personalization=pers)


def graph_anomaly_score(graph, signal, alpha: float = 0.85, tau: float = 1.0) -> dict:
    """Score every node by combining a local residual and PageRank.

    The residual is the difference between the signal and its heat
    kernel smoothing, ``r = x - exp(-tau L) x``: it is large where the
    signal deviates from its local trend (a localized anomaly), and
    small where the signal follows the local trend. The GFT magnitude
    is **not** used here: it is a frequency-domain transform, not a
    node-local score.

    For each node ``i``:
    ``score[i] = zscore(|r[i]|) + page_rank[i]``.

    Args:
        graph: a :class:`Graph`.
        signal: signal of length ``n_nodes``.
        alpha: PageRank damping.
        tau: heat kernel time (``exp(-tau L)``).

    Returns:
        dict ``{node_index: score}``.

    Choices stated explicitly:
        The combination ``zscore(|x - exp(-tau L) x|) + page_rank`` is
        a design choice of this module. The papers in ``papers/`` describe the
        heat kernel and PageRank separately; neither defines a
        node-level anomaly score. A different residual (e.g. the
        median deviation instead of the heat kernel smoothing) would
        be a different choice.
    """
    from dense_armor.core.graph.core import Graph
    from dense_armor.core.graph.filters import heat_kernel

    if hasattr(graph, "to_networkx"):
        G_nx = graph.to_networkx()
        n = G_nx.number_of_nodes()
        edges = [
            (min(u, v), max(u, v), float(d.get("weight", 1.0)))
            for u, v, d in G_nx.edges(data=True)
            if u != v
        ]
    else:
        n = graph.n_nodes
        edges = graph.edges
    g = Graph(n, edges)
    x = np.asarray(signal, dtype=float)
    H = heat_kernel(g.laplacian, tau)
    residual = x - H @ x
    mag = np.abs(residual)
    mu = mag.mean()
    sd = mag.std() if mag.std() > 0 else 1.0
    z = (mag - mu) / sd
    pr = page_rank(g, alpha=alpha)
    pr_vec = np.array([pr[i] for i in range(n)])
    return {i: float(z[i] + pr_vec[i]) for i in range(n)}
