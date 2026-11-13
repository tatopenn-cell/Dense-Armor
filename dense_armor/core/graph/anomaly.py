"""PageRank, random walk with restart, graph anomaly score.

References:
    Page, Brin, Motwani, Winograd (1999), "The PageRank Citation
    Ranking", Stanford InfoLab (bibliographic only).
    Tong, Faloutsos, Pan (2006), "Fast Random Walk with Restart and
    Its Applications", ICDM (bibliographic only).
    Khan, Farea (2026), "eIRWR: Enhanced Iterative Random Walk with
    Restart for Scalable Root Cause Analysis in Microservices",
    arXiv:2608.08073, Sections 3.1-3.5 and Algorithm 1: ``root_cause_walk``.
    Lin, Chang, Wang, Wang, Peng (2024), "Root Cause Analysis In
    Microservice Using Neural Granger Causal Discovery", arXiv:2402.01140
    (AAAI 2024): PageRank with a personalization vector ranks the root
    causes (abstract only).
    Wang (2026), "AgentTrace: Causal Graph Tracing for Root Cause
    Analysis in Deployed Multi-Agent Systems", arXiv:2603.14688
    (abstract only).
    Akoglu, Tong, Koutra (2015), "Graph based anomaly detection and
    description: a survey", DMKD (bibliographic only).
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp


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
    sd = mag.std()
    z = (mag - mu) / sd if sd > 1e-12 * max(float(np.abs(x).max()), 1.0) else 0.0 * mag
    pr = page_rank(g, alpha=alpha)
    pr_vec = np.array([pr[i] for i in range(n)])
    return {i: float(z[i] + pr_vec[i]) for i in range(n)}


def _adaptive_exponent(b: np.ndarray, q: float) -> float:
    """``q_adapt = 1 + (q - 1) min(1, max(0, (sigma - 5) / 15))``, Eq. (10).

    ``sigma`` is the ratio of the largest belief to the mean belief: a weak
    signal (``sigma <= 5``) keeps the plain teleportation, a strong one
    (``sigma >= 20``) takes the full exponent ``q``.
    """
    mean = float(b.mean())
    sigma = float(b.max()) / mean if mean > 0 else 1.0
    return 1.0 + (q - 1.0) * min(1.0, max(0.0, (sigma - 5.0) / 15.0))


def _row_normalise(M):
    rs = np.asarray(M.sum(axis=1)).ravel()
    inv = np.divide(1.0, rs, out=np.zeros_like(rs), where=rs > 0)
    return (sp.diags(inv) @ M).tocsr()


def _enhanced_transition(Mbase, W, C: np.ndarray, rho: float):
    """The transition matrix of Eq. (5)-(8): beliefs, backward edges, self-loops."""
    n = W.shape[0]
    Mc = (Mbase @ sp.diags(C)).tocsr()
    P = (W != 0).astype(float).tocsr()
    nr = (P - P.multiply(P.T)).tocsr()
    nr.eliminate_zeros()
    nr = nr.tocoo()
    A_bwd = sp.csr_matrix((rho * C[nr.row], (nr.col, nr.row)), shape=(n, n))
    maxrow = Mc.max(axis=1).toarray().ravel()
    A_self = sp.diags(np.maximum(0.0, C - maxrow))
    return _row_normalise(Mc + A_bwd + A_self)


def root_cause_walk(
    weights,
    anomaly,
    resilience: float = 0.0,
    beta: float = 0.0,
    alpha: float = 0.15,
    rho: float = 0.0,
    q: float = 1.0,
    momentum: float = 0.1,
    eps: float = 1e-6,
    outer: int = 1,
    enhanced: bool = True,
    max_iter: int = 10000,
) -> np.ndarray:
    """Rank the nodes of a dependency graph by how likely each is the root cause.

    The walk starts from the observed anomaly and runs backward along the
    dependencies: the score of a caller collects the scores of the nodes it
    calls. With ``enhanced=False`` it is the iterative random walk with
    restart of Eq. (3): ``r <- (1 - alpha) M r + alpha s`` with
    ``M = diag(1 - R) W``. With ``enhanced=True`` it is Algorithm 1 of
    eIRWR: the resilience depends on the anomaly (Eq. (4)), the transition
    follows the belief and adds backward edges and self-loops for suspicious
    nodes (Eq. (5)-(8)), the restart is sharpened (Eq. (9)-(10)) and the
    belief is refined in an outer loop (Eq. (11)).

    Args:
        weights: the dependency weights, shape ``(n, n)``: entry ``(i, j)``
            is the traffic from ``i`` to ``j`` (a dense or sparse matrix, or
            a :class:`Graph`, read as symmetric). Each row is normalised to
            sum to 1.
        anomaly: the observed anomaly, ``n`` non-negative numbers, not all 0;
            normalised to sum to 1.
        resilience: ``R_base`` in ``[0, 1]``, the share of a failure a node
            absorbs.
        beta: sensitivity of the resilience to the anomaly, at least 0.
        alpha: restart probability in ``(0, 1]``.
        rho: discount of the backward edges, in ``[0, 1)``.
        q: exponent of the sharpening of the restart, at least 1.
        momentum: weight of the previous result in the belief, in ``[0, 1]``.
        eps: the inner iteration stops when the ``l1`` change is below ``eps``.
        outer: number of outer iterations, at least 1.
        enhanced: use the enhanced walk (default) or the basic one.
        max_iter: most inner iterations.

    Returns:
        The score of each node, shape ``(n,)``; the root cause is its argmax.

    Raises:
        ValueError: on a shape that is not square, an ``anomaly`` of the wrong
            length, negative or all 0, or a parameter out of range.
    """
    if hasattr(weights, "adjacency"):
        weights = weights.adjacency
    W = sp.csr_matrix(weights, dtype=float)
    n = W.shape[0]
    if W.shape != (n, n):
        raise ValueError(f"weights must be square, got {W.shape}")
    s = np.asarray(anomaly, dtype=float).ravel()
    if s.shape != (n,) or np.any(s < 0) or s.sum() <= 0:
        raise ValueError(f"anomaly must be {n} non-negative numbers, not all 0")
    if not (0 <= resilience <= 1 and beta >= 0 and 0 < alpha <= 1):
        raise ValueError("resilience in [0, 1], beta >= 0 and alpha in (0, 1] needed")
    if not (0 <= rho < 1 and q >= 1 and 0 <= momentum <= 1 and eps > 0 and outer >= 1):
        raise ValueError(
            "rho in [0, 1), q >= 1, momentum in [0, 1], eps > 0, outer >= 1"
        )
    s = s / s.sum()
    W = _row_normalise(W)
    R = resilience * np.exp(-beta * s / s.max())
    Mbase = (sp.diags(1.0 - R) @ W).tocsr()
    r = s.copy()
    for _ in range(outer):
        if enhanced:
            b = (1.0 - momentum) * s + momentum * r
            M = _enhanced_transition(Mbase, W, b / b.max(), rho)
            v = b ** _adaptive_exponent(b, q)
            v = v / v.sum()
        else:
            M, v = Mbase, s
        for _ in range(max_iter):
            r_prev = r
            r = (1.0 - alpha) * (M @ r_prev) + alpha * v
            if np.abs(r - r_prev).sum() < eps:
                break
    return r
