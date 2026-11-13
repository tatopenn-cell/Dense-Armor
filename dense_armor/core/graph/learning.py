"""Informed graph learning: a graph from smooth signals and domain knowledge.

References:
    Niresi, Kuhn, Frusque, Fink (2024), "Informed Graph Learning By Domain
    Knowledge Injection and Smooth Graph Signal Representation",
    arXiv:2406.03898, Section III: the physics-inspired graph (III.A, Eq.
    (6)-(8)), the problem (9)-(10) and its reformulation (11). The problem
    is solved here with a primal-dual splitting (Condat-Vu form, as in
    Eq. (10) of arXiv:2202.06432); the paper names the Monotone+Lipschitz
    Forward-Backward-Forward algorithm of its reference [16].
"""

from __future__ import annotations

import numpy as np

from dense_armor.core.graph.core import Graph


def weights_from_drops(drops, cut: float = 0.1) -> np.ndarray:
    """Edge weights from the drop of a quantity along each link (Section III.A).

    A link with a small drop is a strong one: ``W = 1 / |drop|``, rescaled so
    that the weights are in ``(0, 1]``, and the weights below ``cut`` are
    set to 0 to keep the graph sparse. A drop of 0 or an infinite one gives
    weight 0.

    Args:
        drops: array of the absolute drops, any shape; 0 where there is no link.
        cut: weights below this value are removed.

    Example:
        >>> weights_from_drops([0.0, 1.0, 2.0, 20.0]).round(2).tolist()
        [0.0, 1.0, 0.5, 0.0]
    """
    d = np.abs(np.asarray(drops, dtype=float))
    with np.errstate(divide="ignore"):
        w = np.where((d > 0) & np.isfinite(d), 1.0 / np.where(d > 0, d, 1.0), 0.0)
    top = w.max() if w.size else 0.0
    if top > 0:
        w = w / top
    return np.where(w >= cut, w, 0.0)


def _pairs(n: int):
    return np.triu_indices(n, k=1)


def _degree_operator(n: int) -> np.ndarray:
    i, j = _pairs(n)
    S = np.zeros((n, i.size))
    S[i, np.arange(i.size)] = 1.0
    S[j, np.arange(i.size)] = 1.0
    return S


def learning_objective(w, z, alpha: float, beta: float, upsilon: float, m, w_pi, S):
    """The objective (11): ``2 w^T z - alpha 1^T log(S w) + beta ||w||^2 +
    upsilon ||m * w - w_pi||^2``, ``inf`` if a degree is not positive."""
    d = S @ w
    if np.any(d <= 0):
        return np.inf
    return float(
        2 * w @ z
        - alpha * np.log(d).sum()
        + beta * w @ w
        + upsilon * np.sum((m * w - w_pi) ** 2)
    )


def learn_graph(
    X,
    alpha: float = 1.0,
    beta: float = 0.1,
    known=None,
    upsilon: float = 0.0,
    max_iter: int = 5000,
    tol: float = 1e-8,
    threshold: float = 1e-4,
    return_weights: bool = False,
):
    """Learn a graph from signals that are smooth on it, with prior knowledge.

    Minimises the problem (9)-(11) over the non-negative weights of an
    undirected graph with no self-loop: the smoothness of the signals,
    a log barrier on the degrees (every node gets an edge), an ``l2`` term
    on the weights, and a term that keeps the weights close to the graph
    ``known`` wherever it has an edge.

    Args:
        X: signals, shape ``(n, m)``: ``n`` nodes and ``m`` observations
            (a vector of length ``n`` is one observation).
        alpha: weight of the log barrier on the degrees, positive.
        beta: weight of the ``l2`` term, positive.
        known: the prior weights ``W_PI``, a symmetric ``(n, n)`` matrix
            (``M`` marks its non-zero entries), or ``None`` for no prior.
        upsilon: weight of the prior term, at least 0.
        max_iter: most iterations.
        tol: stop when the weights change by less than ``tol`` in relative norm.
        threshold: a weight below this is not an edge of the returned graph.
        return_weights: also return the ``(n, n)`` weight matrix.

    Returns:
        A :class:`Graph`, or ``(Graph, W)`` if ``return_weights``.

    Raises:
        ValueError: on a wrong shape, a non-positive ``alpha`` or ``beta``, a
            negative ``upsilon``, ``n < 2``, or a ``known`` that is not a
            symmetric matrix with a zero diagonal.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    if X.ndim != 2 or X.shape[0] < 2:
        raise ValueError(f"X must have shape (n, m) with n >= 2, got {X.shape}")
    if alpha <= 0 or beta <= 0 or upsilon < 0:
        raise ValueError("alpha and beta must be > 0 and upsilon >= 0")
    n = X.shape[0]
    i, j = _pairs(n)
    sq = ((X[:, None, :] - X[None, :, :]) ** 2).sum(axis=2)
    z = sq[i, j]
    if known is None:
        m_vec, w_pi = np.zeros_like(z), np.zeros_like(z)
        upsilon = 0.0
    else:
        K = np.asarray(known, dtype=float)
        if K.shape != (n, n) or not np.allclose(K, K.T) or np.any(np.diag(K) != 0):
            raise ValueError(
                "known must be a symmetric (n, n) matrix with a zero diagonal"
            )
        w_pi = K[i, j]
        m_vec = (w_pi != 0).astype(float)
    S = _degree_operator(n)
    zeta = 2.0 * beta + 2.0 * upsilon
    norm_s2 = 2.0 * (n - 1)
    sigma = 0.5 / norm_s2
    tau = 1.0 / (zeta / 2.0 + sigma * norm_s2)
    w = np.zeros_like(z)
    v = np.zeros(n)
    for _ in range(max_iter):
        grad = 2.0 * z + 2.0 * beta * w + 2.0 * upsilon * m_vec * (w - w_pi)
        wn = np.maximum(w - tau * (grad + S.T @ v), 0.0)
        u = v + sigma * (S @ (2.0 * wn - w))
        x = u / sigma
        vn = u - sigma * (x + np.sqrt(x * x + 4.0 * alpha / sigma)) / 2.0
        change = float(np.linalg.norm(wn - w)) / max(float(np.linalg.norm(w)), 1.0)
        change_v = float(np.linalg.norm(vn - v)) / max(float(np.linalg.norm(v)), 1.0)
        w, v = wn, vn
        if change <= tol and change_v <= tol:
            break
    W = np.zeros((n, n))
    W[i, j] = w
    W = W + W.T
    G = Graph(
        n,
        [(int(a), int(b), float(W[a, b])) for a, b in zip(i, j) if W[a, b] > threshold],
    )
    return (G, W) if return_weights else G
