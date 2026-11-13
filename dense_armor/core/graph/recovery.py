"""Recovery of a time-varying graph signal on a dynamic graph.

References:
    Yamagata, Naganuma, Ono (2022), "Robust Time-Varying Graph Signal
    Recovery for Dynamic Physical Sensor Network Data", arXiv:2202.06432,
    Section III: the problem (13), the proximity operators (23)-(28) and
    Algorithm 1 (primal-dual splitting, Eq. (10) of Section II.E).
"""

from __future__ import annotations

import numpy as np


def project_l1_ball(z, radius: float) -> np.ndarray:
    """Projection of ``z`` on ``{v : ||v||_1 <= radius}``.

    The threshold ``s`` is the one with ``sum max(0, |z_j| - s) = radius``
    (Eq. (23)-(24) of the paper).

    Example:
        >>> project_l1_ball([3.0, 1.0], 2.0).tolist()
        [2.0, 0.0]
    """
    z = np.asarray(z, dtype=float)
    if radius <= 0:
        return np.zeros_like(z)
    a = np.abs(z).ravel()
    if a.sum() <= radius:
        return z.copy()
    u = np.sort(a)[::-1]
    c = np.cumsum(u) - radius
    j = np.arange(1, u.size + 1)
    k = np.nonzero(u - c / j > 0)[0][-1]
    s = c[k] / (k + 1)
    return np.sign(z) * np.maximum(np.abs(z) - s, 0.0)


def prox_l1(z, gamma: float) -> np.ndarray:
    """Soft-thresholding: the proximity operator of ``gamma ||.||_1`` (Eq. (25))."""
    z = np.asarray(z, dtype=float)
    return np.sign(z) * np.maximum(np.abs(z) - gamma, 0.0)


def project_l2_ball(z, center, radius: float) -> np.ndarray:
    """Projection of ``z`` on ``{v : ||v - center||_2 <= radius}`` (Eq. (28)).

    Example:
        >>> project_l2_ball([3.0, 4.0], [0.0, 0.0], 1.0).round(3).tolist()
        [0.6, 0.8]
    """
    z = np.asarray(z, dtype=float)
    c = np.asarray(center, dtype=float)
    d = z - c
    nd = float(np.linalg.norm(d))
    if nd <= radius:
        return z.copy()
    return c + radius * d / nd


def _dense(L) -> np.ndarray:
    if hasattr(L, "laplacian"):
        L = L.laplacian
    return L.toarray() if hasattr(L, "toarray") else np.asarray(L, dtype=float)


def _diff(X: np.ndarray) -> np.ndarray:
    return X[:, 1:] - X[:, :-1]


def _diff_t(V: np.ndarray) -> np.ndarray:
    out = np.zeros((V.shape[0], V.shape[1] + 1))
    out[:, :-1] -= V
    out[:, 1:] += V
    return out


def recover_graph_signal(
    Y,
    laplacians,
    mask=None,
    lam: float = 0.1,
    eps: float = 0.0,
    eta: float = 0.0,
    temporal: str = "l1",
    smoothness: str = "signal",
    max_iter: int = 2000,
    tol: float = 1e-4,
    gamma1: float | None = None,
    gamma2: float | None = None,
    return_outliers: bool = False,
):
    """Estimate a smooth time-varying graph signal from a corrupted one.

    Solves ``min sum_k R_v^(k)(P(X)) + lam R_t(D(X))`` subject to
    ``||Y - Phi(X + S)||_F <= eps`` and ``||S||_1 <= eta`` (Eq. (13)) with
    the primal-dual splitting of Algorithm 1.

    Args:
        Y: observation, shape ``(n, p)``: ``n`` vertices, ``p`` time-slots.
        laplacians: the graph Laplacian of each time-slot: a sequence of
            ``p`` matrices or :class:`Graph` objects, or one matrix used for
            all the slots (the static case, Eq. (18)).
        mask: 1 where ``Y`` is observed and 0 where it is missing, shape
            ``(n, p)``; all observed by default.
        lam: weight of the temporal term, at least 0.
        eps: radius of the data-fidelity ball, at least 0.
        eta: radius of the ``l1`` ball of the outliers, at least 0.
        temporal: ``"l1"`` for ``||D(X)||_1`` or ``"l2"`` for ``||D(X)||_F^2``.
        smoothness: ``"signal"`` for ``x_k^T L_k x_k`` (Eq. (16)) or
            ``"difference"`` for ``(x_(k+1) - x_k)^T L_k (x_(k+1) - x_k)``
            (Eq. (17)).
        max_iter: most iterations.
        tol: stop when ``X`` and ``S`` change by less than ``tol`` in
            relative norm (the criterion of Section IV.B).
        gamma1: primal step; by default the largest one allowed by
            ``1 / gamma1 - gamma2 L_A >= beta / 2``, with ``L_A = 6`` an upper
            bound of the largest eigenvalue of ``A^T A`` and ``beta`` the
            Lipschitz constant of the gradient of the smoothness term.
        gamma2: dual step; ``1/6`` by default.
        return_outliers: also return the estimated outliers ``S``.

    Returns:
        ``X`` with shape ``(n, p)``, or ``(X, S)`` if ``return_outliers``.

    Raises:
        ValueError: on a wrong shape, ``p < 2``, a negative ``lam``, ``eps``
            or ``eta``, or an unknown ``temporal`` or ``smoothness``.
    """
    Y = np.asarray(Y, dtype=float)
    if Y.ndim != 2 or Y.shape[1] < 2:
        raise ValueError(f"Y must have shape (n, p) with p >= 2, got {Y.shape}")
    n, p = Y.shape
    if lam < 0 or eps < 0 or eta < 0:
        raise ValueError("lam, eps and eta must be >= 0")
    if temporal not in ("l1", "l2"):
        raise ValueError(f"temporal must be 'l1' or 'l2', got {temporal!r}")
    if smoothness not in ("signal", "difference"):
        raise ValueError(
            f"smoothness must be 'signal' or 'difference', got {smoothness!r}"
        )
    if mask is None:
        M = np.ones((n, p))
    else:
        M = np.asarray(mask, dtype=float)
        if M.shape != (n, p):
            raise ValueError(f"mask must have shape {(n, p)}, got {M.shape}")
    if hasattr(laplacians, "laplacian") or (
        hasattr(laplacians, "shape") and np.ndim(laplacians) == 2
    ):
        Ls = [_dense(laplacians)] * p
    else:
        Ls = [_dense(L) for L in laplacians]
        if len(Ls) != p:
            raise ValueError(f"expected {p} Laplacians, got {len(Ls)}")
    if any(L.shape != (n, n) for L in Ls):
        raise ValueError(f"every Laplacian must have shape {(n, n)}")
    Yo = M * Y
    lmax = max(float(np.linalg.eigvalsh(L).max()) for L in Ls)
    beta = 2.0 * lmax * (1.0 if smoothness == "signal" else 4.0)
    g2 = 1.0 / 6.0 if gamma2 is None else float(gamma2)
    g1 = 1.0 / (beta / 2.0 + 6.0 * g2) if gamma1 is None else float(gamma1)

    def grad(X):
        G = np.zeros((n, p))
        if smoothness == "signal":
            for k in range(p):
                G[:, k] = 2.0 * Ls[k] @ X[:, k]
        else:
            for k in range(p - 1):
                g = 2.0 * Ls[k] @ (X[:, k + 1] - X[:, k])
                G[:, k + 1] += g
                G[:, k] -= g
        return G

    X, S = Yo.copy(), np.zeros((n, p))
    v1, v2 = np.zeros((n, p - 1)), np.zeros((n, p))
    for _ in range(max_iter):
        Xn = X - g1 * (grad(X) + _diff_t(v1) + M * v2)
        Sn = project_l1_ball(S - g1 * (M * v2), eta)
        dX, dS = 2.0 * Xn - X, 2.0 * Sn - S
        v1 = v1 + g2 * _diff(dX)
        v2 = v2 + g2 * (M * (dX + dS))
        z1 = v1 / g2
        if temporal == "l1":
            v1 = v1 - g2 * prox_l1(z1, lam / g2)
        else:
            v1 = v1 - g2 * z1 / (2.0 * lam / g2 + 1.0)
        v2 = v2 - g2 * project_l2_ball(v2 / g2, Yo, eps)
        rx = np.linalg.norm(Xn - X) / max(np.linalg.norm(X), 1e-300)
        rs = np.linalg.norm(Sn - S) / max(np.linalg.norm(S), 1e-300)
        X, S = Xn, Sn
        if rx <= tol and rs <= tol:
            break
    return (X, S) if return_outliers else X
