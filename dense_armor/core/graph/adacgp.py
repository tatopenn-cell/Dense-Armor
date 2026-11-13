"""AdaCGP: online estimation of the graph shift operator from a time series.

References:
    Jenkins, Variddhisai, El-Medany, Ng, Mandic (2024), "Online Graph
    Topology Learning via Time-Vertex Adaptive Filters: From Theory to
    Cardiac Fibrillation", arXiv:2411.01567: the model (5) of Section
    II.C, the objectives (6)-(10) of Section III, Algorithm 1 and
    Algorithm 2 of Section IV and the constants of Section IV.D.
"""

from __future__ import annotations

import numpy as np


def _commutator_term(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    C = A @ B - B @ A
    return C @ B.T - B.T @ C


class AdaCGP:
    """Estimate, one sample at a time, the weights ``W`` of a causal graph process.

    The signal follows ``x_t = sum_p H_p(W, h_p) x_(t-p) + w_t`` with the
    graph filters ``H_p`` polynomials of ``W`` (Eq. (1) and (5)). The
    filters ``Psi_p = H_p`` are estimated with a sparse, forgetting-factor
    least squares, written with the positive and negative parts of the
    variables and solved by projected gradient (Algorithm 1), and ``W`` is
    taken from ``Psi_1``.

    Args:
        n_nodes: number of nodes ``N``, at least 1.
        order: the order ``P`` of the process, at least 1.
        forgetting: the forgetting factor ``lambda`` in ``(0, 1]``.
        mu: the sparsity constants ``mu_p``: a number or ``P`` numbers; the
            constant of step ``t`` is ``mu_p ||P_(p,t) - gamma Q_(p,t)||_inf``
            (Section IV.D).
        gamma: weight of the commutation term of Eq. (8) and (9).
        alpha: the step sizes ``alpha_p`` of the filters: a number or ``P``
            numbers.
        beta: the step size of ``W`` in path 1.
        path: ``2`` takes ``W = Psi_1`` and keeps the commutation term in
            the filter step; ``1`` drops it there and optimises Eq. (9) for
            ``W`` instead.

    Raises:
        ValueError: on an argument out of range.

    Example:
        >>> m = AdaCGP(3, order=1, forgetting=1.0, mu=0.0, alpha=0.5)
        >>> _ = m.update([1.0, 0.0, 0.0]).update([0.0, 2.0, 0.0])
        >>> float(m.psi[1, 0])
        1.0
    """

    def __init__(
        self,
        n_nodes: int,
        order: int = 1,
        forgetting: float = 0.99,
        mu: float | list[float] = 0.0,
        gamma: float = 0.0,
        alpha: float | list[float] = 0.01,
        beta: float = 0.01,
        path: int = 2,
    ) -> None:
        if n_nodes < 1 or order < 1:
            raise ValueError("n_nodes and order must be >= 1")
        if not 0 < forgetting <= 1:
            raise ValueError(f"forgetting must be in (0, 1], got {forgetting}")
        if path not in (1, 2):
            raise ValueError(f"path must be 1 or 2, got {path}")
        if gamma < 0 or beta <= 0:
            raise ValueError("gamma must be >= 0 and beta > 0")
        self.n, self.order = int(n_nodes), int(order)
        self.forgetting, self.gamma, self.beta, self.path = (
            float(forgetting),
            float(gamma),
            float(beta),
            path,
        )
        self.mu = self._per_order(mu, "mu")
        self.alpha = self._per_order(alpha, "alpha")
        if np.any(self.mu < 0) or np.any(self.alpha <= 0):
            raise ValueError("mu must be >= 0 and alpha > 0")
        n, P = self.n, self.order
        self._pp, self._pm = np.zeros((n, n * P)), np.zeros((n, n * P))
        self._wp, self._wm = np.zeros((n, n)), np.zeros((n, n))
        self._R, self._Pc = np.zeros((n * P, n * P)), np.zeros((n, n * P))
        self._hist = [np.zeros(n) for _ in range(P)]
        self.t = 0
        self.filter_coefficients: np.ndarray | None = None

    def _per_order(self, value, name: str) -> np.ndarray:
        v = np.atleast_1d(np.asarray(value, dtype=float))
        if v.size == 1:
            v = np.repeat(v, self.order)
        if v.shape != (self.order,):
            raise ValueError(f"{name} must be a number or {self.order} numbers")
        return v

    @property
    def psi(self) -> np.ndarray:
        """The first filter ``Psi_1`` (shape ``(N, N)``)."""
        return self.filters[0]

    @property
    def filters(self) -> list[np.ndarray]:
        """The estimated filters ``Psi_1 .. Psi_P``."""
        n = self.n
        full = self._pp - self._pm
        return [full[:, p * n : (p + 1) * n] for p in range(self.order)]

    @property
    def weights(self) -> np.ndarray:
        """The estimate of ``W`` (shape ``(N, N)``)."""
        if self.path == 2:
            return self.psi
        return self._wp - self._wm

    def _blocks(self, M: np.ndarray) -> list[np.ndarray]:
        n = self.n
        return [M[:, p * n : (p + 1) * n] for p in range(self.order)]

    def _correlations(self, x: np.ndarray) -> None:
        xp = np.concatenate(self._hist)
        lam = self.forgetting
        self._R = lam * self._R + np.outer(xp, xp)
        self._Pc = lam * self._Pc + np.outer(x, xp)

    def _push(self, x: np.ndarray) -> None:
        self._hist = [x.copy()] + self._hist[:-1]
        self.t += 1

    def _check(self, x) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if x.shape != (self.n,):
            raise ValueError(f"x must have shape ({self.n},), got {x.shape}")
        return x

    def update(self, x) -> AdaCGP:
        """One step of Algorithm 1 with the sample ``x``."""
        x = self._check(x)
        n, P = self.n, self.order
        self._correlations(x)
        full = self._pp - self._pm
        blocks = self._blocks(full)
        Q = np.zeros((n, n * P))
        if self.path == 2 and self.gamma > 0 and P > 1:
            for p in range(P):
                Q[:, p * n : (p + 1) * n] = sum(
                    _commutator_term(blocks[p], blocks[k]) for k in range(P) if k != p
                )
        target = self._Pc - self.gamma * Q
        mu_t = np.array(
            [self.mu[p] * np.abs(b).max() for p, b in enumerate(self._blocks(target))]
        )
        Mt = np.concatenate([np.full((n, n), m) for m in mu_t], axis=1)
        G = full @ self._R - target
        step = np.repeat(self.alpha, n)[None, :]
        self._pp = np.maximum(self._pp - (Mt + G) * step, 0.0)
        self._pm = np.maximum(self._pm - (Mt - G) * step, 0.0)
        if self.path == 1:
            new = self.filters
            W = self._wp - self._wm
            S = sum(
                (_commutator_term(W, new[k]) for k in range(1, P)),
                start=np.zeros((n, n)),
            )
            V = W - (new[0] - self.gamma * S)
            self._wp = np.maximum(self._wp - self.beta * (mu_t[0] + V), 0.0)
            self._wm = np.maximum(self._wm - self.beta * (mu_t[0] - V), 0.0)
        self._push(x)
        return self

    def debias(self, x) -> AdaCGP:
        """One step of Algorithm 2, lines 5-11: refit the non-zero entries.

        The gradient of the least squares is projected on the entries that
        are not zero in the filters, and ``W`` is set to ``Psi_1``.
        """
        x = self._check(x)
        n = self.n
        self._correlations(x)
        full = self._pp - self._pm
        G = (full @ self._R - self._Pc) * (full != 0)
        full = full - G * np.repeat(self.alpha, n)[None, :]
        self._pp, self._pm = np.maximum(full, 0.0), np.maximum(-full, 0.0)
        if self.path == 1:
            self._wp, self._wm = (
                np.maximum(self.filters[0], 0.0),
                np.maximum(-self.filters[0], 0.0),
            )
        self._push(x)
        return self

    def update_filter(
        self, x, rho: float = 0.01, eta: float = 0.0, eps: float = 1e-3
    ) -> AdaCGP:
        """One step of the estimate of the filter coefficients ``h`` (Eq. (19)).

        Uses the instantaneous form of the paper (``lambda = 0``) with a
        descent step on the objective (10): ``h <- h + rho (Y^T e - eta_t b)``,
        ``e = x - Y h``, ``b_i = sign(h_i) / (eps + |h_i|)`` and
        ``eta_t = eta ||Y^T x||_inf``. The matrix ``Y`` has the columns
        ``W^k x_(t-p)`` for ``p = 1 .. P`` and ``k = 0 .. p``, so ``h`` has
        ``P (P + 3) / 2`` entries. Call it before ``update`` or ``debias``
        with the same sample, so that the history is the one of ``x_t``.
        """
        x = self._check(x)
        W = self.weights
        cols = []
        for p in range(1, self.order + 1):
            v = self._hist[p - 1]
            for _ in range(p + 1):
                cols.append(v)
                v = W @ v
        Y = np.stack(cols, axis=1)
        if self.filter_coefficients is None:
            self.filter_coefficients = np.zeros(Y.shape[1])
        h = self.filter_coefficients
        e = x - Y @ h
        eta_t = eta * np.abs(Y.T @ x).max()
        b = np.sign(h) / (eps + np.abs(h))
        self.filter_coefficients = h + rho * (Y.T @ e - eta_t * b)
        return self
