"""AdaCGP: online estimation of the graph shift operator from a time series.

References:
    Jenkins, Variddhisai, El-Medany, Ng, Mandic (2024), "Online Graph
    Topology Learning via Time-Vertex Adaptive Filters: From Theory to
    Cardiac Fibrillation", arXiv:2411.01567: the model (5) of Section
    II.C, the objectives (6)-(10) of Section III, Algorithm 1 and
    Algorithm 2 of Section IV and the constants of Section IV.D.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import numpy as np

from dense_armor.roles.estimator import Estimator


def _commutator_term(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    C = A @ B - B @ A
    return C @ B.T - B.T @ C


class AdaCGP(Estimator):
    """Estimate, one sample at a time, the weights ``W`` of a causal graph process.

    The signal follows ``x_t = sum_p H_p(W, h_p) x_(t-p) + w_t`` with the
    graph filters ``H_p`` polynomials of ``W`` (Eq. (1) and (5)). The
    filters ``Psi_p = H_p`` are estimated with a sparse, forgetting-factor
    least squares, written with the positive and negative parts of the
    variables and solved by projected gradient (Algorithm 1), and ``W`` is
    taken from ``Psi_1``.

    The estimator learns without labels: ``learn_one(x)`` takes the signal
    of one time-slot as a dict (a :class:`Signal` too) or as a sequence,
    one value per node. The number of nodes is the one of the first sample
    unless ``n_nodes`` is given.

    Args:
        n_nodes: number of nodes ``N``, at least 1, or ``None`` to take it
            from the first sample.
        order: the order ``P`` of the process, at least 1.
        forgetting: the forgetting factor ``lambda`` in ``(0, 1]``.
        mu: the sparsity constants ``mu_p``: a number, or ``P`` numbers; the
            constant of step ``t`` is ``mu_p ||P_(p,t) - gamma Q_(p,t)||_inf``
            (Section IV.D).
        gamma: weight of the commutation term of Eq. (8) and (9).
        alpha: the step sizes ``alpha_p`` of the filters: a number, or ``P``
            numbers.
        beta: the step size of ``W`` in path 1.
        path: ``2`` takes ``W = Psi_1`` and keeps the commutation term in
            the filter step; ``1`` drops it there and optimises Eq. (9) for
            ``W`` instead.

    Raises:
        ValueError: on an argument out of range, or on a sample whose
            length differs from the number of nodes.

    Example:
        >>> m = AdaCGP(3, order=1, forgetting=1.0, mu=0.0, alpha=0.5)
        >>> _ = m.learn_one([1.0, 0.0, 0.0]).learn_one([0.0, 2.0, 0.0])
        >>> float(m.psi[1, 0])
        1.0
    """

    _supervised = False

    def __init__(
        self,
        n_nodes: int | None = None,
        order: int = 1,
        forgetting: float = 0.99,
        mu: float | list[float] = 0.0,
        gamma: float = 0.0,
        alpha: float | list[float] = 0.01,
        beta: float = 0.01,
        path: int = 2,
    ) -> None:
        if (n_nodes is not None and n_nodes < 1) or order < 1:
            raise ValueError("n_nodes and order must be >= 1")
        if not 0 < forgetting <= 1:
            raise ValueError(f"forgetting must be in (0, 1], got {forgetting}")
        if path not in (1, 2):
            raise ValueError(f"path must be 1 or 2, got {path}")
        if gamma < 0 or beta <= 0:
            raise ValueError("gamma must be >= 0 and beta > 0")
        self.n_nodes, self.order = n_nodes, int(order)
        self.forgetting, self.gamma, self.beta, self.path = (
            float(forgetting),
            float(gamma),
            float(beta),
            path,
        )
        self.mu, self.alpha = mu, alpha
        if np.any(self._mu() < 0) or np.any(self._alpha() <= 0):
            raise ValueError("mu must be >= 0 and alpha > 0")
        self.n_nodes_: int | None = None
        self.psi_plus_: np.ndarray | None = None
        self.psi_minus_: np.ndarray | None = None
        self.w_plus_: np.ndarray | None = None
        self.w_minus_: np.ndarray | None = None
        self.corr_: np.ndarray | None = None
        self.cross_: np.ndarray | None = None
        self.history_: np.ndarray | None = None
        self.coef_: np.ndarray | None = None
        self.steps_ = 0
        if n_nodes is not None:
            self._allocate(int(n_nodes))

    @classmethod
    def _unit_test_params(cls) -> Iterator[dict[str, Any]]:
        yield {}
        yield {"order": 2, "gamma": 0.1, "mu": 0.01}
        yield {"order": 2, "path": 1, "gamma": 0.1, "mu": 0.01}

    @property
    def memory_class(self) -> str:
        """``"O(1)"``: every array has a size fixed by ``N`` and ``P``."""
        return "O(1)"

    @property
    def budget_s(self) -> float:
        """Declared latency of ``learn_one``, in seconds.

        The cost of a step grows as ``N^3 P^2`` (the product ``Psi R`` and the
        commutators). The budget is ``1e-2`` s up to ``N = 16`` nodes and
        ``P = 3``, more than ten times the 99th percentile measured there
        (0.52 ms), and grows with the cost beyond that.
        """
        n = self.n_nodes_ or 1
        return 1e-2 * max(1.0, (n / 16.0) ** 3 * (self.order / 3.0) ** 2)

    def _per_order(self, value: Any, name: str) -> np.ndarray:
        v = np.atleast_1d(np.asarray(value, dtype=float))
        if v.size == 1:
            v = np.repeat(v, self.order)
        if v.shape != (self.order,):
            raise ValueError(f"{name} must be a number or {self.order} numbers")
        return v

    def _mu(self) -> np.ndarray:
        return self._per_order(self.mu, "mu")

    def _alpha(self) -> np.ndarray:
        return self._per_order(self.alpha, "alpha")

    def _allocate(self, n: int) -> None:
        P = self.order
        self.n_nodes_ = n
        self.psi_plus_, self.psi_minus_ = np.zeros((n, n * P)), np.zeros((n, n * P))
        self.w_plus_, self.w_minus_ = np.zeros((n, n)), np.zeros((n, n))
        self.corr_, self.cross_ = np.zeros((n * P, n * P)), np.zeros((n, n * P))
        self.history_ = np.zeros((P, n))
        self.coef_ = None
        self.steps_ = 0

    def _vector(self, x: Any) -> np.ndarray:
        if hasattr(x, "values") and callable(x.values):
            v = np.array([float(a) for a in x.values()])
        else:
            v = np.asarray(x, dtype=float).ravel()
        if self.n_nodes_ is None:
            if v.size < 1:
                raise ValueError("a sample needs at least one value")
            self._allocate(int(v.size))
        elif v.shape != (self.n_nodes_,):
            raise ValueError(f"x must have {self.n_nodes_} values, got {v.shape[0]}")
        return v

    def _ready(self) -> int:
        if self.n_nodes_ is None:
            raise ValueError("no sample has been seen yet")
        return self.n_nodes_

    @property
    def psi(self) -> np.ndarray:
        """The first filter ``Psi_1`` (shape ``(N, N)``)."""
        return self.filters[0]

    @property
    def filters(self) -> list[np.ndarray]:
        """The estimated filters ``Psi_1 .. Psi_P``."""
        n = self._ready()
        full = self.psi_plus_ - self.psi_minus_  # type: ignore[operator]
        return [full[:, p * n : (p + 1) * n] for p in range(self.order)]

    @property
    def weights(self) -> np.ndarray:
        """The estimate of ``W`` (shape ``(N, N)``)."""
        if self.path == 2:
            return self.psi
        self._ready()
        return self.w_plus_ - self.w_minus_  # type: ignore[operator]

    @property
    def filter_coefficients(self) -> np.ndarray | None:
        """The coefficients ``h`` of the polynomials, or ``None`` before ``update_filter``."""
        return self.coef_

    def _blocks(self, M: np.ndarray) -> list[np.ndarray]:
        n = self._ready()
        return [M[:, p * n : (p + 1) * n] for p in range(self.order)]

    def _correlations(self, x: np.ndarray) -> None:
        xp = self.history_.reshape(-1)  # type: ignore[union-attr]
        lam = self.forgetting
        self.corr_ = lam * self.corr_ + np.outer(xp, xp)  # type: ignore[operator]
        self.cross_ = lam * self.cross_ + np.outer(x, xp)  # type: ignore[operator]

    def _step(self) -> np.ndarray:
        """The step sizes ``alpha_p`` of the filters, one per column of ``Psi``.

        Each is capped at ``1 / trace(R)``: the gradient step on the least
        squares cannot diverge when ``alpha lambda_max(R) < 2``, and
        ``lambda_max(R) <= trace(R)``. A step already below the cap is not
        changed.
        """
        cap = 1.0 / max(float(np.trace(self.corr_)), 1e-12)  # type: ignore[arg-type]
        return np.minimum(np.repeat(self._alpha(), self._ready()), cap)[None, :]

    def _push(self, x: np.ndarray) -> None:
        self.history_ = np.vstack([x[None, :], self.history_[:-1]])  # type: ignore[index]
        self.steps_ += 1

    def learn_one(self, x: Any, t: float | None = None) -> AdaCGP:
        """One step of Algorithm 1 with the sample ``x``."""
        x = self._vector(x)
        self._time_step(t)
        n, P = self._ready(), self.order
        self._correlations(x)
        full = self.psi_plus_ - self.psi_minus_  # type: ignore[operator]
        blocks = self._blocks(full)
        Q = np.zeros((n, n * P))
        if self.path == 2 and self.gamma > 0 and P > 1:
            for p in range(P):
                Q[:, p * n : (p + 1) * n] = sum(
                    _commutator_term(blocks[p], blocks[k]) for k in range(P) if k != p
                )
        target = self.cross_ - self.gamma * Q  # type: ignore[operator]
        mu_t = np.array(
            [m * np.abs(b).max() for m, b in zip(self._mu(), self._blocks(target))]
        )
        Mt = np.concatenate([np.full((n, n), m) for m in mu_t], axis=1)
        G = full @ self.corr_ - target  # type: ignore[operator]
        step = self._step()
        self.psi_plus_ = np.maximum(self.psi_plus_ - (Mt + G) * step, 0.0)  # type: ignore[operator]
        self.psi_minus_ = np.maximum(self.psi_minus_ - (Mt - G) * step, 0.0)  # type: ignore[operator]
        if self.path == 1:
            new = self.filters
            W = self.w_plus_ - self.w_minus_  # type: ignore[operator]
            S = sum(
                (_commutator_term(W, new[k]) for k in range(1, P)),
                start=np.zeros((n, n)),
            )
            V = W - (new[0] - self.gamma * S)
            self.w_plus_ = np.maximum(self.w_plus_ - self.beta * (mu_t[0] + V), 0.0)  # type: ignore[operator]
            self.w_minus_ = np.maximum(self.w_minus_ - self.beta * (mu_t[0] - V), 0.0)  # type: ignore[operator]
        self._push(x)
        return self

    def update(self, x: Any, t: float | None = None) -> AdaCGP:
        """The same as :meth:`learn_one`."""
        return self.learn_one(x, t)

    def debias(self, x: Any, t: float | None = None) -> AdaCGP:
        """One step of Algorithm 2, lines 5-11: refit the non-zero entries.

        The gradient of the least squares is projected on the entries that
        are not zero in the filters, and ``W`` is set to ``Psi_1``.
        """
        x = self._vector(x)
        self._time_step(t)
        self._correlations(x)
        full = self.psi_plus_ - self.psi_minus_  # type: ignore[operator]
        G = (full @ self.corr_ - self.cross_) * (full != 0)  # type: ignore[operator]
        full = full - G * self._step()
        self.psi_plus_, self.psi_minus_ = np.maximum(full, 0.0), np.maximum(-full, 0.0)
        if self.path == 1:
            f0 = self.filters[0]
            self.w_plus_, self.w_minus_ = np.maximum(f0, 0.0), np.maximum(-f0, 0.0)
        self._push(x)
        return self

    def update_filter(
        self, x: Any, rho: float = 0.01, eta: float = 0.0, eps: float = 1e-3
    ) -> AdaCGP:
        """One step of the estimate of the filter coefficients ``h`` (Eq. (19)).

        Uses the instantaneous form of the paper (``lambda = 0``) with a
        descent step on the objective (10): ``h <- h + rho (Y^T e - eta_t b)``,
        ``e = x - Y h``, ``b_i = sign(h_i) / (eps + |h_i|)`` and
        ``eta_t = eta ||Y^T x||_inf``. The matrix ``Y`` has the columns
        ``W^k x_(t-p)`` for ``p = 1 .. P`` and ``k = 0 .. p``, so ``h`` has
        ``P (P + 3) / 2`` entries. Call it before ``learn_one`` or ``debias``
        with the same sample, so that the history is the one of ``x_t``.
        """
        x = self._vector(x)
        W = self.weights
        cols = []
        for p in range(1, self.order + 1):
            v = self.history_[p - 1]  # type: ignore[index]
            for _ in range(p + 1):
                cols.append(v)
                v = W @ v
        Y = np.stack(cols, axis=1)
        h = np.zeros(Y.shape[1]) if self.coef_ is None else self.coef_
        e = x - Y @ h
        eta_t = eta * np.abs(Y.T @ x).max()
        b = np.sign(h) / (eps + np.abs(h))
        self.coef_ = h + rho * (Y.T @ e - eta_t * b)
        return self
