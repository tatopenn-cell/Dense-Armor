"""Sequential k-means over a data stream.

Each new point is assigned to its nearest centre and that centre moves
towards the point. The step is ``1 / n_j`` with ``n_j`` the number of
points already assigned to centre ``j``, so the centre is exactly the
running mean of the points that fell into it. With a ``halflife`` the
step becomes ``1 - 2 ** (-1 / halflife)``, the exponential weight that
halves the influence of a sample after ``halflife`` steps, so the
centres follow a drifting stream instead of averaging over the whole
past.

The online k-means setting of Bhattacharjee et al. (2021) compares the
total loss

.. math::

    \\sum_{t=2}^{n} d(x_t, S_{t-1})^2,

where ``S_t`` is the set of centres announced after time ``t``, to the
best fixed set of ``k`` centres in hindsight,

.. math::

    L_k(X_n) = \\inf_{|S| = k} \\sum_{x \\in X_n} d(x, S)^2.

Their Theorem 1 (page 3) shows that a certain randomised algorithm
keeps ``O(k poly(log n))`` centres and achieves
``O(L_k(X_n) + Lambda(X_n))`` loss, with
``Lambda(X_n) = \\sum_t d(x_t, X_{t-1})^2`` the lower-bound term of
the paper. Theorem 2 (page 4) shows that no algorithm with
fewer than ``n`` centres can beat ``Omega(Lambda(X_n))`` on some
stream. Only the lower bound applies here, because the upper bound
needs the randomised centre selection and the poly-log number of
centres; this scheme keeps exactly ``k`` centres. On an i.i.d. stream
the centres converge to the batch k-means solution (the classical
MacQueen step); on an adversarial stream a single far-away point can
push a centre anywhere.

Initialisation: the first ``warmup`` points (default ``10 * k``) are
buffered, ``k`` distinct centres are picked among them by greedy
farthest-first selection (each new centre is the point farthest from the
centres already chosen, the minimax rule of the PatchCore coreset, Roth
et al. 2021, eq. 5), and the buffered points are then assigned in order.
Starting from the first ``k`` points instead can put two centres in the
same group and leave another group without one.

Args:
    k: number of clusters.
    halflife: exponential half-life in samples. ``None`` (default)
        uses the ``1 / n_j`` step; a positive number uses
        ``1 - 2 ** (-1 / halflife)``.
    seed: reserved for future random initialisation; kept for the
        ``Root`` interface.
    warmup: number of points buffered before the centres are chosen.
        Default ``10 * k``.

Raises:
    ValueError: if ``k < 1``, or ``halflife`` is not positive.

Examples:
    >>> from dense_armor.utility.cluster.kmeans import OnlineKMeans
    >>> km = OnlineKMeans(k=2, warmup=4)
    >>> for v in [0.0, 0.1, 10.0, 10.1, 0.2, 9.9]:
    ...     _ = km.learn_one({"x": [v]})
    >>> sorted(round(float(c[0]), 2) for c in km.centers_)
    [0.1, 10.0]

References:
    Bhattacharjee, R., Dasgupta, S., Imola, J. J., Moshkovitz, M. (2021).
        Online k-means clustering on arbitrary data streams.
        arXiv:2102.09101.
    Roth, K. et al. (2021). Towards total recall in industrial anomaly
        detection. arXiv:2106.08265 (eq. 5, greedy minimax selection).
"""
from typing import Any

import numpy as np

from dense_armor.roles import Transformer


def _vec(x: Any) -> np.ndarray | None:
    """Return a finite 1D array from a dict, a Signal, or an array."""
    if hasattr(x, "to_dict"):
        x = x.to_dict()
    if isinstance(x, dict):
        if not x:
            return None
        x = x[min(x)]
    try:
        v = np.asarray(x, dtype=float).ravel()
    except (TypeError, ValueError):
        return None
    if v.size == 0 or not np.all(np.isfinite(v)):
        return None
    return v


class OnlineKMeans(Transformer):
    """Sequential k-means, one sample at a time.

    Each point is assigned to the nearest centre and that centre moves
    towards the point. When two centres end up on the same side of the
    data the assignment is still decided by the nearest distance, so
    empty clusters are not refilled; this is the MacQueen scheme.

    Memory grows with ``k`` and the dimensionality ``d`` of a point:
    two arrays of size ``k`` and ``k * d``, so the footprint is fixed
    after the first ``k`` points.

    Args:
        k: number of clusters.
        halflife: see the module docstring.
        seed: reserved for the ``Root`` interface.

    Raises:
        ValueError: if ``k < 1`` or ``halflife <= 0``.
    """

    budget_s = 2e-3
    memory_class = "O(1)"

    def __init__(
        self,
        k: int,
        halflife: float | None = None,
        seed: int = 0,
        warmup: int | None = None,
    ) -> None:
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        if halflife is not None and halflife <= 0:
            raise ValueError(f"halflife must be > 0, got {halflife}")
        self.k = k
        self.halflife = halflife
        self.seed = seed
        self.warmup = warmup if warmup is not None else 10 * k
        if self.warmup < k:
            raise ValueError(f"warmup must be >= k, got {self.warmup}")
        self._buf: list[np.ndarray] = []
        self.centers_: np.ndarray | None = None
        self.counts_: np.ndarray | None = None
        self.n_seen_ = 0
        self.n_missing_ = 0

    def _unit_test_skips(self) -> set:
        return set()

    @property
    def n_missing(self) -> int:
        return self.n_missing_

    def _alpha(self, j: int) -> float:
        if self.halflife is not None:
            return 1.0 - 2.0 ** (-1.0 / self.halflife)
        counts = self.counts_
        assert counts is not None
        return 1.0 / (int(counts[j]) + 1)

    def _farthest_first(self, buf: np.ndarray) -> np.ndarray:
        """Pick ``k`` distinct centres from ``buf`` by greedy farthest-first."""
        uniq = np.unique(buf, axis=0)
        mean = uniq.mean(axis=0)
        chosen = [int(np.argmin(((uniq - mean) ** 2).sum(axis=1)))]
        d2 = ((uniq - uniq[chosen[0]]) ** 2).sum(axis=1)
        while len(chosen) < min(self.k, uniq.shape[0]):
            j = int(np.argmax(d2))
            chosen.append(j)
            d2 = np.minimum(d2, ((uniq - uniq[j]) ** 2).sum(axis=1))
        return uniq[chosen].copy()

    def _warm_up(self, v: np.ndarray) -> bool:
        """Buffer the first points; return True once the centres exist."""
        if self.centers_ is not None:
            return True
        if self._buf and v.shape[0] != self._buf[0].shape[0]:
            return False
        self._buf.append(v.copy())
        if len(self._buf) < self.warmup:
            return False
        buf = np.stack(self._buf, axis=0)
        self._buf = []
        self.centers_ = self._farthest_first(buf)
        self.counts_ = np.ones(self.centers_.shape[0], dtype=int)
        pending = [c.copy() for c in self.centers_]
        for x in buf:
            hit = next((i for i, c in enumerate(pending) if np.array_equal(c, x)), None)
            if hit is not None:
                pending.pop(hit)
                continue
            d2 = ((self.centers_ - x) ** 2).sum(axis=1)
            c = int(np.argmin(d2))
            self.centers_[c] += self._alpha(c) * (x - self.centers_[c])
            self.counts_[c] += 1
        return False

    def learn_one(
        self, x: Any, y: Any = None, t: float | None = None
    ) -> "OnlineKMeans":
        """Assign ``x`` to the nearest centre and move that centre."""
        self._time_step(t)
        v = _vec(x)
        if v is None:
            self.n_missing_ += 1
            return self
        if not self._warm_up(v):
            self.n_seen_ += 1
            return self
        centers = self.centers_
        assert centers is not None
        if v.shape[0] != centers.shape[1]:
            self.n_missing_ += 1
            return self
        d2 = ((centers - v) ** 2).sum(axis=1)
        j = int(np.argmin(d2))
        a = self._alpha(j)
        centers[j] += a * (v - centers[j])
        assert self.counts_ is not None
        self.counts_[j] += 1
        self.n_seen_ += 1
        return self

    def predict_one(self, x: Any, t: float | None = None) -> int:
        """Return the index of the nearest centre (or 0 if not fitted)."""
        v = _vec(x)
        if v is None or self.centers_ is None:
            return 0
        centers = self.centers_
        if v.shape[0] != centers.shape[1]:
            return 0
        d2 = ((centers - v) ** 2).sum(axis=1)
        return int(np.argmin(d2))

    def transform_one(self, x: Any, t: float | None = None) -> dict:
        """Return ``{"cluster": i, "distances": [...]}``."""
        v = _vec(x)
        if v is None or self.centers_ is None:
            return {"cluster": 0, "distances": []}
        centers = self.centers_
        if v.shape[0] != centers.shape[1]:
            return {"cluster": 0, "distances": []}
        d = np.sqrt(((centers - v) ** 2).sum(axis=1))
        return {"cluster": int(np.argmin(d)), "distances": d.tolist()}
