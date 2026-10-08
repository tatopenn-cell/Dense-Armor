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

The centres are created from the first ``k`` distinct points seen; until ``k``
exist, assignments use the centres created so far.

Args:
    k: number of clusters.
    halflife: exponential half-life in samples. ``None`` (default)
        uses the ``1 / n_j`` step; a positive number uses
        ``1 - 2 ** (-1 / halflife)``.
    seed: reserved for future random initialisation; kept for the
        ``Root`` interface.

Raises:
    ValueError: if ``k < 1``, or ``halflife`` is not positive.

Examples:
    >>> from dense_armor.utility.cluster.kmeans import OnlineKMeans
    >>> km = OnlineKMeans(k=2)
    >>> for v in [0.0, 0.1, 10.0, 10.1, 0.2, 9.9]:
    ...     _ = km.learn_one({"x": [v]})
    >>> km.predict_one({"x": [0.0]}), km.predict_one({"x": [10.0]})
    (0, 1)

References:
    Bhattacharjee, R., Dasgupta, S., Imola, J. J., Moshkovitz, M. (2021).
        Online k-means clustering on arbitrary data streams.
        arXiv:2102.09101.
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

    budget_s = 1e-4
    memory_class = "O(1)"

    def __init__(
        self, k: int, halflife: float | None = None, seed: int = 0
    ) -> None:
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        if halflife is not None and halflife <= 0:
            raise ValueError(f"halflife must be > 0, got {halflife}")
        self.k = k
        self.halflife = halflife
        self.seed = seed
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

    def _maybe_add_centre(self, v: np.ndarray) -> bool:
        """Create a new centre from ``v`` while fewer than ``k`` exist.

        Returns True when ``v`` became a new centre (nothing else to do).
        """
        centers = self.centers_
        if centers is None:
            self.centers_ = v.copy()[None, :]
            self.counts_ = np.ones(1, dtype=int)
            return True
        if centers.shape[0] >= self.k or v.shape[0] != centers.shape[1]:
            return False
        if any(np.array_equal(v, c) for c in centers):
            return False
        self.centers_ = np.vstack([centers, v])
        assert self.counts_ is not None
        self.counts_ = np.append(self.counts_, 1)
        return True

    def learn_one(
        self, x: Any, y: Any = None, t: float | None = None
    ) -> "OnlineKMeans":
        """Assign ``x`` to the nearest centre and move that centre."""
        self._time_step(t)
        v = _vec(x)
        if v is None:
            self.n_missing_ += 1
            return self
        if self._maybe_add_centre(v):
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
