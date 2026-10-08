"""Density-based stream clustering with micro-clusters and a damped window.

A micro-cluster is a small set of close points, kept as a compact
summary (centre, weight, timestamp of the last update). Two
micro-clusters are directly density reachable when the distance
between their centres is at most the sum of their radii; a chain of
direct reachability is density reachable (Zubaroglu and Atalay 2020,
page 9). Recent points weigh more than old ones through a
damped window with the decay function ``f(t) = 2 ** (-lam * t)``
(Section 2.3.1, page 5). A micro-cluster whose cardinality is far
below the average is an outlier (Section 2.4, page 6).

The stream is summarised with two kinds of micro-clusters:

- potential micro-clusters (pMCs), with weight at least ``beta * mu``,
  which are the real clusters;
- outlier micro-clusters (oMCs), with weight below ``beta * mu``, which
  are the candidates.

A new point is added to the nearest pMC within ``eps`` if any, else to
the nearest oMC within ``eps``, else it opens a new oMC. An oMC whose
weight reaches ``beta * mu`` is promoted to pMC; a pMC whose weight
drops below ``beta * mu`` is removed, and an oMC whose weight drops
below ``beta * mu / 4`` is removed. The last threshold is a choice of
this implementation: the review (Section 2.4, page 6) defines an
outlier as a cluster whose cardinality is much smaller than the
average and does not give a numeric threshold, so a quarter of
``beta * mu`` was chosen as a small fraction. In the same section the
review is also silent on how to reduce the weight over time for an
oMC versus a pMC, so the same decay rate is used for both.

The offline step :meth:`macro_clusters` merges reachable pMCs on
demand into macro-clusters, which is the final answer of the
algorithm. Reachability uses ``eps`` as the distance between two
centres; the review (page 9) states the sum-of-radii rule,
but each pMC has radius at most ``eps`` (points are only added within
``eps``), so the sum is at most ``2 eps`` and ``eps`` is the
conservative choice. The macro-cluster centre is the weight-weighted
mean of its pMCs.

The weights and centres use the real timestamps passed to
``learn_one``.

Examples:
    >>> from dense_armor.utility.cluster.denstream import DenStream
    >>> ds = DenStream(eps=0.5, beta=0.4, mu=1.0, decay=0.1)
    >>> for i in range(30):
    ...     _ = ds.learn_one({"x": [0.0, 0.0]}, t=float(i))
    >>> clusters, _ = ds.macro_clusters()
    >>> len(clusters)
    1

References:
    Zubaroglu, A., Atalay, V. (2020). Data stream clustering: a
        review. arXiv:2007.10781. Sections 2.3.1 (page 5), 2.4 (page 6);
        density reachability on page 9.
"""
from typing import Any

import numpy as np

from dense_armor.roles import Transformer


def _vec(x: Any) -> np.ndarray | None:
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


class _MC:
    """A micro-cluster: centre, weight, last timestamp, count."""

    __slots__ = ("c", "n", "t", "w")

    def __init__(self, c: np.ndarray, w: float, t: float) -> None:
        self.c = c
        self.w = w
        self.t = t
        self.n = 1


class DenStream(Transformer):
    """Density-based stream clustering with a damped window.

    Potential and outlier micro-clusters fade with time; new points
    join the nearest reachable micro-cluster, and isolated points stay
    in the outlier list. The final answer comes from
    :meth:`macro_clusters`.

    Memory grows with the number of micro-clusters kept: bounded by
    the effective window set by ``decay`` and ``mu``, so the footprint
    is the size of the current micro-cluster lists.

    Args:
        eps: neighbourhood radius.
        beta: multiplier that separates pMCs from oMCs.
        mu: minimum weight of a pMC.
        decay: decay rate ``lam`` of the damped window.

    Raises:
        ValueError: if any parameter is not positive, or ``beta`` is not
            in ``(0, 1]``.
    """

    budget_s = 1e-3
    memory_class = "O(window)"

    def __init__(
        self,
        eps: float = 0.5,
        beta: float = 0.2,
        mu: float = 1.0,
        decay: float = 0.1,
    ) -> None:
        if eps <= 0:
            raise ValueError(f"eps must be > 0, got {eps}")
        if not 0.0 < beta <= 1.0:
            raise ValueError(f"beta must be in (0, 1], got {beta}")
        if mu <= 0:
            raise ValueError(f"mu must be > 0, got {mu}")
        if decay <= 0:
            raise ValueError(f"decay must be > 0, got {decay}")
        self.eps = eps
        self.beta = beta
        self.mu = mu
        self.decay = decay
        self.pmc_: list[_MC] = []
        self.omc_: list[_MC] = []
        self.n_seen_ = 0
        self.n_missing_ = 0

    def _unit_test_skips(self) -> set:
        return set()

    @property
    def n_missing(self) -> int:
        return self.n_missing_

    def _advance(self, mc: _MC, t: float) -> None:
        if t > mc.t:
            mc.w *= 2.0 ** (-self.decay * (t - mc.t))
            mc.t = t

    def _merge(self, mc: _MC, v: np.ndarray, t: float) -> None:
        self._advance(mc, t)
        new_w = mc.w + 1.0
        mc.c = (mc.w * mc.c + v) / new_w
        mc.w = new_w
        mc.n += 1

    def _nearest(self, pool: list[_MC], v: np.ndarray) -> tuple[_MC | None, float]:
        if not pool:
            return None, float("inf")
        best: _MC | None = None
        best_d = float("inf")
        for mc in pool:
            d = float(np.linalg.norm(mc.c - v))
            if d < best_d:
                best_d = d
                best = mc
        return best, best_d

    def _prune(self, t: float) -> None:
        thresh_p = self.beta * self.mu
        thresh_o = thresh_p / 4.0
        for pool, thresh in ((self.pmc_, thresh_p), (self.omc_, thresh_o)):
            keep: list[_MC] = []
            for mc in pool:
                self._advance(mc, t)
                if mc.w >= thresh:
                    keep.append(mc)
            pool[:] = keep

    def learn_one(
        self, x: Any, y: Any = None, t: float | None = None
    ) -> "DenStream":
        """Process one sample: fade, join, promote, prune."""
        self._time_step(t)
        v = _vec(x)
        if v is None:
            self.n_missing_ += 1
            return self
        tt = float(t) if t is not None else float(self.n_seen_)
        self.n_seen_ += 1
        p, dp = self._nearest(self.pmc_, v)
        if p is not None and dp <= self.eps:
            self._merge(p, v, tt)
        else:
            o, do = self._nearest(self.omc_, v)
            if o is not None and do <= self.eps:
                self._merge(o, v, tt)
                if o.w >= self.beta * self.mu:
                    self.omc_.remove(o)
                    self.pmc_.append(o)
            else:
                self.omc_.append(_MC(v.copy(), 1.0, tt))
        self._prune(tt)
        return self

    def transform_one(self, x: Any, t: float | None = None) -> dict:
        """Return the nearest pMC index within ``eps``, or ``-1``."""
        v = _vec(x)
        if v is None or not self.pmc_:
            return {"cluster": -1, "distance": float("inf")}
        best_i = -1
        best_d = float("inf")
        for i, mc in enumerate(self.pmc_):
            d = float(np.linalg.norm(mc.c - v))
            if d < best_d:
                best_d = d
                best_i = i
        if best_d > self.eps:
            return {"cluster": -1, "distance": best_d}
        return {"cluster": best_i, "distance": best_d}

    def macro_clusters(self) -> tuple[list[dict], list[int]]:
        """Merge reachable pMCs into macro-clusters on demand.

        Two pMCs are directly density reachable when the distance
        between their centres is at most ``eps`` (see the module
        docstring for the choice). The macro-cluster centre is the
        weight-weighted mean of its pMCs.

        Returns:
            A tuple ``(clusters, labels)``: ``clusters`` is a list of
            dicts with keys ``"center"``, ``"weight"`` and
            ``"n_pmc"``, and ``labels`` assigns each pMC in
            :attr:`pmc_` to a macro-cluster index.
        """
        n = len(self.pmc_)
        labels = [-1] * n
        clusters: list[dict] = []
        for i in range(n):
            if labels[i] != -1:
                continue
            cid = len(clusters)
            stack = [i]
            labels[i] = cid
            members: list[int] = []
            while stack:
                j = stack.pop()
                members.append(j)
                cj = self.pmc_[j].c
                for k in range(n):
                    if labels[k] != -1:
                        continue
                    if float(np.linalg.norm(cj - self.pmc_[k].c)) <= self.eps:
                        labels[k] = cid
                        stack.append(k)
            ws = np.array([self.pmc_[m].w for m in members])
            cs = np.stack([self.pmc_[m].c for m in members])
            center = (ws[:, None] * cs).sum(axis=0) / ws.sum()
            clusters.append(
                {
                    "center": center.tolist(),
                    "weight": float(ws.sum()),
                    "n_pmc": len(members),
                }
            )
        return clusters, labels
