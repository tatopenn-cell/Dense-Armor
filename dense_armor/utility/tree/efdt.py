"""Hoeffding Anytime Tree classifier.

The Hoeffding tree (Domingos and Hulten 2000) decides to split a leaf
only when the best candidate attribute beats the second best by more
than the Hoeffding bound, and it never revisits that decision. The
Hoeffding Anytime Tree (HATT) of Manapragada et al. 2018 changes both
points:

- a leaf splits as soon as the best candidate beats *no split* by
  more than the bound (Manapragada et al. 2018, Function 3.2);
- an internal node re-evaluates its own split every ``grace_period``
  samples that reach that node, and replaces the split when the best
  attribute beats the current one by more than the bound
  (Function 3.3). When no split is significantly better, the node
  becomes a leaf again.

Every node on the path of a sample updates its own sufficient
statistics in ``learn_one`` (Algorithm 3.1, the ``foreach node in
path`` loop).

Args:
    grace_period: minimum samples at a node before a split attempt and
        the interval between re-evaluations of an internal node.
    delta: one minus the confidence of the Hoeffding bound.
    tau: tie threshold.
    max_depth: maximum depth, or ``None``.
    max_nodes: maximum number of nodes. Default ``10000``.
    leaf: ``"nb"`` or ``"majority"``.

Raises:
    ValueError: on a bad ``leaf`` or a non-positive parameter.

Examples:
    >>> import numpy as np
    >>> from dense_armor.utility.tree.efdt import HoeffdingAnytimeTreeClassifier
    >>> rng = np.random.default_rng(0)
    >>> h = HoeffdingAnytimeTreeClassifier(grace_period=30)
    >>> for _ in range(400):
    ...     x = float(rng.normal(0.0 if rng.random() < 0.5 else 5.0, 0.4))
    ...     y = 0 if x < 2.5 else 1
    ...     _ = h.learn_one({"x": x}, y=y)
    >>> h.predict_one({"x": 0.1}) != h.predict_one({"x": 5.0})
    True

References:
    Manapragada, C., Webb, G. I., Salehi, M. (2018). Extremely fast
        decision tree. In KDD, Algorithm 3.1, Functions 3.2 and 3.3.
    Domingos, P., Hulten, G. (2000). Mining high-speed data streams.
        In KDD.
"""

import math
from typing import Any

from dense_armor.utility.tree.hoeffding import (
    HoeffdingTreeClassifier,
    _as_dict,
    _Node,
)


class HoeffdingAnytimeTreeClassifier(HoeffdingTreeClassifier):
    """Hoeffding Anytime Tree classifier.

    Same interface and same leaf classifiers as
    :class:`HoeffdingTreeClassifier`. Memory grows with the number of
    nodes; ``max_nodes`` (default ``10000``) bounds the total.

    ``budget_s`` = 1e-3 s: p99 of ``learn_one`` measured at 2.8e-5 s on a
    2000-sample, two-feature stream on a desktop CPU (default
    parameters); the budget leaves room for slower machines.
    """

    budget_s = 1e-3

    def __init__(
        self,
        grace_period: int = 200,
        delta: float = 1e-7,
        tau: float = 0.05,
        max_depth: int | None = None,
        max_nodes: int | None = 10000,
        leaf: str = "nb",
    ) -> None:
        super().__init__(
            grace_period=grace_period,
            delta=delta,
            tau=tau,
            max_depth=max_depth,
            max_nodes=max_nodes,
            leaf=leaf,
        )
        self.n_reevals_ = 0
        self.n_replaced_ = 0
        self.n_collapsed_ = 0

    def _bound(self, node: _Node) -> float:
        k = len(node.class_counts_)
        if k < 2:
            return math.inf
        r = math.log2(k)
        return math.sqrt(r * r * math.log(1.0 / self.delta) / (2.0 * node.n_))

    def _try_split(self, node: _Node) -> bool:
        if node.n_ < self.grace_period or not self._can_grow(node):
            return False
        k = len(node.class_counts_)
        if k < 2:
            return False
        best, f, s, _ = self._best_split(node)
        if f is None or s is None or best <= 0.0:
            return False
        eps = self._bound(node)
        if best > eps or eps < self.tau:
            node.split_feature_ = f
            node.split_threshold_ = s
            node.left_ = self._new_leaf(node.depth + 1)
            node.right_ = self._new_leaf(node.depth + 1)
            return True
        return False

    def _current_gain(self, node: _Node) -> float:
        f = node.split_feature_
        s = node.split_threshold_
        if f is None or s is None:
            return 0.0
        g = self._split_gain(node, f, s)
        return 0.0 if g is None else g

    def _collapse(self, node: _Node) -> None:
        node.split_feature_ = None
        node.split_threshold_ = None
        node.left_ = None
        node.right_ = None

    def _re_evaluate(self, node: _Node) -> None:
        if node.is_leaf or not self._can_grow(node):
            return
        k = len(node.class_counts_)
        if k < 2:
            return
        self.n_reevals_ += 1
        best, f, s, _ = self._best_split(node)
        current = self._current_gain(node)
        eps = self._bound(node)
        if f is not None and s is not None and best > current + eps:
            node.split_feature_ = f
            node.split_threshold_ = s
            node.left_ = self._new_leaf(node.depth + 1)
            node.right_ = self._new_leaf(node.depth + 1)
            self.n_replaced_ += 1
            return
        if best < eps and current < eps:
            self._collapse(node)
            self.n_collapsed_ += 1

    def learn_one(
        self, x: Any, y: Any, t: float | None = None
    ) -> "HoeffdingAnytimeTreeClassifier":
        self._time_step(t)
        d = _as_dict(x)
        if not d or y is None:
            self.n_missing_ += 1
            return self
        for v in d.values():
            if v is None:
                self.n_missing_ += 1
                return self
            try:
                fv = float(v)
            except (TypeError, ValueError):
                self.n_missing_ += 1
                return self
            if not math.isfinite(fv):
                self.n_missing_ += 1
                return self
        self.classes_.add(y)
        if self.root_ is None:
            self.root_ = self._new_leaf(0)
        path = self._path(d)
        for n in path:
            self._update_leaf(n, d, y)
        self.n_seen_ += 1
        for n in path:
            if n.is_leaf:
                self._try_split(n)
            elif n.n_ % self.grace_period == 0:
                self._re_evaluate(n)
        return self
