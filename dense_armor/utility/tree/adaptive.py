"""Hoeffding Adaptive Tree classifier.

The Hoeffding tree never revisits a split. The Hoeffding Adaptive Tree
(HAT) of Bifet and Gavalda (2009) adds a drift detector at each node:
when the detector warns, an alternate subtree grows in the background
on the recent samples; when the alternate is significantly more
accurate than the main one, it replaces the corresponding branch.

The detector and the error counters live on the node, not in a dict
keyed by ``id(node)``, so clone, pickle and state round-trips work.

Args:
    grace_period: minimum samples at a node before a split attempt.
    delta: one minus the confidence of the Hoeffding bound for splits.
    tau: tie threshold.
    max_depth: maximum depth, or ``None``.
    max_nodes: maximum number of nodes. Default ``10000``.
    leaf: ``"nb"`` or ``"majority"``.
    drift_detector: a fresh ``DriftDetector`` instance, cloned per node.
    delta_alt: one minus the confidence of the alternate bound.
    kappa_alt: minimum samples for an alternate to be considered.
    error_alpha: smoothing of the per-node error estimate.

Raises:
    ValueError: on a bad ``leaf`` or a non-positive parameter.

Examples:
    >>> import numpy as np
    >>> from dense_armor.utility.drift.adwin import ADWIN
    >>> from dense_armor.utility.tree.adaptive import HoeffdingAdaptiveTreeClassifier
    >>> rng = np.random.default_rng(0)
    >>> h = HoeffdingAdaptiveTreeClassifier(
    ...     grace_period=30, drift_detector=ADWIN()
    ... )
    >>> for _ in range(400):
    ...     x = float(rng.normal(0.0 if rng.random() < 0.5 else 5.0, 0.4))
    ...     y = 0 if x < 2.5 else 1
    ...     _ = h.learn_one({"x": x}, y=y)
    >>> h.predict_one({"x": 0.1}) != h.predict_one({"x": 5.0})
    True

References:
    Bifet, A., Gavalda, R. (2009). Adaptive learning from evolving
        data streams. In IDA.
    Esteban, A., Cano, A., Zafra, A., Ventura, S. (2024). Hoeffding
        adaptive trees for multi-label classification on data streams.
    Domingos, P., Hulten, G. (2000). Mining high-speed data streams.
        In KDD.
"""

import contextlib
import copy
import math
from typing import Any, cast

from dense_armor.roles import DriftDetector
from dense_armor.utility.tree.hoeffding import (
    HoeffdingTreeClassifier,
    _as_dict,
    _Node,
)


def _clone_detector(detector: Any) -> Any:
    """Return a fresh detector of the same kind."""
    clone = getattr(detector, "clone", None)
    if callable(clone):
        with contextlib.suppress(Exception):
            return clone()
    cls = type(detector)
    try:
        return cls()
    except TypeError:
        return copy.deepcopy(detector)


class _ANode(_Node):
    """A node of the adaptive tree."""

    __slots__ = (
        "alt_",
        "alt_n_",
        "alt_wrong_",
        "detector_",
        "err_",
        "err_n_",
        "main_wrong_",
    )

    def __init__(self, depth: int, detector: Any) -> None:
        super().__init__(depth)
        self.detector_ = _clone_detector(detector)
        self.err_ = 0.0
        self.err_n_ = 0
        self.alt_: _ANode | None = None
        self.alt_n_ = 0
        self.alt_wrong_ = 0
        self.main_wrong_ = 0


class HoeffdingAdaptiveTreeClassifier(HoeffdingTreeClassifier):
    """Hoeffding Adaptive Tree classifier.

    Memory grows with the number of nodes of the main tree and of any
    alternate subtrees; ``max_nodes`` (default ``10000``) bounds the
    total.

    ``budget_s`` = 5e-3 s: p99 of ``learn_one`` measured at 3.7e-4 s on a
    2000-sample, two-feature stream on a desktop CPU (default
    parameters); the budget leaves room for slower machines.
    """

    budget_s = 5e-3

    def __init__(
        self,
        grace_period: int = 200,
        delta: float = 1e-7,
        tau: float = 0.05,
        max_depth: int | None = None,
        max_nodes: int | None = 10000,
        leaf: str = "nb",
        drift_detector: DriftDetector | None = None,
        delta_alt: float = 0.01,
        kappa_alt: int = 100,
        error_alpha: float = 0.005,
    ) -> None:
        super().__init__(
            grace_period=grace_period,
            delta=delta,
            tau=tau,
            max_depth=max_depth,
            max_nodes=max_nodes,
            leaf=leaf,
        )
        if drift_detector is None:
            raise ValueError("drift_detector is required")
        if not 0.0 < delta_alt < 1.0:
            raise ValueError(f"delta_alt must be in (0, 1), got {delta_alt}")
        if kappa_alt < 1:
            raise ValueError(f"kappa_alt must be >= 1, got {kappa_alt}")
        if not 0.0 < error_alpha <= 1.0:
            raise ValueError(f"error_alpha must be in (0, 1], got {error_alpha}")
        self.drift_detector = drift_detector
        self.delta_alt = delta_alt
        self.kappa_alt = kappa_alt
        self.error_alpha = error_alpha
        self.n_drifts_ = 0
        self.n_alt_starts_ = 0
        self.n_swaps_ = 0

    def _new_leaf(self, depth: int) -> _ANode:
        node = _ANode(depth, self.drift_detector)
        self.n_nodes_ += 1
        return node

    def _update_error(self, node: _ANode, correct: bool) -> None:
        node.err_n_ += 1
        r = 0.0 if correct else 1.0
        a = self.error_alpha
        node.err_ = (1.0 - a) * node.err_ + a * r

    def _alt_bound(self, n: int) -> float:
        """Hoeffding bound on the difference of two error rates over ``n``."""
        return math.sqrt(math.log(2.0 / self.delta_alt) / n)

    def _alt_more_accurate(self, node: _ANode) -> bool:
        """Swap test on the errors counted since the alternate started."""
        n = node.alt_n_
        if node.alt_ is None or n < self.kappa_alt:
            return False
        return (node.main_wrong_ - node.alt_wrong_) / n > self._alt_bound(n)

    def _alt_worse(self, node: _ANode) -> bool:
        n = node.alt_n_
        if node.alt_ is None or n < self.kappa_alt:
            return False
        return (node.alt_wrong_ - node.main_wrong_) / n > self._alt_bound(n)

    def _descend(self, node: _ANode, x: dict) -> _ANode | None:
        f = node.split_feature_
        s = node.split_threshold_
        if f is None or s is None:
            return None
        v = x.get(f)
        if v is None:
            return None
        try:
            fv = float(v)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(fv):
            return None
        nxt = node.left_ if fv <= s else node.right_
        if nxt is None:
            return None
        return cast(_ANode, nxt)

    def _path_of(self, root: _ANode, d: dict) -> list:
        path: list = []
        cur: _ANode | None = root
        while cur is not None:
            path.append(cur)
            if cur.is_leaf:
                return path
            cur = self._descend(cur, d)
        return path

    def _copy_alt_into(self, main: _ANode, alt: _ANode) -> None:
        main.class_counts_ = alt.class_counts_
        main.feature_stats_ = alt.feature_stats_
        main.split_feature_ = alt.split_feature_
        main.split_threshold_ = alt.split_threshold_
        main.left_ = alt.left_
        main.right_ = alt.right_
        main.n_ = alt.n_
        main.err_ = alt.err_
        main.err_n_ = alt.err_n_
        main.detector_ = _clone_detector(self.drift_detector)
        main.alt_ = None
        main.alt_n_ = 0
        main.alt_wrong_ = 0
        main.main_wrong_ = 0

    def _grow_alt_on(self, node: _ANode, d: dict, y: Any) -> None:
        if node.alt_ is None:
            return
        path = self._path_of(node.alt_, d)
        leaf = path[-1]
        self._update_leaf(leaf, d, y)
        if leaf.is_leaf:
            self._try_split(leaf)
        pred = self._predict_proba_leaf(leaf, d)
        chosen = max(pred.items(), key=lambda kv: kv[1])[0] if pred else None
        node.alt_n_ += 1
        node.alt_wrong_ += 0 if chosen == y else 1
        if self._alt_more_accurate(node):
            self._copy_alt_into(node, node.alt_)
            self.n_swaps_ += 1
        elif self._alt_worse(node):
            node.alt_ = None
            node.alt_n_ = 0
            node.alt_wrong_ = 0
            node.main_wrong_ = 0

    def learn_one(
        self, x: Any, y: Any, t: float | None = None
    ) -> "HoeffdingAdaptiveTreeClassifier":
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
        leaf = path[-1]
        pred = self.predict_one(d)
        correct = pred == y
        for n in path:
            self._update_leaf(n, d, y)
        self.n_seen_ += 1
        if leaf.is_leaf:
            self._try_split(leaf)
        for n in path:
            if not isinstance(n, _ANode):
                continue
            self._update_error(n, correct)
            det = n.detector_
            if det is not None:
                det.update(0.0 if correct else 1.0, t=t)
                if getattr(det, "drift_detected", False):
                    self.n_drifts_ += 1
                    n.detector_ = _clone_detector(self.drift_detector)
                    if n.alt_ is None:
                        n.alt_ = _ANode(n.depth, self.drift_detector)
                        n.alt_n_ = 0
                        n.alt_wrong_ = 0
                        n.main_wrong_ = 0
                        self.n_alt_starts_ += 1
        for n in path:
            if isinstance(n, _ANode) and n.alt_ is not None:
                n.main_wrong_ += 0 if correct else 1
                self._grow_alt_on(n, d, y)
        return self
