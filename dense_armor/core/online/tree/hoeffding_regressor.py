"""Hoeffding tree regressor, with the variance reduction as the split criterion.

The regression counterpart of :class:`HoeffdingTreeClassifier`. A leaf keeps
the count, the sum and the sum of squares of the target, and for every
feature an observer that can tell the best split of that feature. The merit
of a split is the reduction of the variance of the target,

.. math::

    VR = \\mathrm{Var}(S) - \\frac{n_L}{n}\\mathrm{Var}(S_L) - \\frac{n_R}{n}\\mathrm{Var}(S_R).

The merit used in the test is the *relative* reduction, ``VR / Var(S)``, which
lies in ``[0, 1]`` whatever the scale of the target, so the range of the merit
in the Hoeffding bound is 1. The tree splits a leaf when the best merit exceeds
the larger of the second-best merit and 0 (the merit of not splitting) by more
than ``epsilon``, or when ``epsilon < tau`` (a tie); so a leaf whose best split
reduces the variance by no more than noise does is not split. The leaf predicts
the mean of its target.

**Numeric features.** ``splitter="ebst"`` (default) keeps every distinct value
with its target sums and evaluates every boundary exactly (the E-BST of the
paper above); ``splitter="quantile"`` quantizes the values into slots of a
fixed width and evaluates the boundaries between occupied slots (the quantizer
of Mastelini and de Carvalho 2021), with a memory bounded by the number of
slots. **Categorical features** are those named in ``nominal_attributes`` (or all
the features whose first value is a non-numeric string, with ``"auto"``);
the split is "the value is this category" against the rest.

**Memory.** As in the classifier: ``max_active_leaves`` (default 1000) keeps the
statistics of the leaves that have made the largest total absolute error, and
the other leaves keep only the mean of the target. The tree has no node cap by
default.

Examples:
    >>> import numpy as np
    >>> from dense_armor.core.online.tree.hoeffding_regressor import HoeffdingTreeRegressor
    >>> rng = np.random.default_rng(0)
    >>> t = HoeffdingTreeRegressor(grace_period=50)
    >>> for _ in range(1500):
    ...     x = float(rng.random())
    ...     _ = t.learn_one({"x": x}, (5.0 if x > 0.5 else 1.0) + float(rng.normal(0, 0.1)))
    >>> round(t.predict_one({"x": 0.9})), round(t.predict_one({"x": 0.1}))
    (5, 1)

References:
    Ikonomovska, E., Gama, J., Dzeroski, S. (2011). Learning model trees from
        evolving data streams. Data Mining and Knowledge Discovery 23(1).
    Mastelini, S. M., de Carvalho, A. C. P. L. F. (2021). Using dynamical
        quantization to perform split attempts in online tree regressors.
        Pattern Recognition Letters 145.
"""

import math
from typing import Any

from dense_armor.core.online.tree.hoeffding import _as_dict
from dense_armor.core.online.tree.splitters import (
    TARGET_SPLITTERS,
    TargetNominalObserver,
    TargetNumericObserver,
)
from dense_armor.roles import Regressor


class _RNode:
    """One node of the regression tree."""

    __slots__ = (
        "active_",
        "depth",
        "errors_",
        "feature_stats_",
        "left_",
        "n_",
        "right_",
        "split_category_",
        "split_feature_",
        "split_threshold_",
        "sum_",
        "sum_sq_",
    )

    def __init__(self, depth: int) -> None:
        self.depth = depth
        self.n_ = 0
        self.sum_ = 0.0
        self.sum_sq_ = 0.0
        self.errors_ = 0.0
        self.active_ = True
        self.feature_stats_: dict = {}
        self.split_feature_: str | None = None
        self.split_threshold_: float | None = None
        self.split_category_: Any = None
        self.left_: _RNode | None = None
        self.right_: _RNode | None = None

    @property
    def is_leaf(self) -> bool:
        return self.split_feature_ is None

    @property
    def mean(self) -> float:
        return self.sum_ / self.n_ if self.n_ > 0 else 0.0


class HoeffdingTreeRegressor(Regressor):
    """Hoeffding tree for incremental regression, split by the variance reduction.

    Args:
        grace_period: samples at a leaf between two split attempts.
        delta: one minus the confidence of the Hoeffding bound.
        tau: tie threshold.
        max_depth: maximum depth, or ``None`` (default).
        max_nodes: maximum number of nodes, or ``None`` (default).
        splitter: ``"ebst"`` (default) or ``"quantile"``.
        nominal_attributes: names of the categorical features, or ``"auto"``,
            or ``None`` (all numeric).
        max_active_leaves: leaves that keep statistics, or ``None`` for no
            limit. Default 1000.
        memory_period: samples between two memory checks. Default 1000.
        quantile_radius: slot width of the ``"quantile"`` splitter, or ``None``
            for a third of the standard deviation of the first 30 values.

    Raises:
        ValueError: on a bad ``splitter`` or a non-positive parameter.
    """

    budget_s = 1e-2
    memory_class = "O(window)"

    def __init__(
        self,
        grace_period: int = 200,
        delta: float = 1e-7,
        tau: float = 0.05,
        max_depth: int | None = None,
        max_nodes: int | None = None,
        splitter: str = "ebst",
        nominal_attributes: Any = None,
        max_active_leaves: int | None = 1000,
        memory_period: int = 1000,
        quantile_radius: float | None = None,
    ) -> None:
        if grace_period < 1:
            raise ValueError(f"grace_period must be >= 1, got {grace_period}")
        if not 0.0 < delta < 1.0:
            raise ValueError(f"delta must be in (0, 1), got {delta}")
        if tau < 0.0:
            raise ValueError(f"tau must be >= 0, got {tau}")
        if max_depth is not None and max_depth < 1:
            raise ValueError(f"max_depth must be >= 1, got {max_depth}")
        if max_nodes is not None and max_nodes < 1:
            raise ValueError(f"max_nodes must be >= 1, got {max_nodes}")
        if splitter not in TARGET_SPLITTERS:
            raise ValueError(
                f"splitter must be one of {TARGET_SPLITTERS}, got {splitter!r}"
            )
        if max_active_leaves is not None and max_active_leaves < 1:
            raise ValueError(f"max_active_leaves must be >= 1, got {max_active_leaves}")
        if memory_period < 1:
            raise ValueError(f"memory_period must be >= 1, got {memory_period}")
        if quantile_radius is not None and quantile_radius <= 0.0:
            raise ValueError(f"quantile_radius must be > 0, got {quantile_radius}")
        if isinstance(nominal_attributes, str) and nominal_attributes != "auto":
            raise ValueError(
                f"nominal_attributes must be a list of names or 'auto', got {nominal_attributes!r}"
            )
        self.grace_period = grace_period
        self.delta = delta
        self.tau = tau
        self.max_depth = max_depth
        self.max_nodes = max_nodes
        self.splitter = splitter
        self.nominal_attributes = nominal_attributes
        self.max_active_leaves = max_active_leaves
        self.memory_period = memory_period
        self.quantile_radius = quantile_radius
        self._auto_nominal = nominal_attributes == "auto"
        self._nominal_set = (
            frozenset()
            if isinstance(nominal_attributes, str)
            else frozenset(nominal_attributes or ())
        )
        self._kinds_: dict = {}
        self.root_: _RNode | None = None
        self.n_nodes_ = 0
        self.n_seen_ = 0
        self.n_missing_ = 0

    @property
    def n_missing(self) -> int:
        return self.n_missing_

    @property
    def depth_(self) -> int:
        return self._depth(self.root_)

    @property
    def n_active_leaves_(self) -> int:
        return sum(1 for lf in self._leaves() if lf.active_)

    def _depth(self, node: _RNode | None) -> int:
        if node is None or node.is_leaf:
            return 0
        return 1 + max(self._depth(node.left_), self._depth(node.right_))

    def _leaves(self):
        stack = [self.root_] if self.root_ is not None else []
        while stack:
            node = stack.pop()
            if node.is_leaf:
                yield node
            else:
                stack.append(node.right_)
                stack.append(node.left_)

    def _new_leaf(self, depth: int) -> _RNode:
        node = _RNode(depth)
        self.n_nodes_ += 1
        return node

    def _kind_of(self, f: str, v: Any) -> str:
        if f in self._nominal_set:
            return "nom"
        if self._auto_nominal and isinstance(v, (str, bytes)):
            try:
                float(v)
            except ValueError:
                return "nom"
        return "num"

    def _validate(self, x: Any, y: Any) -> tuple | None:
        d = _as_dict(x)
        try:
            fy = float(y)
        except (TypeError, ValueError):
            fy = math.nan
        if not d or not math.isfinite(fy):
            self.n_missing_ += 1
            return None
        out: dict = {}
        new_kinds: dict = {}
        for f, v in d.items():
            if v is None:
                self.n_missing_ += 1
                return None
            kind = self._kinds_.get(f) or new_kinds.get(f) or self._kind_of(f, v)
            if kind == "num":
                try:
                    fv = float(v)
                except (TypeError, ValueError):
                    self.n_missing_ += 1
                    return None
                if not math.isfinite(fv):
                    self.n_missing_ += 1
                    return None
                out[f] = fv
            else:
                try:
                    hash(v)
                except TypeError:
                    self.n_missing_ += 1
                    return None
                if isinstance(v, float) and v != v:
                    self.n_missing_ += 1
                    return None
                out[f] = v
            if f not in self._kinds_:
                new_kinds[f] = kind
        self._kinds_.update(new_kinds)
        return out, fy

    def _goes_left(self, node: _RNode, v: Any) -> bool:
        if node.split_category_ is not None:
            return bool(v == node.split_category_)
        return float(v) <= node.split_threshold_

    def _path(self, x: dict) -> list:
        path: list = []
        cur: _RNode | None = self.root_
        while cur is not None:
            path.append(cur)
            if cur.split_feature_ is None:
                return path
            v = x.get(cur.split_feature_)
            if v is None:
                return path
            try:
                left = self._goes_left(cur, v)
            except (TypeError, ValueError):
                return path
            cur = cur.left_ if left else cur.right_
        return path

    def _new_observer(self, f: str) -> Any:
        if self._kinds_.get(f) == "nom":
            return TargetNominalObserver()
        return TargetNumericObserver(self.splitter, self.quantile_radius)

    def _update_leaf(self, node: _RNode, x: dict, y: float) -> None:
        node.errors_ += abs(y - node.mean) if node.n_ > 0 else 0.0
        node.n_ += 1
        node.sum_ += y
        node.sum_sq_ += y * y
        if not node.active_:
            return
        fs = node.feature_stats_
        for f, v in x.items():
            obs = fs.get(f)
            if obs is None:
                obs = self._new_observer(f)
                fs[f] = obs
            obs.update(v, y)

    def _best_split(self, node: _RNode):
        best = -1.0
        best_f: str | None = None
        best_s: Any = None
        second = 0.0
        var = max(0.0, node.sum_sq_ / node.n_ - node.mean**2) if node.n_ > 0 else 0.0
        if var <= 1e-12:
            return best, best_f, best_s, second
        for f, obs in node.feature_stats_.items():
            vr, s = obs.best_candidate()
            vr /= var
            if s is None or vr <= 0.0:
                continue
            if vr > best:
                second = max(best, 0.0)
                best, best_f, best_s = vr, f, s
            elif vr > second:
                second = vr
        return best, best_f, best_s, second

    def _can_grow(self, node: _RNode) -> bool:
        depth_ok = self.max_depth is None or node.depth < self.max_depth
        nodes_ok = self.max_nodes is None or self.n_nodes_ + 2 <= self.max_nodes
        return depth_ok and nodes_ok

    def _try_split(self, node: _RNode) -> bool:
        if not node.active_ or node.n_ < self.grace_period or not self._can_grow(node):
            return False
        best, f, s, second = self._best_split(node)
        if f is None or s is None or best <= 0.0:
            return False
        eps = math.sqrt(math.log(1.0 / self.delta) / (2.0 * node.n_))
        if (best - second) > eps or eps < self.tau:
            node.split_feature_ = f
            if self._kinds_.get(f) == "nom":
                node.split_category_ = s
            else:
                node.split_threshold_ = s
            node.left_ = self._new_leaf(node.depth + 1)
            node.right_ = self._new_leaf(node.depth + 1)
            node.feature_stats_ = {}
            return True
        return False

    def _enforce_memory(self) -> None:
        cap = self.max_active_leaves
        if cap is None or self.root_ is None:
            return
        leaves = list(self._leaves())
        if len(leaves) <= cap:
            for lf in leaves:
                lf.active_ = True
            return
        leaves.sort(
            key=lambda lf: (lf.n_ < self.grace_period, lf.errors_), reverse=True
        )
        for i, lf in enumerate(leaves):
            if i < cap:
                lf.active_ = True
            elif lf.active_:
                lf.active_ = False
                lf.feature_stats_ = {}

    def learn_one(
        self, x: dict, y: Any, t: float | None = None
    ) -> "HoeffdingTreeRegressor":
        """Update the tree with one sample."""
        self._time_step(t)
        checked = self._validate(x, y)
        if checked is None:
            return self
        d, fy = checked
        if self.root_ is None:
            self.root_ = self._new_leaf(0)
        leaf = self._path(d)[-1]
        self._update_leaf(leaf, d, fy)
        self.n_seen_ += 1
        if leaf.is_leaf and leaf.n_ % self.grace_period == 0:
            self._try_split(leaf)
        if (
            self.max_active_leaves is not None
            and self.n_seen_ % self.memory_period == 0
        ):
            self._enforce_memory()
        return self

    def predict_one(self, x: dict, t: float | None = None, return_std: bool = False):
        """The mean of the target at the reached leaf (and its standard deviation)."""
        if self.root_ is None:
            return (0.0, 0.0) if return_std else 0.0
        leaf = self._path(_as_dict(x))[-1]
        mean = leaf.mean
        if not return_std:
            return mean
        if leaf.n_ < 2:
            return mean, 0.0
        var = max(0.0, leaf.sum_sq_ / leaf.n_ - mean * mean)
        return mean, math.sqrt(var)

    def explain_one(self, x: dict, t: float | None = None) -> list:
        """The path from the root to the leaf of ``x``, as a list of dicts."""
        if self.root_ is None:
            return []
        d = _as_dict(x)
        cur: _RNode | None = self.root_
        out: list = []
        while cur is not None and not cur.is_leaf:
            f = cur.split_feature_
            v = d.get(f)
            if v is None:
                break
            try:
                left = self._goes_left(cur, v)
            except (TypeError, ValueError):
                break
            side = "left" if left else "right"
            if cur.split_category_ is not None:
                out.append(
                    {"feature": f, "category": cur.split_category_, "side": side}
                )
            else:
                out.append(
                    {"feature": f, "threshold": cur.split_threshold_, "side": side}
                )
            cur = cur.left_ if left else cur.right_
        return out
