"""Hoeffding tree classifier.

A non-linear classifier that grows one sample at a time. Each leaf
keeps enough statistics to estimate the gain of a possible split; when
the difference in gain between the best and the second-best candidate is
larger than the Hoeffding bound, the leaf is split.

The Hoeffding bound of equation 1 of Manapragada et al. 2018 is

.. math::

    \\epsilon = \\sqrt{\\frac{R^2 \\ln(1/\\delta)}{2 n}}

where ``R`` is the range of the split-evaluation function (``log2(K)`` for
the entropy and ``1 - 1/K`` for the Gini impurity, with ``K`` classes),
``delta`` is one minus the required confidence, and ``n`` is the number of
samples seen at the leaf.

**Split criterion.** ``criterion="entropy"`` (information gain, the
default) or ``"gini"`` (reduction of the Gini impurity).

**Numeric features.** A running Gaussian per class and per feature inside
each leaf (also the naive-Bayes leaf). The candidates of a split come from
``splitter``: ``"class_means"`` (default; the midpoints between the means of
the classes), ``"gaussian"`` (``n_splits`` thresholds between the smallest and
the largest value, with the class shares from the normal CDF), ``"ebst"``
(every boundary between two consecutive distinct values, on exact counts) or
``"quantile"`` (the boundaries between the slots of a quantizer); see
:mod:`dense_armor.core.online.tree.splitters`.

**Categorical features.** A feature is categorical when its name is in
``nominal_attributes``, or when ``nominal_attributes`` is ``"auto"`` and its
first value is a string that is not a number (without either, a value that is
not a number makes the sample count as missing, as before). A categorical feature is split as "the value is this category"
against all the others; the leaf classifier uses a Laplace-smoothed
``P(category | class)``.

**Memory.** ``max_active_leaves`` (default ``1000``) bounds the statistics of
the tree: every ``memory_period`` samples the leaves are ranked by the errors
they have made (the promise of a leaf, ``p_l * e_l``, of Domingos and Hulten
2000, since ``p_l * e_l`` is the share of the stream's errors made at the
leaf) and only the most promising ones keep their statistics and go on trying
to split. A deactivated leaf keeps counting the classes, so it still
predicts the majority class, and it is reactivated, with new statistics, when
it becomes one of the most promising. The tree itself is not capped (``max_nodes``
defaults to ``None``) and can grow to any depth. ``max_nodes`` and
``max_depth`` remain as hard limits when a user wants them.

Examples:
    >>> import numpy as np
    >>> from dense_armor.core.online.tree.hoeffding import HoeffdingTreeClassifier
    >>> rng = np.random.default_rng(0)
    >>> h = HoeffdingTreeClassifier(grace_period=30)
    >>> for _ in range(400):
    ...     x = float(rng.normal(0.0 if rng.random() < 0.5 else 5.0, 0.4))
    ...     y = 0 if x < 2.5 else 1
    ...     _ = h.learn_one({"x": x}, y=y)
    >>> h.predict_one({"x": 0.1}) != h.predict_one({"x": 5.0})
    True

    A categorical feature, with the Gini criterion:

    >>> h = HoeffdingTreeClassifier(
    ...     grace_period=30, criterion="gini", nominal_attributes=["colour"]
    ... )
    >>> for _ in range(400):
    ...     colour = "red" if rng.random() < 0.5 else "blue"
    ...     _ = h.learn_one({"colour": colour}, y=colour == "red")
    >>> h.predict_one({"colour": "red"}), h.predict_one({"colour": "blue"})
    (True, False)

References:
    Manapragada, C., Webb, G. I., Salehi, M. (2018). Extremely fast
        decision tree. In KDD.
    Domingos, P., Hulten, G. (2000). Mining high-speed data streams.
        In KDD.
"""

import math
from typing import Any

from dense_armor.core.online.tree._stats import _Gauss, _phi, _two_sum  # noqa: F401
from dense_armor.core.online.tree.splitters import (
    CRITERIA,
    SPLITTERS,
    NominalObserver,
    NumericObserver,
    merit_range,
)
from dense_armor.roles import Classifier


def _as_dict(x: Any) -> dict:
    if hasattr(x, "to_dict"):
        return dict(x.to_dict())
    try:
        return dict(x)
    except (TypeError, ValueError):
        return {}


def _entropy(counts: dict) -> float:
    total = sum(counts.values())
    if total <= 0.0:
        return 0.0
    h = 0.0
    for c in counts.values():
        if c > 0.0:
            p = c / total
            h -= p * math.log2(p)
    return h


def _observer(obj: Any) -> Any:
    """An observer; a plain ``{class: _Gauss}`` dict is read as a numeric one."""
    if hasattr(obj, "kind"):
        return obj
    ob = NumericObserver()
    dict.update(ob, obj)
    return ob


class _Node:
    """One node of the tree: a leaf or an internal split."""

    __slots__ = (
        "active_",
        "class_counts_",
        "depth",
        "errors_",
        "feature_stats_",
        "left_",
        "n_",
        "right_",
        "split_category_",
        "split_feature_",
        "split_threshold_",
    )

    def __init__(self, depth: int) -> None:
        self.depth = depth
        self.n_ = 0
        self.errors_ = 0
        self.active_ = True
        self.class_counts_: dict = {}
        self.feature_stats_: dict = {}
        self.split_feature_: str | None = None
        self.split_threshold_: float | None = None
        self.split_category_: Any = None
        self.left_: _Node | None = None
        self.right_: _Node | None = None

    @property
    def is_leaf(self) -> bool:
        return self.split_feature_ is None

    @property
    def split_test(self) -> Any:
        """The threshold of a numeric split, or the category of a categorical one."""
        if self.split_category_ is not None:
            return self.split_category_
        return self.split_threshold_


class HoeffdingTreeClassifier(Classifier):
    """Hoeffding tree for incremental classification.

    Memory is bounded by ``max_active_leaves``: only that many leaves keep
    statistics, so ``memory_class`` is ``"O(window)"`` in the statistics (the
    nodes themselves are small, and ``max_nodes`` can cap them).

    ``budget_s`` = 1e-3 s: p99 of ``learn_one`` measured at 2.3e-5 s on a
    2000-sample, two-feature stream on a desktop CPU (default
    parameters); the budget leaves room for slower machines.

    Args:
        grace_period: minimum number of samples at a leaf before a split
            is attempted.
        delta: one minus the confidence of the Hoeffding bound.
        tau: tie threshold.
        max_depth: maximum depth, or ``None`` (default).
        max_nodes: maximum number of nodes, or ``None`` (default).
        leaf: ``"nb"`` or ``"majority"``. With ``"nb"``, a leaf in which
            any class has fewer than two samples answers like
            ``"majority"``, because the Gaussian likelihood of that class
            cannot be estimated.
        criterion: ``"entropy"`` (information gain, default) or ``"gini"``.
        splitter: the candidates of a numeric split: ``"class_means"``
            (default), ``"gaussian"``, ``"ebst"`` or ``"quantile"``.
        n_splits: number of thresholds of the ``"gaussian"`` splitter.
            Default 10.
        nominal_attributes: names of the categorical features, or ``"auto"``
            to take as categorical every feature whose first value is a
            non-numeric string. Default ``None``: all the features are
            numeric.
        max_active_leaves: number of leaves that keep statistics, or ``None``
            for no limit. Default 1000.
        memory_period: samples between two memory checks. Default 1000.
        quantile_radius: slot width of the ``"quantile"`` splitter, or
            ``None`` for a third of the standard deviation of the first
            30 values.

    Raises:
        ValueError: on a bad ``leaf``, ``criterion`` or ``splitter``, or a
            non-positive parameter.
    """

    budget_s = 1e-3
    memory_class = "O(window)"
    _keep_internal_stats = False

    def __init__(
        self,
        grace_period: int = 200,
        delta: float = 1e-7,
        tau: float = 0.05,
        max_depth: int | None = None,
        max_nodes: int | None = None,
        leaf: str = "nb",
        dtype: Any = None,
        criterion: str = "entropy",
        splitter: str = "class_means",
        n_splits: int = 10,
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
        if leaf not in ("nb", "majority"):
            raise ValueError(f"leaf must be 'nb' or 'majority', got {leaf!r}")
        if criterion not in CRITERIA:
            raise ValueError(f"criterion must be one of {CRITERIA}, got {criterion!r}")
        if splitter not in SPLITTERS:
            raise ValueError(f"splitter must be one of {SPLITTERS}, got {splitter!r}")
        if n_splits < 1:
            raise ValueError(f"n_splits must be >= 1, got {n_splits}")
        if max_active_leaves is not None and max_active_leaves < 1:
            raise ValueError(f"max_active_leaves must be >= 1, got {max_active_leaves}")
        if memory_period < 1:
            raise ValueError(f"memory_period must be >= 1, got {memory_period}")
        if quantile_radius is not None and quantile_radius <= 0.0:
            raise ValueError(f"quantile_radius must be > 0, got {quantile_radius}")
        self.grace_period = grace_period
        self.delta = delta
        self.tau = tau
        self.max_depth = max_depth
        self.max_nodes = max_nodes
        self.leaf = leaf
        self.dtype = dtype
        self.criterion = criterion
        self.splitter = splitter
        self.n_splits = n_splits
        self.nominal_attributes = nominal_attributes
        self.max_active_leaves = max_active_leaves
        self.memory_period = memory_period
        self.quantile_radius = quantile_radius
        if isinstance(nominal_attributes, str) and nominal_attributes != "auto":
            raise ValueError(
                f"nominal_attributes must be a list of names or 'auto', got {nominal_attributes!r}"
            )
        self._auto_nominal = nominal_attributes == "auto"
        self._nominal_set = (
            frozenset()
            if isinstance(nominal_attributes, str)
            else frozenset(nominal_attributes or ())
        )
        self._kinds_: dict = {}
        self.root_: _Node | None = None
        self.n_nodes_ = 0
        self.classes_: set = set()
        self.features_: set = set()
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
        """Number of leaves that keep statistics and try to split."""
        return sum(1 for lf in self._leaves() if lf.active_)

    def _depth(self, node: _Node | None) -> int:
        if node is None or node.is_leaf:
            return 0
        return 1 + max(self._depth(node.left_), self._depth(node.right_))

    def _leaves(self):
        """The leaves of the main tree."""
        stack = [self.root_] if self.root_ is not None else []
        while stack:
            node = stack.pop()
            if node.is_leaf:
                yield node
            else:
                stack.append(node.right_)
                stack.append(node.left_)

    def _new_leaf(self, depth: int) -> _Node:
        node = _Node(depth)
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

    def _validate(self, x: Any, y: Any) -> dict | None:
        """The sample with its numeric values as floats, or ``None`` if it is missing something."""
        d = _as_dict(x)
        if not d or y is None:
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
        return out

    def _goes_left(self, node: _Node, v: Any) -> bool:
        if node.split_category_ is not None:
            return bool(v == node.split_category_)
        return float(v) <= node.split_threshold_

    def _path(self, x: dict) -> list:
        """Nodes from the root to the leaf of ``x``."""
        path: list = []
        cur: _Node | None = self.root_
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
            return NominalObserver()
        return NumericObserver(
            self.splitter, self.dtype, self.n_splits, self.quantile_radius
        )

    def _update_leaf(self, node: _Node, x: dict, y: Any) -> None:
        node.n_ += 1
        cc = node.class_counts_
        if not cc or max(cc, key=cc.get) != y:
            node.errors_ += 1
        cc[y] = cc.get(y, 0) + 1
        known = self.features_
        if node.active_:
            fs = node.feature_stats_
            for f, v in x.items():
                obs = fs.get(f)
                if obs is None:
                    obs = self._new_observer(f)
                    fs[f] = obs
                obs.update(v, y)
                if f not in known:
                    known.add(f)
        else:
            known.update(x)

    def _entropy(self, node: _Node) -> float:
        return _entropy(node.class_counts_)

    def _split_gain(self, node: _Node, f: str, s: Any):
        obs = node.feature_stats_.get(f)
        if not obs or node.n_ <= 0:
            return None
        obs = _observer(obs)
        return obs.test_gain(self.criterion, node.class_counts_, node.n_, s)

    def _best_split(self, node: _Node):
        best_gain = -1.0
        best_f: str | None = None
        best_s: Any = None
        second_gain = -1.0
        cc = node.class_counts_
        n = node.n_
        for f, obs in node.feature_stats_.items():
            f_best, f_s = _observer(obs).best_candidate(self.criterion, cc, n)
            if f_s is None or f_best < 0.0:
                continue
            if f_best > best_gain:
                second_gain = best_gain
                best_gain = f_best
                best_f = f
                best_s = f_s
            elif f_best > second_gain:
                second_gain = f_best
        return best_gain, best_f, best_s, second_gain

    def _can_grow(self, node: _Node) -> bool:
        depth_ok = self.max_depth is None or node.depth < self.max_depth
        nodes_ok = self.max_nodes is None or self.n_nodes_ + 2 <= self.max_nodes
        return depth_ok and nodes_ok

    def _set_split(self, node: _Node, f: str, test: Any) -> None:
        node.split_feature_ = f
        if self._kinds_.get(f) == "nom":
            node.split_category_ = test
            node.split_threshold_ = None
        else:
            node.split_threshold_ = test
            node.split_category_ = None
        node.left_ = self._new_leaf(node.depth + 1)
        node.right_ = self._new_leaf(node.depth + 1)
        if not self._keep_internal_stats:
            node.feature_stats_ = {}

    def _try_split(self, node: _Node) -> bool:
        if not node.active_ or node.n_ < self.grace_period or not self._can_grow(node):
            return False
        k = len(node.class_counts_)
        if k < 2:
            return False
        best, f, s, second = self._best_split(node)
        if f is None or s is None or best <= 0.0:
            return False
        r = merit_range(self.criterion, k)
        eps = math.sqrt(r * r * math.log(1.0 / self.delta) / (2.0 * node.n_))
        if (best - second) > eps or eps < self.tau:
            self._set_split(node, f, s)
            return True
        return False

    def _enforce_memory(self) -> None:
        """Keep the statistics of the most promising leaves, drop those of the others."""
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

    def _tick_memory(self) -> None:
        if (
            self.max_active_leaves is not None
            and self.n_seen_ % self.memory_period == 0
        ):
            self._enforce_memory()

    def learn_one(
        self, x: Any, y: Any, t: float | None = None
    ) -> "HoeffdingTreeClassifier":
        """Update the tree with one sample."""
        self._time_step(t)
        d = self._validate(x, y)
        if d is None:
            return self
        self.classes_.add(y)
        if self.root_ is None:
            self.root_ = self._new_leaf(0)
        path = self._path(d)
        leaf = path[-1]
        self._update_leaf(leaf, d, y)
        self.n_seen_ += 1
        if leaf.is_leaf and leaf.n_ % self.grace_period == 0:
            self._try_split(leaf)
        self._tick_memory()
        return self

    def _predict_proba_leaf(self, leaf: _Node, x: dict) -> dict:
        classes = list(self.classes_)
        if not classes:
            return {}
        cc = leaf.class_counts_
        if (
            self.leaf == "majority"
            or not leaf.feature_stats_
            or any(cc.get(c, 0) < 2 for c in classes)
        ):
            total = 0.0
            for v in cc.values():
                total += v
            if total <= 0.0:
                p = 1.0 / len(classes)
                return {c: p for c in classes}
            return {c: cc.get(c, 0) / total for c in classes}
        fs = leaf.feature_stats_
        logp: dict = {}
        total = len(classes)
        for v in cc.values():
            total += v
        for c in classes:
            prior = cc.get(c, 0) + 1.0
            lp = math.log(prior / total)
            for f, v in x.items():
                obs = fs.get(f)
                if obs is None:
                    continue
                if getattr(obs, "kind", "num") == "nom":
                    lp += math.log(obs.cond_proba(v, c, cc.get(c, 0)))
                    continue
                g = obs.get(c)
                if g is None or g.n_ < 2:
                    continue
                var = g.m2_ / g.n_ + 1e-9
                try:
                    dd = float(v) - g.mean_
                except (TypeError, ValueError):
                    continue
                lp -= 0.5 * math.log(6.283185307179586 * var) + 0.5 * dd * dd / var
            logp[c] = lp
        mx = max(logp.values())
        exps = {c: math.exp(v - mx) for c, v in logp.items()}
        z = sum(exps.values())
        return {c: e / z for c, e in exps.items()}

    def predict_proba_one(self, x: Any, t: float | None = None) -> dict:
        """Return the class-probability dict at the reached leaf."""
        if self.root_ is None:
            return {}
        d = _as_dict(x)
        path = self._path(d)
        return self._predict_proba_leaf(path[-1], d)

    def _predict_class_leaf(self, leaf: _Node, x: dict) -> Any:
        """Argmax of the leaf posterior, without building the dict."""
        classes = self.classes_
        if not classes:
            return None
        cc = leaf.class_counts_
        if (
            self.leaf == "majority"
            or not leaf.feature_stats_
            or any(cc.get(c, 0) < 2 for c in classes)
        ):
            best_c = None
            best_n = -1
            for c in classes:
                n_c = cc.get(c, 0)
                if n_c > best_n:
                    best_n = n_c
                    best_c = c
            return best_c
        fs = leaf.feature_stats_
        total = float(len(classes))
        for v in cc.values():
            total += v
        log_pi = math.log(6.283185307179586)
        best_c = None
        best_lp = -1e300
        for c in classes:
            prior = cc.get(c, 0) + 1.0
            lp = math.log(prior / total)
            for f, v in x.items():
                obs = fs.get(f)
                if obs is None:
                    continue
                if getattr(obs, "kind", "num") == "nom":
                    lp += math.log(obs.cond_proba(v, c, cc.get(c, 0)))
                    continue
                g = obs.get(c)
                if g is None or g.n_ < 2:
                    continue
                var = g.m2_ / g.n_ + 1e-9
                try:
                    dd = float(v) - g.mean_
                except (TypeError, ValueError):
                    continue
                lp -= 0.5 * (log_pi + math.log(var)) + 0.5 * dd * dd / var
            if lp > best_lp:
                best_lp = lp
                best_c = c
        return best_c

    def predict_one(self, x: Any, t: float | None = None) -> Any:
        """Return the most likely class."""
        if self.root_ is None:
            return None
        d = _as_dict(x)
        path = self._path(d)
        return self._predict_class_leaf(path[-1], d)

    def explain_one(self, x: Any, t: float | None = None) -> list:
        """Return the path from root to leaf as a list of dicts.

        A numeric split gives ``{"feature", "threshold", "side"}`` and a
        categorical one ``{"feature", "category", "side"}``.

        Returns an empty list when the root itself is a leaf, that is
        when the tree never split from the root. This happens on
        small datasets where the Hoeffding bound (or the t-test for
        SGT) is never satisfied, so no split is statistically
        significant. The prediction is then a constant prior over the
        classes and there is no path to explain.
        """
        if self.root_ is None:
            return []
        d = _as_dict(x)
        cur: _Node | None = self.root_
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
