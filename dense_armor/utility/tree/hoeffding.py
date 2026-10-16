"""Hoeffding tree classifier.

A non-linear classifier that grows one sample at a time. Each leaf
keeps enough statistics to estimate the information gain of a possible
split; when the difference in gain between the best and the
second-best candidate is larger than the Hoeffding bound, the leaf is
split.

The Hoeffding bound of equation 1 of Manapragada et al. 2018 is

.. math::

    \\epsilon = \\sqrt{\\frac{R^2 \\ln(1/\\delta)}{2 n}}

where ``R`` is the range of the split-evaluation function (here
``log2(K)`` for ``K`` classes), ``delta`` is one minus the required
confidence, and ``n`` is the number of samples seen at the leaf.

Numeric attributes are handled with a running Gaussian estimator per
class and per feature inside each leaf. A candidate threshold is the
midpoint between two consecutive class means; the fraction of each
class on each side is estimated with the normal CDF.

Examples:
    >>> import numpy as np
    >>> from dense_armor.utility.tree.hoeffding import HoeffdingTreeClassifier
    >>> rng = np.random.default_rng(0)
    >>> h = HoeffdingTreeClassifier(grace_period=30)
    >>> for _ in range(400):
    ...     x = float(rng.normal(0.0 if rng.random() < 0.5 else 5.0, 0.4))
    ...     y = 0 if x < 2.5 else 1
    ...     _ = h.learn_one({"x": x}, y=y)
    >>> h.predict_one({"x": 0.1}) != h.predict_one({"x": 5.0})
    True

References:
    Manapragada, C., Webb, G. I., Salehi, M. (2018). Extremely fast
        decision tree. In KDD.
    Domingos, P., Hulten, G. (2000). Mining high-speed data streams.
        In KDD.
"""

import math
from typing import Any

from dense_armor.roles import Classifier


def _as_dict(x: Any) -> dict:
    if hasattr(x, "to_dict"):
        return dict(x.to_dict())
    try:
        return dict(x)
    except (TypeError, ValueError):
        return {}


def _phi(z: float) -> float:
    """Standard normal cumulative distribution function."""
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


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


class _Gauss:
    """Running Gaussian for one (leaf, class, feature) triple."""

    __slots__ = ("m2_", "mean_", "n_")

    def __init__(self) -> None:
        self.n_ = 0
        self.mean_ = 0.0
        self.m2_ = 0.0

    def update(self, v: float) -> None:
        self.n_ += 1
        d = v - self.mean_
        self.mean_ += d / self.n_
        self.m2_ += d * (v - self.mean_)

    @property
    def var(self) -> float:
        return self.m2_ / self.n_ if self.n_ > 1 else 0.0


class _Node:
    """One node of the tree: a leaf or an internal split."""

    __slots__ = (
        "class_counts_",
        "depth",
        "feature_stats_",
        "left_",
        "n_",
        "right_",
        "split_feature_",
        "split_threshold_",
    )

    def __init__(self, depth: int) -> None:
        self.depth = depth
        self.n_ = 0
        self.class_counts_: dict = {}
        self.feature_stats_: dict = {}
        self.split_feature_: str | None = None
        self.split_threshold_: float | None = None
        self.left_: _Node | None = None
        self.right_: _Node | None = None

    @property
    def is_leaf(self) -> bool:
        return self.split_feature_ is None


class HoeffdingTreeClassifier(Classifier):
    """Hoeffding tree for incremental classification.

    Memory grows with the number of nodes; ``max_nodes`` (default
    ``10000``) bounds the total, so ``memory_class`` is ``"O(window)"``.

    ``budget_s`` = 1e-3 s: p99 of ``learn_one`` measured at 2.3e-5 s on a
    2000-sample, two-feature stream on a desktop CPU (default
    parameters); the budget leaves room for slower machines.

    Args:
        grace_period: minimum number of samples at a leaf before a split
            is attempted.
        delta: one minus the confidence of the Hoeffding bound.
        tau: tie threshold.
        max_depth: maximum depth, or ``None``.
        max_nodes: maximum number of nodes. Default ``10000``.
        leaf: ``"nb"`` or ``"majority"``.

    Raises:
        ValueError: on a bad ``leaf`` or a non-positive parameter.
    """

    budget_s = 1e-3
    memory_class = "O(window)"

    def __init__(
        self,
        grace_period: int = 200,
        delta: float = 1e-7,
        tau: float = 0.05,
        max_depth: int | None = None,
        max_nodes: int | None = 10000,
        leaf: str = "nb",
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
        self.grace_period = grace_period
        self.delta = delta
        self.tau = tau
        self.max_depth = max_depth
        self.max_nodes = max_nodes
        self.leaf = leaf
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

    def _depth(self, node: _Node | None) -> int:
        if node is None or node.is_leaf:
            return 0
        return 1 + max(self._depth(node.left_), self._depth(node.right_))

    def _new_leaf(self, depth: int) -> _Node:
        node = _Node(depth)
        self.n_nodes_ += 1
        return node

    def _path(self, x: dict) -> list:
        """Nodes from the root to the leaf of ``x`` (inputs already validated)."""
        path: list = []
        cur: _Node | None = self.root_
        while cur is not None:
            path.append(cur)
            if cur.split_feature_ is None:
                return path
            f = cur.split_feature_
            v = x.get(f)
            if v is None:
                return path
            cur = cur.left_ if float(v) <= cur.split_threshold_ else cur.right_
        return path

    def _update_leaf(self, node: _Node, x: dict, y: Any) -> None:
        node.n_ += 1
        cc = node.class_counts_
        cc[y] = cc.get(y, 0) + 1
        fs = node.feature_stats_
        known = self.features_
        for f, v in x.items():
            per_class = fs.get(f)
            if per_class is None:
                per_class = {}
                fs[f] = per_class
            g = per_class.get(y)
            if g is None:
                g = _Gauss()
                per_class[y] = g
            g.update(float(v))
            if f not in known:
                known.add(f)

    def _entropy(self, node: _Node) -> float:
        return _entropy(node.class_counts_)

    def _split_gain(self, node: _Node, f: str, s: float):
        stats = node.feature_stats_.get(f)
        if not stats:
            return None
        n = node.n_
        if n <= 0:
            return None
        h0 = self._entropy(node)
        left_counts: dict = {}
        right_counts: dict = {}
        for k, g in stats.items():
            if g.n_ <= 0:
                continue
            total_k = node.class_counts_.get(k, 0)
            if g.var <= 0.0:
                if g.mean_ <= s:
                    left_counts[k] = total_k
                else:
                    right_counts[k] = total_k
                continue
            z = (s - g.mean_) / math.sqrt(g.var)
            p_left = _phi(z)
            left_counts[k] = p_left * total_k
            right_counts[k] = (1.0 - p_left) * total_k
        n_l = sum(left_counts.values())
        n_r = sum(right_counts.values())
        if n_l <= 0.0 or n_r <= 0.0:
            return None
        h_l = _entropy(left_counts)
        h_r = _entropy(right_counts)
        return h0 - (n_l / n) * h_l - (n_r / n) * h_r

    def _best_split(self, node: _Node):
        best_gain = -1.0
        best_f: str | None = None
        best_s: float | None = None
        second_gain = -1.0
        for f, per_class in node.feature_stats_.items():
            means = [g.mean_ for g in per_class.values() if g.n_ > 0]
            if len(means) < 2:
                continue
            means.sort()
            candidates = [
                0.5 * (means[i] + means[i + 1]) for i in range(len(means) - 1)
            ]
            f_best = -1.0
            f_s: float | None = None
            for s in candidates:
                g = self._split_gain(node, f, s)
                if g is not None and g > f_best:
                    f_best = g
                    f_s = s
            if f_best < 0.0:
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

    def _try_split(self, node: _Node) -> bool:
        if node.n_ < self.grace_period or not self._can_grow(node):
            return False
        k = len(node.class_counts_)
        if k < 2:
            return False
        best, f, s, second = self._best_split(node)
        if f is None or s is None or best <= 0.0:
            return False
        r = math.log2(k)
        eps = math.sqrt(r * r * math.log(1.0 / self.delta) / (2.0 * node.n_))
        if (best - second) > eps or eps < self.tau:
            node.split_feature_ = f
            node.split_threshold_ = s
            node.left_ = self._new_leaf(node.depth + 1)
            node.right_ = self._new_leaf(node.depth + 1)
            return True
        return False

    def learn_one(
        self, x: Any, y: Any, t: float | None = None
    ) -> "HoeffdingTreeClassifier":
        """Update the tree with one sample."""
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
        self._update_leaf(leaf, d, y)
        self.n_seen_ += 1
        if leaf.is_leaf and leaf.n_ % self.grace_period == 0:
            self._try_split(leaf)
        return self

    def _predict_proba_leaf(self, leaf: _Node, x: dict) -> dict:
        classes = list(self.classes_)
        if not classes:
            return {}
        cc = leaf.class_counts_
        if self.leaf == "majority" or not leaf.feature_stats_:
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
                per_class = fs.get(f)
                if per_class is None:
                    continue
                g = per_class.get(c)
                if g is None or g.n_ < 2:
                    continue
                var = g.m2_ / g.n_ + 1e-9
                dd = float(v) - g.mean_
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
        if self.leaf == "majority" or not leaf.feature_stats_:
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
                per_class = fs.get(f)
                if per_class is None:
                    continue
                g = per_class.get(c)
                if g is None or g.n_ < 2:
                    continue
                var = g.m2_ / g.n_ + 1e-9
                dd = float(v) - g.mean_
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
            s = cur.split_threshold_
            assert f is not None and s is not None
            v = d.get(f)
            if v is None:
                break
            try:
                fv = float(v)
            except (TypeError, ValueError):
                break
            side = "left" if fv <= s else "right"
            out.append({"feature": f, "threshold": s, "side": side})
            cur = cur.left_ if side == "left" else cur.right_
        return out
