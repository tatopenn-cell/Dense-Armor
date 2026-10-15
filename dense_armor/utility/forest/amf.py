"""Aggregated Mondrian forests (AMF) for classification and regression.

A Mondrian tree cuts the input box into cells; a forecaster in every
cell predicts from the samples that fell in it. A deep tree fits well
and a shallow one is stable, and which one is right is not known in
advance. AMF does not choose: it answers with the average of the
predictions of every pruning of the tree, each pruning weighted by how
well it has predicted so far, following Mourtada, Gaiffas and Scornet
(2021).

For a tree with cell forecasters ``y_v``, the prediction at ``x`` is
the exponentially weighted average over all finite subtrees ``T``

    f_t(x) = sum_T pi(T) exp(-step L_{t-1}(T)) y_T(x)
             / sum_T pi(T) exp(-step L_{t-1}(T)),         (6, page 5)

with the prior ``pi(T) = 2^{-|T|}`` for ``|T|`` nodes (7, page 5) and
``L_{t-1}(T)`` the cumulative loss of the pruning ``T``. There are
exponentially many prunings; the prior makes the sum collapse to one
weight pair per node, updated along one root-to-leaf path
(Algorithm 2, page 9) and read along one path (Algorithm 3, page 11):

    w_v    = exp(-step L_v)
    wbar_v = w_v                            at a leaf
    wbar_v = (w_v + wbar_v0 wbar_v1) / 2    at an inner node
    y~_v   = (w_v / (2 wbar_v)) y_v + (wbar_v0 wbar_v1 / (2 wbar_v)) y~_va

where ``va`` is the child that contains ``x`` and ``y~`` at the leaf is
the forecaster of the leaf. The weights are kept as logarithms.

Relation to the Mondrian forest of the library. The Mondrian forest
smooths the prediction of a tree with a fixed hierarchical prior over
where a path leaves the tree, so a node weighs the same whatever the
tree has predicted. AMF replaces that fixed prior with weights learned
from the loss of every node, so a cell that has predicted badly is
discounted, at a cost of two weights per node and no lifetime
parameter.

What is implemented, and what is not. This is the version of Section 2
of the paper, on an unrestricted Mondrian partition of a box (Algorithm
2, page 9): the first sample in a cell is kept, and a second sample
splits the cell until the two are separated. The version of Section 4
(restricted partition, splits above existing nodes, Algorithms 4 to 6,
pages 17-19), which the paper uses in its experiments and for which it
states no guarantee, is not implemented. The regret bounds below are
those of the implemented version.

Regret, as the paper states it (page 13). With ``step = 1`` and the
logarithmic loss, for every finite subtree ``T`` of the partition and
every sequence of samples (Corollary 1, equation 10),

    sum_t loss(f_t(x_t), y_t) - sum_t loss(g_T(x_t), y_t)
        <= |T| log 2 + (|T| + 1) (K - 1) / 4 * log(4 n)

for any ``g_T`` constant on the leaves of ``T``, with ``K`` classes and
``n`` samples. With the quadratic loss, targets in ``[-B, B]`` and
``step = 1 / (8 B^2)`` (Corollary 2, equation 11),

    sum_t loss(f_t(x_t), y_t) - sum_t loss(g_T(x_t), y_t)
        <= 4 B^2 (|T| + 1) log n.

Both come from the exponential-weights bound of Lemma 1 (equation 9),
``(1 / step) |T| log 2`` against the forecasters of the nodes. Nothing
is claimed here beyond this: the bounds are for one tree, on a fixed
partition, with the prediction made after the partition has been
updated with ``x_t``; the forest prediction is the mean of the tree
predictions (equation 1) and, for a convex loss, inherits them.

Choices the paper does not fix, stated as my choice:

- ``step`` and ``dirichlet`` are the names of this class. ``step`` is
  the learning rate ``eta`` of the paper (page 6). ``dirichlet`` is the
  parameter ``a`` of the symmetric Dirichlet prior of the forecaster of
  a node, ``(n_y + a) / (n + K a)``; ``a = 1/2`` is the
  Krichevsky-Trofimov forecaster of equation 5 (page 5), the Jeffreys
  prior of its Remark 1. The paper does not name either parameter in
  code and calls neither ``dirichlet``.
- Equation 5 prints ``t - 1`` in the denominator; the forecaster that
  sums to one over the labels of a node uses the count ``n`` of the
  samples of that node, which is what is implemented.
- The label set grows with the stream: ``K`` is the number of labels
  seen so far, the current one included, unless ``classes`` fixes it
  as in the paper.
- The input box is ``[low, high]`` on every feature; a value outside is
  clipped to the box, since the paper takes inputs in ``[0, 1]^d``.
- Prediction draws the splits that would separate the query from the
  sample of its cell and discards them (page 10), so ``predict_one``
  is random; it never changes the trees.

Examples:
    >>> from dense_armor.utility.forest.amf import (
    ...     AggregatedMondrianForestClassifier,
    ... )
    >>> f = AggregatedMondrianForestClassifier(n_trees=5, seed=0)
    >>> for i in range(200):
    ...     x = (i % 20) / 19.0
    ...     _ = f.learn_one({"x": x}, 0 if x < 0.5 else 1)
    >>> f.predict_one({"x": 0.1}), f.predict_one({"x": 0.9})
    (0, 1)

References:
    Mourtada, J., Gaiffas, S., Scornet, E. (2021). AMF: Aggregated
        Mondrian forests for online learning. Journal of the Royal
        Statistical Society B 83(3), 505-533 (arXiv:1906.10529v2,
        a manuscript version). Equations 1, 5, 6, 7 and 9 to 11,
        Algorithms 2 and 3, Lemma 1, Corollaries 1 and 2.
    Lakshminarayanan, B., Roy, D. M., Teh, Y. W. (2014). Mondrian
        forests: efficient online random forests. In NIPS.
"""

import math
from typing import Any

import numpy as np

from dense_armor.roles import Classifier, Regressor

_LOG2 = math.log(2.0)


def _as_dict(x: Any) -> dict:
    if hasattr(x, "to_dict"):
        return dict(x.to_dict())
    try:
        return dict(x)
    except (TypeError, ValueError):
        return {}


class _Node:
    """One node of a Mondrian tree with its two weights and its forecaster."""

    __slots__ = ("j", "left", "lw", "lwb", "point", "right", "s", "stat")

    def __init__(self, stat: Any) -> None:
        self.j = 0
        self.s = 0.0
        self.left: _Node | None = None
        self.right: _Node | None = None
        self.point: np.ndarray | None = None
        self.lw = 0.0
        self.lwb = 0.0
        self.stat = stat


class _Frequency:
    """Forecaster of a node for classification: Dirichlet-smoothed frequencies."""

    def __init__(self, dirichlet: float, alphabet: list) -> None:
        self.a = dirichlet
        self.alphabet = alphabet

    def empty(self) -> list:
        return [0, {}]

    def copy(self, st: list) -> list:
        return [st[0], dict(st[1])]

    def predict(self, st: list) -> dict:
        den = st[0] + len(self.alphabet) * self.a
        return {c: (st[1].get(c, 0) + self.a) / den for c in self.alphabet}

    def loss(self, st: list, y: Any) -> float:
        den = st[0] + len(self.alphabet) * self.a
        return -math.log((st[1].get(y, 0) + self.a) / den)

    def update(self, st: list, y: Any) -> None:
        st[0] += 1
        st[1][y] = st[1].get(y, 0) + 1

    def mix(self, wa: float, a: dict, wb: float, b: dict) -> dict:
        return {c: wa * a[c] + wb * b[c] for c in self.alphabet}


class _Mean:
    """Forecaster of a node for regression: the empirical mean, 0 when empty."""

    def empty(self) -> list:
        return [0, 0.0]

    def copy(self, st: list) -> list:
        return [st[0], st[1]]

    def predict(self, st: list) -> float:
        return st[1] / st[0] if st[0] > 0 else 0.0

    def loss(self, st: list, y: float) -> float:
        return (self.predict(st) - y) ** 2

    def update(self, st: list, y: float) -> None:
        st[0] += 1
        st[1] += y

    def mix(self, wa: float, a: float, wb: float, b: float) -> float:
        return wa * a + wb * b


class _AmfTree:
    """One tree: unrestricted Mondrian partition and collapsed exponential weights."""

    def __init__(
        self,
        d: int,
        low: float,
        high: float,
        step: float,
        forecaster: Any,
        rng: np.random.Generator,
    ) -> None:
        self.d = d
        self.low = np.full(d, low, dtype=float)
        self.high = np.full(d, high, dtype=float)
        self.step = step
        self.fc = forecaster
        self.rng = rng
        self.root = _Node(forecaster.empty())
        self.n_nodes = 1

    def _descend(self, x: np.ndarray) -> tuple:
        lo = self.low.copy()
        hi = self.high.copy()
        path: list = []
        node = self.root
        while node.left is not None:
            path.append(node)
            if x[node.j] <= node.s:
                hi[node.j] = node.s
                node = node.left
            else:
                lo[node.j] = node.s
                node = node.right
        return path, node, lo, hi

    def _cut(self, lo: np.ndarray, hi: np.ndarray) -> tuple:
        width = hi - lo
        j = int(self.rng.choice(self.d, p=width / width.sum()))
        return j, float(self.rng.uniform(lo[j], hi[j]))

    def add_point(self, x: np.ndarray) -> None:
        """Update the partition with ``x`` (lines 2 to 10 of Algorithm 2)."""
        _, node, lo, hi = self._descend(x)
        while node.point is not None and not np.array_equal(node.point, x):
            xp = node.point
            j, s = self._cut(lo, hi)
            node.j, node.s = j, s
            node.left = _Node(self.fc.empty())
            node.right = _Node(self.fc.empty())
            self.n_nodes += 2
            ax = 0 if x[j] <= s else 1
            ap = 0 if xp[j] <= s else 1
            kept = node.left if ap == 0 else node.right
            kept.stat = self.fc.copy(node.stat)
            kept.lw = node.lw
            kept.lwb = node.lwb
            kept.point = xp
            node.point = None
            if ax == 0:
                hi[j] = s
                node = node.left
            else:
                lo[j] = s
                node = node.right

    def update_weights(self, x: np.ndarray, y: Any) -> None:
        """Update the weights and forecasters along the path of ``x``
        (lines 11 to 18 of Algorithm 2)."""
        path, leaf, _, _ = self._descend(x)
        leaf.point = x.copy()
        for node in [leaf, *reversed(path)]:
            node.lw -= self.step * self.fc.loss(node.stat, y)
            if node.left is None:
                node.lwb = node.lw
            else:
                node.lwb = (
                    float(np.logaddexp(node.lw, node.left.lwb + node.right.lwb)) - _LOG2
                )
            self.fc.update(node.stat, y)

    def leaf_prediction(self, x: np.ndarray) -> Any:
        """Forecaster of the leaf that contains ``x``: the unaggregated prediction."""
        _, leaf, _, _ = self._descend(x)
        return self.fc.predict(leaf.stat)

    def predict(self, x: np.ndarray, temporary: bool = True) -> Any:
        """Aggregated prediction at ``x`` (Algorithm 3).

        With ``temporary`` the splits that would separate ``x`` from the
        sample of its cell are drawn and discarded, as the paper does.
        """
        path, leaf, lo, hi = self._descend(x)
        out = self.fc.predict(leaf.stat)
        if temporary and leaf.point is not None and not np.array_equal(leaf.point, x):
            m = 0
            xp = leaf.point
            separated = False
            while not separated:
                j, s = self._cut(lo, hi)
                m += 1
                separated = (x[j] <= s) != (xp[j] <= s)
                if not separated:
                    if x[j] <= s:
                        hi[j] = s
                    else:
                        lo[j] = s
            q = 0.5**m
            out = self.fc.mix(1.0 - q, out, q, self.fc.predict(self.fc.empty()))
        child = leaf
        for node in reversed(path):
            other = node.right if child is node.left else node.left
            c1 = 0.5 * math.exp(node.lw - node.lwb)
            c2 = 0.5 * math.exp(other.lwb + child.lwb - node.lwb)
            out = self.fc.mix(c1, self.fc.predict(node.stat), c2, out)
            child = node
        return out


class _Forest:
    """Shared machinery of the two forests: keys, box, trees, counters."""

    def _setup(self) -> None:
        self.trees_: list = []
        self.n_seen_ = 0
        self.n_missing_ = 0
        self._keys: list | None = None
        self._rng = np.random.default_rng(self.seed)

    @property
    def n_missing(self) -> int:
        return self.n_missing_

    def _unit_test_skips(self) -> set:
        return {"check_step_is_pure", "check_step_is_jittable"}

    def _vector(self, d: dict) -> np.ndarray | None:
        if self._keys is None:
            keys = (
                list(self.feature_keys) if self.feature_keys is not None else sorted(d)
            )
            self._keys = keys
        try:
            v = np.array([float(d[k]) for k in self._keys], dtype=float)
        except (KeyError, TypeError, ValueError):
            return None
        if not np.isfinite(v).all():
            return None
        return np.clip(v, self.low, self.high)

    def _make_trees(self, d: int, forecaster: Any) -> None:
        self.trees_ = [
            _AmfTree(
                d,
                self.low,
                self.high,
                self.step,
                forecaster,
                np.random.default_rng(int(self._rng.integers(2**31 - 1))),
            )
            for _ in range(self.n_trees)
        ]

    def _check(self, n_trees: int, step: float, low: float, high: float) -> None:
        if n_trees < 1:
            raise ValueError(f"n_trees must be >= 1, got {n_trees}")
        if step <= 0.0:
            raise ValueError(f"step must be > 0, got {step}")
        if not low < high:
            raise ValueError(f"low must be < high, got low={low}, high={high}")


class AggregatedMondrianForestClassifier(_Forest, Classifier):
    """Aggregated Mondrian forest classifier (Mourtada et al.).

    Each of the ``n_trees`` trees predicts a distribution over the
    labels as the exponentially weighted average of all its prunings,
    with the Dirichlet forecaster ``(n_y + dirichlet) / (n + K *
    dirichlet)`` in every cell; the forest averages the trees.

    Cost and memory. ``learn_one`` and ``predict_one`` walk one path of
    every tree down and up, a depth that grows like the logarithm of
    the number of samples (the paper states it for the restricted
    partition, page 11). A tree keeps two weights and a forecaster per
    node, and measured on uniform inputs in 1, 2 and 4 features it has
    3.6 to 4.0 nodes per sample, so memory grows with the stream:
    ``memory_class`` is
    ``"O(n)"`` in the number of samples, linear in ``n_trees`` and in
    the number of features. There is no pure ``init_state`` / ``step``
    form: the state is a tree that changes shape.

    Args:
        n_trees: number of trees ``M`` of the forest. Default 10.
        step: learning rate ``eta`` of the exponential weights. Default
            1.0, the value of Corollary 1 for the logarithmic loss.
        dirichlet: symmetric Dirichlet parameter of the forecaster of a
            node. Default 0.5, the Krichevsky-Trofimov forecaster.
        low: lower end of every feature. Default 0.0.
        high: upper end of every feature. Default 1.0.
        classes: the label set, when known. ``None`` grows it with the
            stream.
        seed: seed of the generator of the partitions.
        feature_keys: names of the features. ``None`` takes the sorted
            keys of the first sample.

    Raises:
        ValueError: on a non-positive ``n_trees``, ``step`` or
            ``dirichlet``, or ``low >= high``.

    Examples:
        >>> from dense_armor.utility.forest.amf import (
        ...     AggregatedMondrianForestClassifier,
        ... )
        >>> f = AggregatedMondrianForestClassifier(n_trees=3, seed=1)
        >>> f.predict_one({"x": 0.5}) is None
        True
        >>> for i in range(40):
        ...     _ = f.learn_one({"x": (i % 2) * 0.8 + 0.1}, i % 2)
        >>> f.predict_one({"x": 0.9})
        1

    References:
        Mourtada, Gaiffas, Scornet (2021), Algorithms 2 and 3, Corollary 1.
    """

    budget_s = 0.02
    memory_class = "O(n)"

    def __init__(
        self,
        n_trees: int = 10,
        step: float = 1.0,
        dirichlet: float = 0.5,
        low: float = 0.0,
        high: float = 1.0,
        classes: list | None = None,
        seed: int | None = None,
        feature_keys: list | None = None,
    ) -> None:
        self._check(n_trees, step, low, high)
        if dirichlet <= 0.0:
            raise ValueError(f"dirichlet must be > 0, got {dirichlet}")
        self.n_trees = n_trees
        self.step = step
        self.dirichlet = dirichlet
        self.low = low
        self.high = high
        self.classes = classes
        self.seed = seed
        self.feature_keys = feature_keys
        self.alphabet_: list = []
        self._setup()

    def learn_one(
        self, x: Any, y: Any, t: float | None = None
    ) -> "AggregatedMondrianForestClassifier":
        """Add one labelled sample to every tree.

        A sample without a label, with a missing or non-finite value,
        or a label outside ``classes`` is skipped and counted in
        ``n_missing``.
        """
        self._time_step(t)
        d = _as_dict(x)
        v = self._vector(d) if d and y is not None else None
        if v is None or (self.classes is not None and y not in self.classes):
            self.n_missing_ += 1
            return self
        if not self.trees_:
            self.alphabet_.extend(self.classes if self.classes is not None else [])
            self._make_trees(len(v), _Frequency(self.dirichlet, self.alphabet_))
        if y not in self.alphabet_:
            self.alphabet_.append(y)
        for tree in self.trees_:
            tree.add_point(v)
            tree.update_weights(v, y)
        self.n_seen_ += 1
        return self

    def _proba(self, x: Any, aggregated: bool) -> dict:
        v = self._vector(_as_dict(x)) if self.trees_ else None
        if v is None or not self.alphabet_:
            return {}
        outs = [
            t.predict(v) if aggregated else t.leaf_prediction(v) for t in self.trees_
        ]
        return {c: sum(o[c] for o in outs) / len(outs) for c in self.alphabet_}

    def predict_proba_one(self, x: Any, t: float | None = None) -> dict:
        """Mean over the trees of the aggregated label distributions."""
        return self._proba(x, True)

    def predict_one(self, x: Any, t: float | None = None) -> Any:
        """Return the most probable label, or ``None`` before learning."""
        p = self.predict_proba_one(x, t=t)
        return max(p.items(), key=lambda kv: kv[1])[0] if p else None

    def predict_unaggregated_one(self, x: Any) -> Any:
        """Most probable label by the leaf forecasters alone, without aggregation."""
        p = self._proba(x, False)
        return max(p.items(), key=lambda kv: kv[1])[0] if p else None


class AggregatedMondrianForestRegressor(_Forest, Regressor):
    """Aggregated Mondrian forest regressor (Mourtada et al.).

    Each tree predicts the exponentially weighted average, over all its
    prunings, of the empirical mean of the targets of the cell (0 for an
    empty cell, equation 4 of the paper, page 5), under the quadratic
    loss; the forest averages the trees. Cost and memory are those of
    :class:`AggregatedMondrianForestClassifier`.

    Args:
        n_trees: number of trees of the forest. Default 10.
        step: learning rate ``eta``. Default 0.125, which is
            ``1 / (8 B^2)`` of Corollary 2 for targets in ``[-1, 1]``;
            for targets in ``[-B, B]`` pass ``1 / (8 B^2)``.
        low: lower end of every feature. Default 0.0.
        high: upper end of every feature. Default 1.0.
        seed: seed of the generator of the partitions.
        feature_keys: names of the features. ``None`` takes the sorted
            keys of the first sample.

    Raises:
        ValueError: on a non-positive ``n_trees`` or ``step``, or
            ``low >= high``.

    Examples:
        >>> from dense_armor.utility.forest.amf import (
        ...     AggregatedMondrianForestRegressor,
        ... )
        >>> f = AggregatedMondrianForestRegressor(n_trees=5, seed=0)
        >>> f.predict_one({"x": 0.5})
        0.0
        >>> for i in range(300):
        ...     x = (i % 30) / 29.0
        ...     _ = f.learn_one({"x": x}, 1.0 if x > 0.5 else -1.0)
        >>> f.predict_one({"x": 0.9}) > 0.5 > -0.5 > f.predict_one({"x": 0.1})
        True

    References:
        Mourtada, Gaiffas, Scornet (2021), Algorithms 2 and 3, Corollary 2.
    """

    budget_s = 0.02
    memory_class = "O(n)"

    def __init__(
        self,
        n_trees: int = 10,
        step: float = 0.125,
        low: float = 0.0,
        high: float = 1.0,
        seed: int | None = None,
        feature_keys: list | None = None,
    ) -> None:
        self._check(n_trees, step, low, high)
        self.n_trees = n_trees
        self.step = step
        self.low = low
        self.high = high
        self.seed = seed
        self.feature_keys = feature_keys
        self._setup()

    def learn_one(
        self, x: Any, y: Any, t: float | None = None
    ) -> "AggregatedMondrianForestRegressor":
        """Add one sample to every tree.

        A sample with a missing or non-finite value or target is skipped
        and counted in ``n_missing``.
        """
        self._time_step(t)
        d = _as_dict(x)
        try:
            target = float(y)
        except (TypeError, ValueError):
            target = math.nan
        v = self._vector(d) if d and math.isfinite(target) else None
        if v is None:
            self.n_missing_ += 1
            return self
        if not self.trees_:
            self._make_trees(len(v), _Mean())
        for tree in self.trees_:
            tree.add_point(v)
            tree.update_weights(v, target)
        self.n_seen_ += 1
        return self

    def _value(self, x: Any, aggregated: bool) -> float | None:
        v = self._vector(_as_dict(x)) if self.trees_ else None
        if v is None:
            return None
        outs = [
            t.predict(v) if aggregated else t.leaf_prediction(v) for t in self.trees_
        ]
        return float(sum(outs) / len(outs))

    def predict_one(
        self, x: Any, t: float | None = None, return_std: bool = False
    ) -> Any:
        """Mean over the trees of the aggregated predictions, 0.0 before learning."""
        out = self._value(x, True)
        return 0.0 if out is None else out

    def predict_unaggregated_one(self, x: Any) -> float:
        """Mean of the leaf forecasters alone, without aggregation."""
        out = self._value(x, False)
        return 0.0 if out is None else out
