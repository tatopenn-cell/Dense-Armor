"""Adaptive Random Forest classifier for evolving data streams.

A forest of Hoeffding trees, each trained on a Poisson-weighted copy of
the stream and each watched by its own pair of change detectors, after
Gomes et al. (2017). A single Hoeffding tree follows a change through
one detector; the forest follows it through one detector per tree, and
the spread of the votes says how sure it is.

The pieces, with the place where the paper states them (page numbers
are those of the PDF):

- Online bagging. Every tree sees every sample ``k`` times, with
  ``k ~ Poisson(lambda_)`` (Algorithm 1, page 7). The paper uses
  ``lambda_ = 6``, as in leveraging bagging, instead of 1.
- Random subspace. When a node is created, a random subset of ``m``
  features is drawn and only they are tried at that node's split
  attempts (Algorithm 1 and the text below it, page 7). The paper uses
  ``m = sqrt(M) + 1`` for ``M`` features (page 15).
- One warning detector and one drift detector per tree, fed with the
  test-then-train error of the tree (Algorithm 2, page 8). A warning
  starts a background tree that is trained but does not vote; a drift
  replaces the tree by its background tree.
- Background trees carry no detectors, so they cannot start further
  background trees (page 9).
- The vote of a tree is weighted by ``c / n``: ``n`` samples seen since
  its last reset and ``c`` of them classified correctly (page 8).

Choices the paper does not fix, stated as my choice:

- The detector thresholds. The paper compares a slow setting
  (``delta_w = 1e-4``, ``delta_d = 1e-5``) with a fast one
  (``delta_w = 0.01``, ``delta_d = 0.001``) on page 17. Without detectors
  given, this class takes the fast setting, because it reacts inside
  streams of a few thousand samples.
- The subset of features of a node is drawn at the first split attempt
  of the node, from the features seen there, and kept for the later
  attempts; the paper draws it at creation, when a streaming tree does
  not yet know its features.
- A drift without a warning before it replaces the tree by a new empty
  tree, since there is no background tree to promote.
- The vote adds the class probabilities of each tree, times its weight,
  and normalises; the paper weights the class votes. When every weight
  is zero, as before any tree has been scored, all trees count equally.
- A sample for which a tree has no answer yet counts as an error for
  that tree.

Examples:
    >>> import numpy as np
    >>> from dense_armor.utility.forest.arf import AdaptiveRandomForestClassifier
    >>> rng = np.random.default_rng(0)
    >>> arf = AdaptiveRandomForestClassifier(n_models=5, seed=1)
    >>> for _ in range(600):
    ...     x = float(rng.normal(0.0 if rng.random() < 0.5 else 5.0, 0.4))
    ...     _ = arf.learn_one({"x": x}, 0 if x < 2.5 else 1)
    >>> arf.predict_one({"x": 0.1}), arf.predict_one({"x": 5.0})
    (0, 1)

References:
    Gomes, H. M., Bifet, A., Read, J., Barddal, J. P., Enembreck, F.,
        Pfahringer, B., Holmes, G., Abdessalem, T. (2017). Adaptive
        random forests for evolving data stream classification. Machine
        Learning 106, 1469-1495. Algorithm 1 (page 7), Algorithm 2
        (page 8), Section 4 (pages 8-9).
"""

import math
from typing import Any

import numpy as np

from dense_armor.roles import Classifier, DriftDetector
from dense_armor.utility.drift.adwin import ADWIN
from dense_armor.utility.tree.hoeffding import (
    HoeffdingTreeClassifier,
    _as_dict,
    _Node,
)


class _SubspaceNode(_Node):
    """Tree node that remembers the feature subset of its split attempts."""

    __slots__ = ("subset_",)

    def __init__(self, depth: int) -> None:
        super().__init__(depth)
        self.subset_: tuple | None = None


class SubspaceHoeffdingTree(HoeffdingTreeClassifier):
    """Hoeffding tree whose split attempts see a random subset of features.

    Args:
        grace_period: samples at a leaf before a split is attempted.
        delta: one minus the confidence of the Hoeffding bound.
        tau: tie threshold.
        max_depth: maximum depth, or ``None``.
        max_nodes: maximum number of nodes, or ``None``.
        leaf: ``"nb"`` or ``"majority"``.
        max_features: size ``m`` of the subset. ``None`` takes
            ``int(sqrt(M)) + 1`` for ``M`` features.
        seed: seed of the generator that draws the subsets.

    Raises:
        ValueError: on a non-positive ``max_features`` or on any bad
            argument of the Hoeffding tree.
    """

    def __init__(
        self,
        grace_period: int = 200,
        delta: float = 1e-7,
        tau: float = 0.05,
        max_depth: int | None = None,
        max_nodes: int | None = 10000,
        leaf: str = "nb",
        max_features: int | None = None,
        seed: int | None = None,
    ) -> None:
        super().__init__(
            grace_period=grace_period,
            delta=delta,
            tau=tau,
            max_depth=max_depth,
            max_nodes=max_nodes,
            leaf=leaf,
        )
        if max_features is not None and max_features < 1:
            raise ValueError(f"max_features must be >= 1, got {max_features}")
        self.max_features = max_features
        self.seed = seed
        self._rng = np.random.default_rng(seed)

    def _new_leaf(self, depth: int) -> _Node:
        node = _SubspaceNode(depth)
        self.n_nodes_ += 1
        return node

    def _subset_size(self, n_features: int) -> int:
        if self.max_features is None:
            return min(n_features, int(math.sqrt(n_features)) + 1)
        return min(n_features, self.max_features)

    def _best_split(self, node: Any) -> tuple:
        full = node.feature_stats_
        if node.subset_ is None:
            names = sorted(full)
            pick = self._rng.choice(
                len(names), size=self._subset_size(len(names)), replace=False
            )
            node.subset_ = tuple(sorted(names[i] for i in pick))
        node.feature_stats_ = {f: full[f] for f in node.subset_ if f in full}
        try:
            return super()._best_split(node)
        finally:
            node.feature_stats_ = full


class _Member:
    """One tree of the forest with its detectors, background tree and score."""

    def __init__(self, tree: Any, drift: Any, warn: Any) -> None:
        self.tree = tree
        self.drift = drift
        self.warn = warn
        self.background: Any = None
        self.correct = 0
        self.seen = 0

    @property
    def weight(self) -> float:
        return self.correct / self.seen if self.seen > 0 else 0.0


def _finite_values(d: dict) -> bool:
    for v in d.values():
        if v is None:
            return False
        try:
            fv = float(v)
        except (TypeError, ValueError):
            return False
        if not math.isfinite(fv):
            return False
    return True


class AdaptiveRandomForestClassifier(Classifier):
    """Adaptive Random Forest of Gomes et al. (2017).

    Each of the ``n_models`` trees is a :class:`SubspaceHoeffdingTree`.
    On every sample each tree is scored on the sample before learning
    it, trained ``Poisson(lambda_)`` times on it, and its error (0 or
    1) goes to its warning detector and its drift detector. The forest
    answers with the weighted vote of the trees.

    A warning creates a background tree for that member; a drift
    replaces the member's tree by the background tree, resets the
    member's score and its detectors.

    Cost and memory. ``learn_one`` does ``n_models`` predictions, about
    ``n_models * lambda_`` tree updates and ``2 * n_models`` detector
    updates; ``predict_one`` does ``n_models`` predictions. Each tree is
    bounded by its ``max_nodes``, so the state is constant in the
    stream length and linear in ``n_models`` and in the number of
    features. There is no pure ``init_state`` / ``step`` form: the state
    is a set of trees of changing shape, which a ``jax.lax.scan`` cannot
    carry.

    Args:
        n_models: number of trees. Default 10.
        max_features: features tried per split attempt. ``None`` takes
            ``int(sqrt(M)) + 1`` (the paper, page 15).
        lambda_: mean of the Poisson weight. Default 6.0 (the paper).
        drift_detector: template of the drift detector, cloned for every
            tree. ``None`` takes ``ADWIN(delta=0.001)``.
        warning_detector: template of the warning detector, cloned for
            every tree. ``None`` takes ``ADWIN(delta=0.01)``.
        seed: seed of the generator of the Poisson weights and of the
            feature subsets.
        grace_period: samples at a leaf before a split is attempted
            (my choice: 50, shorter than the tree default of 200, so
            that trees grow within streams of a few thousand samples).

    Raises:
        ValueError: on a non-positive ``n_models``, ``lambda_``,
            ``max_features`` or ``grace_period``.

    Examples:
        >>> from dense_armor.utility.forest.arf import AdaptiveRandomForestClassifier
        >>> arf = AdaptiveRandomForestClassifier(n_models=3, seed=0)
        >>> arf.predict_one({"x": 0.0}) is None
        True
        >>> for i in range(100):
        ...     _ = arf.learn_one({"x": float(i % 2)}, i % 2)
        >>> arf.predict_one({"x": 1.0})
        1

    References:
        Gomes et al. (2017), Machine Learning 106, 1469-1495,
        Algorithm 1 (page 7) and Algorithm 2 (page 8).
    """

    budget_s = 0.02
    memory_class = "O(window)"

    def __init__(
        self,
        n_models: int = 10,
        max_features: int | None = None,
        lambda_: float = 6.0,
        drift_detector: DriftDetector | None = None,
        warning_detector: DriftDetector | None = None,
        seed: int | None = None,
        grace_period: int = 50,
    ) -> None:
        if n_models < 1:
            raise ValueError(f"n_models must be >= 1, got {n_models}")
        if max_features is not None and max_features < 1:
            raise ValueError(f"max_features must be >= 1, got {max_features}")
        if lambda_ <= 0.0:
            raise ValueError(f"lambda_ must be > 0, got {lambda_}")
        if grace_period < 1:
            raise ValueError(f"grace_period must be >= 1, got {grace_period}")
        self.n_models = n_models
        self.max_features = max_features
        self.lambda_ = lambda_
        self.drift_detector = drift_detector
        self.warning_detector = warning_detector
        self.seed = seed
        self.grace_period = grace_period
        self.members_: list = []
        self.classes_: set = set()
        self.n_seen_ = 0
        self.n_missing_ = 0
        self.n_warnings_ = 0
        self.n_drifts_ = 0
        self._rng = np.random.default_rng(seed)

    @property
    def n_missing(self) -> int:
        return self.n_missing_

    def _unit_test_skips(self) -> set:
        return {"check_step_is_pure", "check_step_is_jittable"}

    def _new_tree(self) -> SubspaceHoeffdingTree:
        return SubspaceHoeffdingTree(
            grace_period=self.grace_period,
            max_features=self.max_features,
            seed=int(self._rng.integers(2**31 - 1)),
        )

    def _new_drift(self) -> Any:
        if self.drift_detector is None:
            return ADWIN(delta=0.001)
        return self.drift_detector.clone()

    def _new_warning(self) -> Any:
        if self.warning_detector is None:
            return ADWIN(delta=0.01)
        return self.warning_detector.clone()

    def _new_member(self) -> _Member:
        return _Member(self._new_tree(), self._new_drift(), self._new_warning())

    def _train(self, tree: Any, d: dict, y: Any) -> None:
        for _ in range(int(self._rng.poisson(self.lambda_))):
            tree.learn_one(d, y)

    def _promote(self, m: _Member) -> None:
        m.tree = m.background if m.background is not None else self._new_tree()
        m.background = None
        m.correct = 0
        m.seen = 0
        m.drift = self._new_drift()
        m.warn = self._new_warning()

    def learn_one(
        self, x: Any, y: Any, t: float | None = None
    ) -> "AdaptiveRandomForestClassifier":
        """Score, train and watch every tree on one sample.

        A sample that is empty, has a missing or non-finite value, or
        has no label is skipped and counted in ``n_missing``.
        """
        self._time_step(t)
        d = _as_dict(x)
        if not d or y is None or not _finite_values(d):
            self.n_missing_ += 1
            return self
        if not self.members_:
            self.members_ = [self._new_member() for _ in range(self.n_models)]
        self.classes_.add(y)
        self.n_seen_ += 1
        for m in self.members_:
            correct = m.tree.predict_one(d) == y
            m.seen += 1
            m.correct += int(correct)
            self._train(m.tree, d, y)
            err = 0.0 if correct else 1.0
            m.warn.update(err)
            m.drift.update(err)
            if m.warn.drift_detected:
                m.background = self._new_tree()
                self.n_warnings_ += 1
            if m.drift.drift_detected:
                self._promote(m)
                self.n_drifts_ += 1
        for m in self.members_:
            if m.background is not None:
                self._train(m.background, d, y)
        return self

    def predict_proba_one(self, x: Any, t: float | None = None) -> dict:
        """Weighted vote of the class probabilities of the trees."""
        if not self.members_:
            return {}
        d = _as_dict(x)
        votes = [(m.weight, m.tree.predict_proba_one(d)) for m in self.members_]
        votes = [(w, p) for w, p in votes if p]
        if not votes:
            return {}
        if sum(w for w, _ in votes) <= 0.0:
            votes = [(1.0, p) for _, p in votes]
        total: dict = dict.fromkeys(self.classes_, 0.0)
        for w, p in votes:
            for c, v in p.items():
                total[c] = total.get(c, 0.0) + w * v
        z = sum(total.values())
        if z <= 0.0:
            return {}
        return {c: v / z for c, v in total.items()}

    def predict_one(self, x: Any, t: float | None = None) -> Any:
        """Return the class with the largest weighted vote, or ``None``."""
        proba = self.predict_proba_one(x, t=t)
        if not proba:
            return None
        return max(proba.items(), key=lambda kv: kv[1])[0]
