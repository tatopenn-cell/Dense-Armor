"""Online Extra Trees regressor (OXT) for data streams.

An ensemble of regression trees whose split points are drawn at random,
after Mastelini, Nakano, Vens and de Carvalho. The source read is the
chapter on OXT of Mastelini's doctoral thesis (USP, 2023), not the journal
article; page numbers below are printed pages of the thesis.

The pieces, as the thesis states them:

- Training (Algorithm 3, page 109). For every tree and every sample:
  predict, feed the absolute deviation to a warning detector and a drift
  detector, start a background tree on a warning, replace the tree by the
  background tree on a drift, skip the sample for this tree with chance
  ``1 - p`` (sub-bagging, ``p = 0.5``, page 115), otherwise train the tree
  and its background tree on it. The prediction is the mean of the trees
  (page 110).
- Random splitter (Algorithm 4, page 111). A leaf keeps, for each feature
  of a random subset of size ``sqrt(M)`` (page 110, page 115), a buffer of
  its first ``s = 5`` observations; the range of the buffer gives a split
  point ``phi`` drawn uniformly in it; from then on only the incremental
  variance of the target on each side of ``phi`` is kept. There is one
  split candidate per monitored feature.
- Split attempts every ``grace_period = 50`` samples with the split
  significance ``delta = 0.05``; the pre-pruning test is disabled (page
  115).
- Detectors: ADWIN with confidence 0.01 for the warning and 0.001 for the
  drift, on the absolute deviation (pages 109 and 114).
- Leaves predict with the target mean or a linear model, whichever has the
  lower faded error, with fading factor 0.95 (page 114). 20 trees in the
  experiments (page 115).

Stated as my choice, because the passages read do not give them:

- The merit of a candidate is the variance reduction of Equation 4.1 of the
  thesis (chapter 4), written as
  ``(n_le * n_gt / n^2) * (mean_le - mean_gt)^2``. The thesis says that
  Hoeffding trees use the Hoeffding bound to decide when to split but does
  not print the test; the one used is the ratio of the second best merit to
  the best one against
  ``1 - eps``, with the Hoeffding bound ``eps = sqrt(ln(1/delta) / (2 n))``,
  or ``eps < tau`` with ``tau = 0.05``. The children start from the target
  statistics of the two sides of the split.
- The linear model of a leaf works on standardised features and target,
  starts from zero weights and learns by the delta rule with step 0.01.
- A drift with no warning before it replaces the tree by an empty one.
- ``max_nodes`` and ``max_depth`` bound a tree; the thesis states no limit.
- The size of the feature subset is ``max(1, int(sqrt(M)))``.

Examples:
    >>> import numpy as np
    >>> from dense_armor.utility.forest.oxt import OnlineExtraTreesRegressor
    >>> rng = np.random.default_rng(0)
    >>> model = OnlineExtraTreesRegressor(n_models=5, seed=0)
    >>> for _ in range(1500):
    ...     a = float(rng.uniform(-1.0, 1.0))
    ...     _ = model.learn_one({"a": a, "b": float(rng.uniform(-1.0, 1.0))}, 4.0 * (a > 0.0))
    >>> round(model.predict_one({"a": 0.5, "b": 0.0}), 1), round(model.predict_one({"a": -0.5, "b": 0.0}), 1)
    (4.0, 0.2)

References:
    Mastelini, S. M., Nakano, F. K., Vens, C., de Carvalho, A. C. P. L. F.
        Online extra trees regressor. IEEE Transactions on Neural Networks
        and Learning Systems 34(10), 6755-6767. Read in: Mastelini, S. M.
        (2023). Efficient online tree, rule-based and distance-based
        algorithms. Doctoral thesis, ICMC-USP, chapter 6, Algorithms 3 and 4,
        Sections 6.4 and 6.5.
"""

import math
from typing import Any

import numpy as np

from dense_armor.roles import DriftDetector, Regressor
from dense_armor.utility.drift.adwin import ADWIN


def _as_dict(x: Any) -> dict:
    if hasattr(x, "to_dict"):
        return dict(x.to_dict())
    try:
        return dict(x)
    except (TypeError, ValueError):
        return {}


class _Moments:
    """Running count, mean and sum of squared deviations (Welford)."""

    __slots__ = ("m2", "mean", "n")

    def __init__(self, n: int = 0, mean: float = 0.0, m2: float = 0.0) -> None:
        self.n = n
        self.mean = mean
        self.m2 = m2

    def update(self, v: float) -> None:
        self.n += 1
        d = v - self.mean
        self.mean += d / self.n
        self.m2 += d * (v - self.mean)

    @property
    def std(self) -> float:
        return math.sqrt(self.m2 / self.n) if self.n > 1 else 0.0


class _RandomSplitter:
    """Algorithm 4: a buffer, one random split point, two variance estimators."""

    __slots__ = ("buf", "gt", "le", "phi")

    def __init__(self) -> None:
        self.buf: list | None = []
        self.phi = 0.0
        self.le = _Moments()
        self.gt = _Moments()

    @property
    def ready(self) -> bool:
        return self.buf is None

    def _add(self, x: float, y: float) -> None:
        (self.le if x <= self.phi else self.gt).update(y)

    def update(self, x: float, y: float, size: int, rng: np.random.Generator) -> None:
        if self.buf is not None:
            if len(self.buf) < size:
                self.buf.append((x, y))
                return
            xs = [b[0] for b in self.buf]
            self.phi = float(rng.uniform(min(xs), max(xs)))
            for bx, by in self.buf:
                self._add(bx, by)
            self.buf = None
        self._add(x, y)

    def merit(self) -> float:
        n = self.le.n + self.gt.n
        if self.le.n == 0 or self.gt.n == 0:
            return 0.0
        return self.le.n * self.gt.n / (n * n) * (self.le.mean - self.gt.mean) ** 2


def _split_choice(merits: list, n: int, delta: float, tau: float) -> int | None:
    """Index of the candidate to split on, or ``None`` (ratio test of the bound)."""
    if not merits:
        return None
    order = sorted(range(len(merits)), key=lambda i: merits[i], reverse=True)
    best = merits[order[0]]
    if best <= 0.0:
        return None
    second = merits[order[1]] if len(order) > 1 else 0.0
    eps = math.sqrt(math.log(1.0 / delta) / (2.0 * n))
    if second / best + eps < 1.0 or eps < tau:
        return order[0]
    return None


class _LeafModel:
    """Target mean or linear model, whichever has the lower faded error."""

    __slots__ = ("attrs", "n_lin", "n_mean", "t_lin", "t_mean", "target", "w")

    def __init__(self, d: int) -> None:
        self.attrs = [_Moments() for _ in range(d)]
        self.w = np.zeros(d + 1)
        self.target = _Moments()
        self.t_mean = self.n_mean = self.t_lin = self.n_lin = 0.0

    def _z(self, v: np.ndarray) -> np.ndarray:
        mu = np.array([a.mean for a in self.attrs])
        sd = np.array([a.std if a.std > 1e-12 else 1.0 for a in self.attrs])
        return np.append((v - mu) / sd, 1.0)

    def _scale(self) -> float:
        return self.target.std if self.target.std > 1e-12 else 1.0

    def predict_mean(self) -> float:
        return self.target.mean

    def predict_linear(self, v: np.ndarray) -> float:
        return float(self.w @ self._z(v)) * self._scale() + self.target.mean

    def predict(self, v: np.ndarray) -> float:
        if self.n_mean == 0.0 or self.n_lin == 0.0:
            return self.predict_mean()
        if self.t_lin / self.n_lin < self.t_mean / self.n_mean:
            return self.predict_linear(v)
        return self.predict_mean()

    def learn(self, v: np.ndarray, y: float, eta: float, fade: float) -> None:
        self.t_mean = fade * self.t_mean + abs(self.predict_mean() - y)
        self.n_mean = fade * self.n_mean + 1.0
        self.t_lin = fade * self.t_lin + abs(self.predict_linear(v) - y)
        self.n_lin = fade * self.n_lin + 1.0
        if self.target.n > 1:
            z = self._z(v)
            yn = (y - self.target.mean) / self._scale()
            self.w = self.w - eta * (float(self.w @ z) - yn) * z
        for j, a in enumerate(self.attrs):
            a.update(float(v[j]))
        self.target.update(y)


class _Node:
    __slots__ = ("depth", "feature", "left", "model", "n", "phi", "right", "splitters")

    def __init__(self, depth: int, d: int) -> None:
        self.depth = depth
        self.model = _LeafModel(d)
        self.n = 0
        self.splitters: dict = {}
        self.feature = -1
        self.phi = 0.0
        self.left: _Node | None = None
        self.right: _Node | None = None


class _RandomTree:
    """Hoeffding-style regression tree with random subsets and random splits."""

    def __init__(self, d: int, cfg: dict, rng: np.random.Generator) -> None:
        self.d = d
        self.cfg = cfg
        self.rng = rng
        self.n_nodes = 0
        self.n_learned = 0
        self.root = self._leaf(0)

    def _leaf(self, depth: int) -> _Node:
        node = _Node(depth, self.d)
        size = self.cfg["max_features"] or max(1, int(math.sqrt(self.d)))
        pick = self.rng.choice(self.d, size=min(size, self.d), replace=False)
        node.splitters = {int(j): _RandomSplitter() for j in pick}
        self.n_nodes += 1
        return node

    def _route(self, v: np.ndarray) -> _Node:
        node = self.root
        while node.left is not None and node.right is not None:
            node = node.left if v[node.feature] <= node.phi else node.right
        return node

    def predict(self, v: np.ndarray) -> float:
        return self._route(v).model.predict(v)

    def learn(self, v: np.ndarray, y: float) -> None:
        cfg = self.cfg
        leaf = self._route(v)
        leaf.model.learn(v, y, cfg["learning_rate"], cfg["fading"])
        leaf.n += 1
        self.n_learned += 1
        for j, sp in leaf.splitters.items():
            sp.update(float(v[j]), y, cfg["buffer_size"], self.rng)
        if leaf.n % cfg["grace_period"] == 0:
            self._attempt(leaf)

    def _attempt(self, leaf: _Node) -> None:
        cfg = self.cfg
        if self.n_nodes + 2 > cfg["max_nodes"]:
            return
        if cfg["max_depth"] is not None and leaf.depth >= cfg["max_depth"]:
            return
        feats = [j for j, sp in leaf.splitters.items() if sp.ready]
        merits = [leaf.splitters[j].merit() for j in feats]
        k = _split_choice(merits, leaf.n, cfg["delta"], cfg["tau"])
        if k is None:
            return
        sp = leaf.splitters[feats[k]]
        leaf.feature, leaf.phi = feats[k], sp.phi
        leaf.left, leaf.right = self._leaf(leaf.depth + 1), self._leaf(leaf.depth + 1)
        leaf.left.model.target = _Moments(sp.le.n, sp.le.mean, sp.le.m2)
        leaf.right.model.target = _Moments(sp.gt.n, sp.gt.mean, sp.gt.m2)
        leaf.splitters = {}


class _Member:
    def __init__(self, tree: _RandomTree, warn: Any, drift: Any) -> None:
        self.tree = tree
        self.warn = warn
        self.drift = drift
        self.background: _RandomTree | None = None


class OnlineExtraTreesRegressor(Regressor):
    """Online Extra Trees regressor (Mastelini et al.).

    Cost and memory. ``learn_one`` predicts with every tree, updates two
    detectors per tree and trains about ``p * n_models`` trees (plus their
    background trees); ``predict_one`` descends every tree. A tree has at
    most ``max_nodes`` nodes, a leaf keeps a buffer or two variance
    estimators for ``sqrt(M)`` features, so the state is constant in the
    length of the stream and linear in ``n_models`` and in the number of
    features. There is no pure ``init_state`` / ``step`` form: the state is
    a set of trees of changing shape.

    Args:
        n_models: number of trees ``Nt``. Default 20.
        p: chance that a tree trains on a sample (sub-bagging). Default 0.5.
        max_features: features tried per split attempt. ``None`` takes
            ``max(1, int(sqrt(M)))``.
        buffer_size: size ``s`` of the buffer of the random splitter.
            Default 5.
        grace_period: samples between split attempts of a leaf. Default 50.
        delta: split significance of the Hoeffding bound. Default 0.05.
        tau: tie threshold. Default 0.05.
        drift_detector: template of the drift detector, cloned for every
            tree. ``None`` takes ``ADWIN(delta=0.001)``.
        warning_detector: template of the warning detector. ``None`` takes
            ``ADWIN(delta=0.01)``.
        fading: fading factor of the error that chooses between the target
            mean and the linear model. Default 0.95.
        learning_rate: step of the delta rule of the linear models. Default
            0.01.
        max_nodes: largest number of nodes of one tree. Default 1000.
        max_depth: largest depth of a tree, or ``None``.
        seed: seed of the generator of the sampling and of the splits.
        feature_keys: names of the features. ``None`` takes the sorted keys
            of the first sample.

    Raises:
        ValueError: on a non-positive ``n_models``, ``buffer_size``,
            ``grace_period``, ``max_nodes`` or ``max_features``, on ``p``
            outside (0, 1], ``delta`` outside (0, 1), ``fading`` outside
            (0, 1), a negative ``tau``, a non-positive ``learning_rate`` or
            ``max_depth``.

    Examples:
        >>> from dense_armor.utility.forest.oxt import OnlineExtraTreesRegressor
        >>> model = OnlineExtraTreesRegressor(n_models=3, seed=0)
        >>> model.predict_one({"a": 1.0})
        0.0
        >>> _ = model.learn_one({"a": 1.0}, 2.0)
        >>> model.predict_one({"a": 1.0})
        1.3333333333333333

    References:
        Mastelini et al., Online extra trees regressor, Algorithms 3 and 4.
    """

    budget_s = 0.2
    memory_class = "O(window)"

    def __init__(
        self,
        n_models: int = 20,
        p: float = 0.5,
        max_features: int | None = None,
        buffer_size: int = 5,
        grace_period: int = 50,
        delta: float = 0.05,
        tau: float = 0.05,
        drift_detector: DriftDetector | None = None,
        warning_detector: DriftDetector | None = None,
        fading: float = 0.95,
        learning_rate: float = 0.01,
        max_nodes: int = 1000,
        max_depth: int | None = None,
        seed: int | None = None,
        feature_keys: list | None = None,
    ) -> None:
        if n_models < 1 or buffer_size < 1 or grace_period < 1 or max_nodes < 1:
            raise ValueError(
                "n_models, buffer_size, grace_period and max_nodes must be >= 1"
            )
        if max_features is not None and max_features < 1:
            raise ValueError(f"max_features must be >= 1, got {max_features}")
        if not 0.0 < p <= 1.0:
            raise ValueError(f"p must be in (0, 1], got {p}")
        if not 0.0 < delta < 1.0 or not 0.0 < fading < 1.0:
            raise ValueError("delta and fading must be in (0, 1)")
        if tau < 0.0 or learning_rate <= 0.0:
            raise ValueError("tau must be >= 0 and learning_rate > 0")
        if max_depth is not None and max_depth < 1:
            raise ValueError(f"max_depth must be >= 1, got {max_depth}")
        self.n_models = n_models
        self.p = p
        self.max_features = max_features
        self.buffer_size = buffer_size
        self.grace_period = grace_period
        self.delta = delta
        self.tau = tau
        self.drift_detector = drift_detector
        self.warning_detector = warning_detector
        self.fading = fading
        self.learning_rate = learning_rate
        self.max_nodes = max_nodes
        self.max_depth = max_depth
        self.seed = seed
        self.feature_keys = feature_keys
        self.members_: list = []
        self.n_seen_ = 0
        self.n_missing_ = 0
        self.n_warnings_ = 0
        self.n_drifts_ = 0
        self._keys: list | None = None
        self._rng = np.random.default_rng(seed)

    @property
    def n_missing(self) -> int:
        return self.n_missing_

    def _vector(self, x: Any) -> np.ndarray | None:
        d = _as_dict(x)
        if not d:
            return None
        if self._keys is None:
            self._keys = (
                list(self.feature_keys) if self.feature_keys is not None else sorted(d)
            )
        try:
            v = np.array([float(d[k]) for k in self._keys], dtype=float)
        except (KeyError, TypeError, ValueError):
            return None
        return v if np.isfinite(v).all() else None

    def _tree(self, d: int) -> _RandomTree:
        cfg = {
            "max_features": self.max_features,
            "buffer_size": self.buffer_size,
            "grace_period": self.grace_period,
            "delta": self.delta,
            "tau": self.tau,
            "fading": self.fading,
            "learning_rate": self.learning_rate,
            "max_nodes": self.max_nodes,
            "max_depth": self.max_depth,
        }
        return _RandomTree(
            d, cfg, np.random.default_rng(int(self._rng.integers(2**31 - 1)))
        )

    def _detector(self, template: Any, delta: float) -> Any:
        return ADWIN(delta=delta) if template is None else template.clone()

    def _member(self, d: int) -> _Member:
        return _Member(
            self._tree(d),
            self._detector(self.warning_detector, 0.01),
            self._detector(self.drift_detector, 0.001),
        )

    def learn_one(
        self, x: Any, y: Any, t: float | None = None
    ) -> "OnlineExtraTreesRegressor":
        """Run Algorithm 3 on one sample.

        A sample with a missing or non-finite value or target is skipped
        and counted in ``n_missing``.
        """
        self._time_step(t)
        try:
            target = float(y)
        except (TypeError, ValueError):
            target = math.nan
        v = self._vector(x) if math.isfinite(target) else None
        if v is None:
            self.n_missing_ += 1
            return self
        d = len(v)
        if not self.members_:
            self.members_ = [self._member(d) for _ in range(self.n_models)]
        self.n_seen_ += 1
        for m in self.members_:
            err = abs(target - m.tree.predict(v))
            m.warn.update(err)
            m.drift.update(err)
            if m.warn.drift_detected:
                m.background = self._tree(d)
                self.n_warnings_ += 1
            if m.drift.drift_detected:
                m.tree = m.background if m.background is not None else self._tree(d)
                m.background = None
                m.warn = self._detector(self.warning_detector, 0.01)
                m.drift = self._detector(self.drift_detector, 0.001)
                self.n_drifts_ += 1
            if self._rng.random() > self.p:
                continue
            m.tree.learn(v, target)
            if m.background is not None:
                m.background.learn(v, target)
        return self

    def predict_one(
        self, x: Any, t: float | None = None, return_std: bool = False
    ) -> Any:
        """Mean of the trees, 0.0 before the first sample.

        With ``return_std`` the result is a pair: the mean and the standard
        deviation of the predictions of the trees.
        """
        v = self._vector(x) if self.members_ else None
        if v is None:
            return (0.0, 0.0) if return_std else 0.0
        preds = np.array([m.tree.predict(v) for m in self.members_])
        if return_std:
            return float(preds.mean()), float(preds.std())
        return float(preds.mean())
