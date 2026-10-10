"""Online Isolation Forest for streaming anomaly detection.

Anomalies are points that a random tree isolates with few splits.
Online-IForest keeps an ensemble of multi-resolution histograms, each
one updated sample by sample: a new point expands the tree, the point
leaving the sliding window contracts it. The score is the average
depth at which the point falls in each tree, mapped to ``[0, 1]`` by
the adjustment factor ``c(window, eta) = log2(window / eta)``.

This module implements Algorithm 1 of Leveni et al. (2024), with the
learning procedure of Algorithm 2 (split when the bin height reaches
``eta * 2^k``), the forgetting procedure of Algorithm 3 (merge when
it drops below ``eta * 2^k``) and the point-depth computation of
Algorithm 4. The node is ``N = (h, R)``: ``h`` the bin height, ``R``
the minimal hyperrectangle that encloses the actual data points that
reached the node.

Ranges are data-driven, as the paper prescribes: at every ``learn_one``
the support ``R`` of each node on the path is extended to include the
new point (Algorithm 2, line 1); the root starts from the first sample
seen. When a leaf splits, ``h`` points are sampled uniformly in its
support, partitioned by a random axis-parallel split, and their two
subsets initialise the children's supports (Algorithm 2, lines 3-7).
When the height of an internal node drops below ``eta * 2^k`` the
node becomes a leaf and its support is rebuilt as the minimal
hyperrectangle that encloses the two children (Algorithm 3, line 3).
The reference implementation (``recursive_unlearn`` ->
``recursive_unbuild``) recursively collapses the whole subtree before
the merge; the support of the merged node is the same as a direct
min/max of ``R_l`` and ``R_r``, so the recursive step only frees
memory and does not change the result. This module follows the
reference implementation.

Details not fixed by the paper, stated explicitly:

- The maximum tree depth is ``delta = log2(window / eta)`` (Section
  4.1); the reference implementation caps it dynamically at
  ``log2(data_size / eta)``, where ``data_size`` is the number of
  points currently in the tree. This module uses the dynamic cap.
- Point depth for a leaf is ``k + log2(h / eta)``, that is
  Algorithm 4's ``k + c(h, eta)`` with ``c(h, eta) =
  log(num_samples / max_leaf_samples) / log(2 * branching_factor)``
  from the reference implementation, ``branching_factor = 2``. The
  paper writes ``c(h, eta)`` in Algorithm 4 without giving the closed
  form; the reference implementation is used here.
- The score's adjustment factor is ``c(window, eta) =
  log2(window / eta)``; the paper writes it in Algorithm 1 line 1 and
  in Section 4.2.

Known limitation on streams with many positives. The paper assumes
"anomalous data are few" (Section 2, ``P(X_i ~ Phi_1) << P(X_i ~
Phi_0)``) and its benchmark (Table 2) has anomalies between 0.03 %
and 10 %. The SyntheticArm benchmark of this repository has 75 % of
positives after the fault (100 healthy, 300 faulty), far outside that
assumption: when the anomalies are the majority, the "sparse" region
of the tree is the healthy one, and the depth score measures the
wrong side. Measured on SyntheticArm, protocol ``learn-on-healthy``,
n_trees=32, eta=8/32:

- 400 samples, 100 healthy (window 200, eta 8): ROC-AUC 0.36
- 1600 samples, 400 healthy (window 2048, eta 32): ROC-AUC 0.54
- 3200 samples, 800 healthy (window 2048, eta 32): ROC-AUC 0.66

The trend is monotonic in the number of healthy samples: with more
data the trees grow and the score improves, but the anomaly
proportion never falls inside the assumption of the paper. On the
paper's own datasets (Cardio, HTTP, Shuttle, ... with 0.03 % to 10 %
anomalies) the reported ROC-AUC is 0.65 to 0.99 (Table 4). On the
``DriftStream`` of this benchmark (50 % positives) the protocol
``learn-on-healthy`` gives 0.67.

References
----------
Leveni, F., Cassales, G. W., Pfahringer, B., Bifet, A., Boracchi, G.
    (2024). Online Isolation Forest. arXiv:2505.09593. Algorithm 1
    (Online-IForest), Algorithm 2 (learn point), Algorithm 3 (forget
    point), Algorithm 4 (point depth), Section 4.1, Section 4.2.
    Reference implementation:
    https://github.com/ineveLoppiliF/Online-Isolation-Forest
    (BoundedRandomProjectionOnlineITree, recursive_learn,
    recursive_build, recursive_unlearn, recursive_unbuild,
    recursive_depth_search).
"""

from collections import deque
from collections.abc import Sequence

import numpy as np

from dense_armor.roles import AnomalyDetector


class _Node:
    """One Online-iTree node.

    Attributes:
        h: number of points that reached the node (bin height).
        depth: depth of the node in the tree, root at 0.
        lo, hi: minimal hyperrectangle that encloses the points that
            reached the node.
        q: dimension of the split, -1 for a leaf.
        p: split value, unused for a leaf.
        children: two ``_Node`` or ``None`` for a leaf.
    """

    __slots__ = ("children", "depth", "h", "hi", "lo", "p", "q")

    def __init__(self, h: int, depth: int, lo: np.ndarray, hi: np.ndarray) -> None:
        self.h = int(h)
        self.depth = int(depth)
        self.lo = lo
        self.hi = hi
        self.q = -1
        self.p = 0.0
        self.children: list[_Node] | None = None


class OnlineIsolationForest(AnomalyDetector):
    """Online Isolation Forest.

    Args:
        n_trees: number of Online-iTrees in the ensemble.
        window: size of the sliding buffer of recent points.
        max_leaf_samples: ``eta`` in the paper; a leaf splits when its
            bin height reaches ``eta * 2 ** k`` at depth ``k``.
        feature_keys: names of the dict keys, in order. ``None`` uses
            the sorted keys of the first sample seen.
        threshold: score above which ``is_outlier`` returns True. The
            score lives in ``[0, 1]``; default ``0.6``.
        seed: seed of the random generator.
        eps: numerical guard.

    Raises:
        ValueError: on a non-positive parameter.

    Examples:
        >>> import numpy as np
        >>> from dense_armor.utility.anomaly.oiforest import (
        ...     OnlineIsolationForest,
        ... )
        >>> ds = OnlineIsolationForest(n_trees=8, window=512,
        ...                             max_leaf_samples=8, seed=0)
        >>> rng = np.random.default_rng(0)
        >>> for _ in range(600):
        ...     _ = ds.learn_one({
        ...         "a": float(rng.normal(0.5, 0.05)),
        ...         "b": float(rng.normal(0.5, 0.05)),
        ...     })
        >>> s = ds.score_one({"a": 0.5, "b": 0.5})
        >>> isinstance(s, float)
        True
    """

    budget_s = 0.07
    memory_class = "O(1)"

    def __init__(
        self,
        n_trees: int = 32,
        window: int = 256,
        max_leaf_samples: int = 8,
        feature_keys: Sequence[str] | None = None,
        threshold: float = 0.6,
        seed: int | None = None,
        eps: float = 1e-9,
    ) -> None:
        if n_trees < 1:
            raise ValueError(f"n_trees must be >= 1, got {n_trees}")
        if window < 2:
            raise ValueError(f"window must be >= 2, got {window}")
        if max_leaf_samples < 2:
            raise ValueError(f"max_leaf_samples must be >= 2, got {max_leaf_samples}")
        if max_leaf_samples > window:
            raise ValueError(
                f"max_leaf_samples ({max_leaf_samples}) must not be "
                f"larger than window ({window})"
            )
        self.n_trees = int(n_trees)
        self.window = int(window)
        self.max_leaf_samples = int(max_leaf_samples)
        self.feature_keys = feature_keys
        self.threshold = float(threshold)
        self.seed = seed
        self.eps = eps
        self.n_missing_ = 0
        self._ready = False
        self._keys_ready = False
        self._keys: list[str] = []
        self._trees: list[_Node | None] = []
        self._tree_size: list[int] = []
        self._tree_limit: list[float] = []
        self._c_score: float = 0.0
        self._buf: deque | None = None
        self._rng: np.random.Generator | None = None

    def _unit_test_skips(self) -> set:
        return set()

    @property
    def n_missing(self) -> int:
        return self.n_missing_

    def _lazy_init(self, x: dict) -> None:
        if self._ready:
            return
        if not self._keys_ready:
            if self.feature_keys is not None:
                self._keys = list(self.feature_keys)
            else:
                self._keys = sorted(x)
            self._keys_ready = True
        self._trees = [None] * self.n_trees
        self._tree_size = [0] * self.n_trees
        self._tree_limit = [0.0] * self.n_trees
        self._c_score = float(np.log2(self.window / self.max_leaf_samples))
        self._buf = deque(maxlen=self.window)
        self._rng = np.random.default_rng(self.seed)
        self._ready = True

    def _to_vec(self, x: dict) -> np.ndarray:
        return np.array([float(x[k]) for k in self._keys], dtype=float)

    def _random_path_length(self, n: int) -> float:
        if n < self.max_leaf_samples:
            return 0.0
        return float(np.log2(n / self.max_leaf_samples))

    def _recursive_build(
        self, data: np.ndarray, depth: int, depth_limit: float
    ) -> _Node:
        n = data.shape[0]
        lo = data.min(axis=0)
        hi = data.max(axis=0)
        threshold = self.max_leaf_samples * (2**depth)
        if n < threshold or depth >= depth_limit:
            return _Node(h=n, depth=depth, lo=lo, hi=hi)
        d = data.shape[1]
        assert self._rng is not None
        q = int(self._rng.integers(0, d))
        if hi[q] - lo[q] < self.eps:
            return _Node(h=n, depth=depth, lo=lo, hi=hi)
        p = float(self._rng.uniform(lo[q], hi[q]))
        left_mask = data[:, q] < p
        right_mask = ~left_mask
        if not left_mask.any() or not right_mask.any():
            return _Node(h=n, depth=depth, lo=lo, hi=hi)
        node = _Node(h=n, depth=depth, lo=lo, hi=hi)
        node.q = q
        node.p = p
        node.children = [
            self._recursive_build(data[left_mask], depth + 1, depth_limit),
            self._recursive_build(data[right_mask], depth + 1, depth_limit),
        ]
        return node

    def _recursive_learn(self, node: _Node, x: np.ndarray, depth_limit: float) -> _Node:
        node.h += 1
        node.lo = np.minimum(node.lo, x)
        node.hi = np.maximum(node.hi, x)
        if node.children is None:
            threshold = self.max_leaf_samples * (2**node.depth)
            if node.h >= threshold and node.depth < depth_limit:
                assert self._rng is not None
                data_sampled = self._rng.uniform(
                    low=node.lo, high=node.hi, size=(node.h, x.size)
                )
                return self._recursive_build(
                    data_sampled, depth=node.depth, depth_limit=depth_limit
                )
            return node
        if x[node.q] < node.p:
            node.children[0] = self._recursive_learn(node.children[0], x, depth_limit)
        else:
            node.children[1] = self._recursive_learn(node.children[1], x, depth_limit)
        return node

    def _recursive_unbuild(self, node: _Node) -> _Node:
        if node.children is None:
            return node
        for i in range(len(node.children)):
            node.children[i] = self._recursive_unbuild(node.children[i])
        node.lo = np.minimum(node.children[0].lo, node.children[1].lo)
        node.hi = np.maximum(node.children[0].hi, node.children[1].hi)
        node.children = None
        node.q = -1
        node.p = 0.0
        return node

    def _recursive_unlearn(
        self, node: _Node, x: np.ndarray, depth_limit: float
    ) -> _Node:
        node.h -= 1
        if node.children is None:
            return node
        threshold = self.max_leaf_samples * (2**node.depth)
        if node.h < threshold:
            return self._recursive_unbuild(node)
        if x[node.q] < node.p:
            node.children[0] = self._recursive_unlearn(node.children[0], x, depth_limit)
        else:
            node.children[1] = self._recursive_unlearn(node.children[1], x, depth_limit)
        node.lo = np.minimum(node.children[0].lo, node.children[1].lo)
        node.hi = np.maximum(node.children[0].hi, node.children[1].hi)
        return node

    def _depth_search(self, node: _Node, x: np.ndarray) -> float:
        if node.children is None:
            return float(node.depth) + self._random_path_length(node.h)
        if x[node.q] < node.p:
            return self._depth_search(node.children[0], x)
        return self._depth_search(node.children[1], x)

    def learn_one(self, x: dict, t: float | None = None) -> "OnlineIsolationForest":
        """Learn one sample: expand the trees, forget the oldest point.

        Args:
            x: a dict of float features.
            t: unused.

        Returns:
            ``self``.
        """
        self._time_step(t)
        if not self._ready:
            self._lazy_init(x)
        try:
            v = np.array([float(x[k]) for k in self._keys], dtype=float)
        except (KeyError, TypeError, ValueError):
            self.n_missing_ += 1
            return self
        if not np.isfinite(v).all():
            self.n_missing_ += 1
            return self
        assert self._buf is not None
        for tr in range(self.n_trees):
            self._tree_size[tr] += 1
            self._tree_limit[tr] = self._random_path_length(self._tree_size[tr])
            cur = self._trees[tr]
            if cur is None:
                self._trees[tr] = self._recursive_build(
                    v.reshape(1, -1),
                    depth=0,
                    depth_limit=self._tree_limit[tr],
                )
            else:
                self._trees[tr] = self._recursive_learn(cur, v, self._tree_limit[tr])
        if len(self._buf) == self.window:
            oldest = self._buf[0]
            for tr in range(self.n_trees):
                self._tree_size[tr] -= 1
                self._tree_limit[tr] = self._random_path_length(self._tree_size[tr])
                cur = self._trees[tr]
                if cur is not None:
                    self._trees[tr] = self._recursive_unlearn(
                        cur, oldest, self._tree_limit[tr]
                    )
        self._buf.append(v)
        return self

    def score_one(self, x: dict, t: float | None = None) -> float:
        """Score one sample: higher means more anomalous.

        Args:
            x: a dict of float features.
            t: unused.

        Returns:
            ``2 ** (-E(D) / c)`` where ``E(D)`` is the average leaf
            depth across the ensemble and ``c = log2(window / eta)``.
            ``1.0`` before any sample is learned.
        """
        if not self._ready:
            return 1.0
        try:
            xv = self._to_vec(x)
        except (KeyError, TypeError, ValueError):
            return 1.0
        depths = np.empty(self.n_trees, dtype=float)
        for tr in range(self.n_trees):
            cur = self._trees[tr]
            if cur is None:
                return 1.0
            depths[tr] = self._depth_search(cur, xv)
        c = max(self._c_score, self.eps)
        return float(2.0 ** (-float(depths.mean()) / c))

    def is_outlier(self, x: dict, threshold: float | None = None) -> bool:
        """Boolean wrapper around :meth:`score_one`."""
        thr = self.threshold if threshold is None else float(threshold)
        return self.score_one(x) > thr
