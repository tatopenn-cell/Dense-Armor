"""Half-Space Trees for streaming anomaly detection.

Anomalous points sit in sparse regions of the space. Half-Space Trees
split the space recursively on random dimensions and count how often
each region is visited: rare leaves flag anomalies.

Design follows Leveni et al. (2024), section 3 (related work), which
describes Half Space Trees (HST) as: an ensemble of complete binary
trees, each split on a random dimension at the mid-point of the current
range, scored from node masses. This module implements that design;
the original paper is Tan et al. (2011).

Choices not given by Leveni et al. (2024), stated explicitly:

- Working range: ``[lo_i, hi_i]`` per feature. When ``feature_ranges``
  is not given, the ranges are learned online from the first
  ``range_init`` samples (min and max per feature, with a small
  margin); until then the score is 0. Points are not clipped: a value
  outside the observed range maps to a negative or >1 normalised
  position and lands in the outermost leaf on that side, which is the
  rarest region.
- Tree depth is fixed at construction; splits are at the mid-point of
  the *current* node's range, which is entirely determined by the BFS
  position of the node.
- Sliding window: exact window of the last ``window`` samples, kept as
  a circular buffer of the node paths (not a decay). No periodic
  retraining; the window itself is the forgetting mechanism.
- Score: for each tree, ``-log2((c_leaf + 1) / (N + 1))``, where
  ``c_leaf`` is the count of samples in the window whose path ended at
  that leaf and ``N`` is the number of samples in the window. The
  ensemble score is the mean over trees. Higher means more anomalous.
- Threshold: when ``threshold`` is ``None`` (default) the effective
  threshold is the ``threshold_quantile``-quantile of the last
  ``threshold_window`` scores (default 0.95 and 500); while fewer than
  ``threshold_warmup`` scores have been seen the effective threshold
  is ``1e9`` (flag nothing). When ``threshold`` is a float, it is used
  as a fixed threshold.

References
----------
Leveni, F., Cassales, G. W., Pfahringer, B., Bifet, A., Boracchi, G.
    (2024). Online Isolation Forest. arXiv:2505.09593. Section 3
    (related work).
Tan, S. C., Ting, K. M., Liu, T. F. (2011). Fast anomaly detection for
    streaming data. In IJCAI.
"""

from collections import deque
from collections.abc import Sequence

import numpy as np

from dense_armor.roles import AnomalyDetector


class HalfSpaceTrees(AnomalyDetector):
    """Half-Space Trees for streaming anomaly detection.

    Args:
        n_trees: number of trees in the ensemble.
        depth: depth of every tree. The tree is a complete binary tree
            with ``2 ** depth`` leaves.
        window: number of most recent samples kept in the sliding
            window. The window is the forgetting mechanism.
        feature_ranges: ``[(lo, hi), ...]``, one per feature. ``None``
            (default) learns the ranges from the first ``range_init``
            samples.
        range_init: number of samples used to learn the working ranges
            when ``feature_ranges`` is ``None``. Default 20.
        feature_keys: names of the dict keys, in order. ``None`` uses
            the sorted keys of the first sample seen.
        threshold: fixed threshold, or ``None`` for the adaptive
            quantile rule (see the module docstring).
        threshold_quantile: quantile of the recent scores used as the
            effective threshold when ``threshold`` is ``None``.
            Default 0.95.
        threshold_window: number of most recent scores kept for the
            adaptive threshold. Default 500.
        threshold_warmup: number of scores required before the
            adaptive threshold becomes effective. Default 50.
        seed: seed of the random generator that chooses the split
            dimensions.
        eps: numerical guard.

    Raises:
        ValueError: on a non-positive parameter.

    Examples:
        >>> import numpy as np
        >>> from dense_armor.utility.anomaly.hst import HalfSpaceTrees
        >>> hst = HalfSpaceTrees(n_trees=5, depth=6, window=30,
        ...                      range_init=5, seed=0)
        >>> for _ in range(50):
        ...     _ = hst.learn_one({"a": 0.5, "b": 0.5})
        >>> s = hst.score_one({"a": 0.5, "b": 0.5})
        >>> isinstance(s, float)
        True
    """

    budget_s = 0.06
    memory_class = "O(1)"

    def __init__(
        self,
        n_trees: int = 25,
        depth: int = 15,
        window: int = 250,
        feature_ranges: Sequence[tuple[float, float]] | None = None,
        range_init: int = 20,
        feature_keys: Sequence[str] | None = None,
        threshold: float | None = None,
        threshold_quantile: float = 0.95,
        threshold_window: int = 500,
        threshold_warmup: int = 50,
        seed: int | None = None,
        eps: float = 1e-9,
    ) -> None:
        if n_trees < 1:
            raise ValueError(f"n_trees must be >= 1, got {n_trees}")
        if depth < 1:
            raise ValueError(f"depth must be >= 1, got {depth}")
        if window < 1:
            raise ValueError(f"window must be >= 1, got {window}")
        if range_init < 1:
            raise ValueError(f"range_init must be >= 1, got {range_init}")
        if not (0.0 < threshold_quantile < 1.0):
            raise ValueError(
                f"threshold_quantile must be in (0, 1), got {threshold_quantile}"
            )
        self.n_trees = int(n_trees)
        self.depth = int(depth)
        self.window = int(window)
        self.feature_ranges = feature_ranges
        self.range_init = int(range_init)
        self.feature_keys = feature_keys
        self.threshold_quantile = float(threshold_quantile)
        self.threshold_window = int(threshold_window)
        self.threshold_warmup = int(threshold_warmup)
        self._threshold_fixed: float | None = (
            None if threshold is None else float(threshold)
        )
        self.threshold = 1e9 if threshold is None else float(threshold)
        self.seed = seed
        self.eps = eps
        self.n_missing_ = 0
        self._ready = False
        self._keys_ready = False
        self._keys: list[str] = []
        self._lo: np.ndarray | None = None
        self._hi: np.ndarray | None = None
        self._feature: np.ndarray | None = None
        self._midpoints: np.ndarray | None = None
        self._counts: np.ndarray | None = None
        self._buf_paths: np.ndarray | None = None
        self._buf_head = 0
        self._buf_size = 0
        self._range_buf: list[np.ndarray] = []
        self._score_hist: deque = deque(maxlen=threshold_window)

    def _unit_test_skips(self) -> set:
        return set()

    @property
    def n_missing(self) -> int:
        return self.n_missing_

    def _lazy_init(self, x: dict) -> None:
        if self._keys_ready:
            return
        if self.feature_keys is not None:
            self._keys = list(self.feature_keys)
        else:
            self._keys = sorted(x)
        self._keys_ready = True
        d = len(self._keys)
        if self.feature_ranges is not None:
            ranges = list(self.feature_ranges)
            if len(ranges) != d:
                raise ValueError(
                    f"feature_ranges has {len(ranges)} entries, expected {d}"
                )
            self._lo = np.array([r[0] for r in ranges], dtype=float)
            self._hi = np.array([r[1] for r in ranges], dtype=float)
            self._finalise_range()

    def _finalise_range(self) -> None:
        assert self._lo is not None and self._hi is not None
        d = len(self._keys)
        rng = np.random.default_rng(self.seed)
        n_internal = 2**self.depth - 1
        self._feature = rng.integers(
            0, d, size=(self.n_trees, n_internal), dtype=np.intp
        )
        levels = np.repeat(np.arange(self.depth), 2 ** np.arange(self.depth))
        k = np.concatenate([np.arange(2**j) for j in range(self.depth)])
        self._midpoints = (2.0 * k + 1.0) / (2.0 ** (levels + 1))
        self._counts = np.zeros(
            (self.n_trees, 2 ** (self.depth + 1) - 1), dtype=np.int64
        )
        self._buf_paths = np.zeros(
            (self.window, self.n_trees, self.depth + 1), dtype=np.intp
        )
        self._buf_head = 0
        self._buf_size = 0
        self._range_buf = []
        self._ready = True

    def _to_norm(self, x: dict) -> np.ndarray:
        v = np.array([float(x[k]) for k in self._keys], dtype=float)
        lo = self._lo
        hi = self._hi
        assert lo is not None and hi is not None
        span = np.maximum(hi - lo, self.eps)
        return (v - lo) / span

    def _path_one(self, tr: int, x_norm: np.ndarray) -> np.ndarray:
        assert self._feature is not None
        assert self._midpoints is not None
        path = np.zeros(self.depth + 1, dtype=np.intp)
        node = 0
        for lv in range(self.depth):
            feat = self._feature[tr, node]
            if x_norm[feat] <= self._midpoints[node]:
                node = 2 * node + 1
            else:
                node = 2 * node + 2
            path[lv + 1] = node
        return path

    def _leaf_one(self, tr: int, x_norm: np.ndarray) -> int:
        assert self._feature is not None
        assert self._midpoints is not None
        node = 0
        for _ in range(self.depth):
            feat = self._feature[tr, node]
            if x_norm[feat] <= self._midpoints[node]:
                node = 2 * node + 1
            else:
                node = 2 * node + 2
        return int(node)

    def _raw_score(self, x: dict) -> float:
        try:
            x_norm = self._to_norm(x)
        except (KeyError, TypeError, ValueError):
            return 0.0
        assert self._counts is not None
        n = self._buf_size
        if n == 0:
            return 0.0
        scores = np.empty(self.n_trees, dtype=float)
        for tr in range(self.n_trees):
            leaf = self._leaf_one(tr, x_norm)
            c = int(self._counts[tr, leaf])
            scores[tr] = -np.log2((c + 1.0) / (n + 1.0))
        return float(scores.mean())

    def _update_threshold(self, s: float) -> None:
        if self._threshold_fixed is not None:
            return
        self._score_hist.append(s)
        if len(self._score_hist) < self.threshold_warmup:
            self.threshold = 1e9
            return
        self.threshold = float(
            np.quantile(list(self._score_hist), self.threshold_quantile)
        )

    def learn_one(self, x: dict, t: float | None = None) -> "HalfSpaceTrees":
        """Add one sample to the sliding window.

        Args:
            x: a dict of float features.
            t: unused.

        Returns:
            ``self``.

        Raises:
            ValueError: if a feature is not finite.
        """
        self._time_step(t)
        if not self._ready:
            self._lazy_init(x)
            if not self._ready:
                try:
                    v_r = np.array([float(x[k]) for k in self._keys], dtype=float)
                except (KeyError, TypeError, ValueError):
                    self.n_missing_ += 1
                    return self
                if not np.isfinite(v_r).all():
                    self.n_missing_ += 1
                    return self
                self._range_buf.append(v_r)
                if len(self._range_buf) >= self.range_init:
                    arr = np.array(self._range_buf, dtype=float)
                    lo = arr.min(axis=0)
                    hi = arr.max(axis=0)
                    span = np.maximum(hi - lo, self.eps)
                    self._lo = lo - 0.1 * span
                    self._hi = hi + 0.1 * span
                    self._finalise_range()
                return self
        try:
            v_check = np.array([x[k] for k in sorted(x)], dtype=float)
        except (KeyError, TypeError, ValueError):
            self.n_missing_ += 1
            return self
        if not np.isfinite(v_check).all():
            self.n_missing_ += 1
            return self
        try:
            x_norm = self._to_norm(x)
        except (KeyError, TypeError, ValueError):
            self.n_missing_ += 1
            return self
        self._update_threshold(self._raw_score(x))
        assert self._counts is not None
        assert self._buf_paths is not None
        paths = np.zeros((self.n_trees, self.depth + 1), dtype=np.intp)
        for tr in range(self.n_trees):
            paths[tr] = self._path_one(tr, x_norm)
            self._counts[tr, paths[tr]] += 1
        if self._buf_size == self.window:
            oldest = self._buf_paths[self._buf_head]
            for tr in range(self.n_trees):
                self._counts[tr, oldest[tr]] -= 1
        self._buf_paths[self._buf_head] = paths
        self._buf_head = (self._buf_head + 1) % self.window
        self._buf_size = min(self._buf_size + 1, self.window)
        return self

    def score_one(self, x: dict, t: float | None = None) -> float:
        """Score one sample: higher means more anomalous.

        Args:
            x: a dict of float features.
            t: unused.

        Returns:
            The mean over trees of ``-log2((c_leaf + 1) / (N + 1))``.
            ``0.0`` while the ranges are being learned or the model has
            not seen any finite sample yet.
        """
        if not self._ready:
            return 0.0
        return self._raw_score(x)

    def is_outlier(self, x: dict, threshold: float | None = None) -> bool:
        """Boolean wrapper around :meth:`score_one`."""
        thr = self.threshold if threshold is None else float(threshold)
        return self.score_one(x) > thr
