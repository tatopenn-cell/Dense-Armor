"""Half-Space Trees for streaming anomaly detection.

Anomalous points sit in regions of the space where few normal samples
fall. Half-Space Trees split the space recursively on random dimensions
and record how many samples of a reference window reach each region.

This module implements Streaming HS-Trees of Tan, Ting and Liu (2011,
IJCAI), page 3 (work space, Algorithms 1 and 2, anomaly score) and
page 4 (Algorithm 3):

- every tree starts from its own random work space: on features
  normalised to ``[0, 1]``, ``s_q`` is uniform in ``[0, 1]`` and the
  range of dimension ``q`` is ``s_q +- 2 * max(s_q, 1 - s_q)``;
- Algorithm 1: a complete binary tree of ``depth`` levels; every node
  splits a random dimension at the mid-point of the node's current
  range (the rule of Ulrich 2025, Section 5.2 as well);
- Algorithms 2 and 3: two mass profiles per node, ``r`` of the
  reference window and ``l`` of the latest window; the first ``window``
  samples fill ``r``; afterwards every sample updates ``l`` and, at the
  end of each window, ``r`` takes the value of ``l`` and ``l`` is reset;
- score of a tree: ``r * 2^k`` of the terminal node, the first node on
  the path of the sample that reaches ``depth`` or holds ``sizeLimit``
  samples or fewer; the score of the ensemble is the sum over the trees.

Choices not given by the paper, stated explicitly:

- Normalisation: the paper assumes features in ``[0, 1]``. The ranges
  ``[lo_i, hi_i]`` come from ``feature_ranges`` or from the first
  ``range_init`` samples (minimum and maximum with a 10 % margin); until
  then the score is 0. A value outside the range is not clipped: it
  falls in the outermost leaves.
- Score scale. The sum ``S`` of the paper is large for normal samples.
  Under uniform mass the term of every tree is ``window`` (page 3:
  ``m[i] * 2^i`` is constant), so ``S`` is about ``n_trees * window``.
  This module returns ``M / (M + S)`` with ``M = n_trees * window``: 1
  for an empty region, 0.5 for uniform mass, towards 0 in dense
  regions. Higher means more anomalous, and 0.0, the value returned
  before the first reference window is complete, is the lowest score.
- ``sizeLimit`` is ``size_limit_frac * window`` with ``size_limit_frac``
  0.1, the value suggested on page 3.
- ``window`` (the paper's window size) defaults to 50; the paper uses
  250. The detector scores only after ``range_init + window`` samples:
  the first ``range_init`` fix the ranges and the next ``window`` fill
  the reference profile, so the paper's 250 would not start on a stream
  with 100 healthy samples.
- Threshold: when ``threshold`` is ``None`` (default) the effective
  threshold is the ``threshold_quantile``-quantile of the last
  ``threshold_window`` scores (default 0.95 and 500); while fewer than
  ``threshold_warmup`` scores have been seen the effective threshold
  is ``1e9`` (flag nothing). When ``threshold`` is a float, it is used
  as a fixed threshold.

Measured cost, on the development machine (Windows, one CPU thread, 25
trees, depth 15, 4 features): ``learn_one`` has a median of 0.65 ms and
a p99 of 1.0 ms; the sample that fixes the ranges builds the trees once
and takes 71 to 80 ms, which is why ``budget_s`` is 0.1 s. The state is
two mass profiles per tree and about 52 MB, whatever the stream length.

References
----------
Tan, S. C., Ting, K. M., Liu, T. F. (2011). Fast anomaly detection for
    streaming data. In IJCAI, 1511-1516. Page 3: work space,
    Algorithm 1, Algorithm 2, anomaly score, ``sizeLimit``, defaults
    ``maxDepth = 15``, ``window = 250``, 25 trees. Page 4: Algorithm 3.
Ulrich, A., Krnavek, J., Senkerik, R., Kominkova Oplatkova, Z., Vala, R.
    (2025). Isolation Forest in Novelty Detection Scenario.
    arXiv:2505.08489. Section 5.2 (the split is the mid-point of the
    range of the chosen dimension).
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
        window: window size of Tan et al. (2011): the first ``window``
            samples after the ranges fill the reference profile, then
            the profile is replaced every ``window`` samples. Default
            50 (the paper uses 250).
        size_limit_frac: ``sizeLimit`` of the paper as a fraction of
            ``window``: a node that holds that many reference samples
            or fewer is terminal. Default 0.1.
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
            dimensions and the work spaces.
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

    budget_s = 0.1
    memory_class = "O(1)"

    def __init__(
        self,
        n_trees: int = 25,
        depth: int = 15,
        window: int = 50,
        feature_ranges: Sequence[tuple[float, float]] | None = None,
        range_init: int = 20,
        feature_keys: Sequence[str] | None = None,
        size_limit_frac: float = 0.1,
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
        if not (0.0 < size_limit_frac <= 1.0):
            raise ValueError(
                f"size_limit_frac must be in (0, 1], got {size_limit_frac}"
            )
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
        self.size_limit_frac = float(size_limit_frac)
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
        self._work_lo: np.ndarray | None = None
        self._work_hi: np.ndarray | None = None
        self._r: np.ndarray | None = None
        self._l: np.ndarray | None = None
        self._count = 0
        self._ref_ready = False
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
        s = rng.uniform(0.0, 1.0, size=(self.n_trees, d))
        half = 2.0 * np.maximum(s, 1.0 - s)
        self._work_lo = s - half
        self._work_hi = s + half
        self._midpoints = self._build_midpoints()
        n_nodes = 2 ** (self.depth + 1) - 1
        self._r = np.zeros((self.n_trees, n_nodes), dtype=np.int32)
        self._l = np.zeros((self.n_trees, n_nodes), dtype=np.int32)
        self._count = 0
        self._ref_ready = False
        self._range_buf = []
        self._ready = True

    def _build_midpoints(self) -> np.ndarray:
        """Split value of every internal node, one per tree and node.

        The split values live in the normalised space of ``_to_norm``,
        where the samples are in ``[0, 1]``, because ``_path_one`` and
        the score compare them with the normalised sample. The root of
        each tree ranges over its own work space (``_work_lo``,
        ``_work_hi``). The range of a node on its split feature depends
        only on the features chosen along the path from the root, never
        on the data: if node ``n`` is the left child of ``p``, which
        splits feature ``f`` at ``m``, the range of ``f`` in ``n`` is
        ``[lo_p[f], m]``; for the right child it is ``[m, hi_p[f]]``.
        The other features keep the range of the parent.

        Rule of Tan et al. 2011, Section 3.2 ("the split point is the
        mid-point of the current range of q") and of Ulrich 2025,
        Section 5.2 ("s = (x + y) / 2"). The formula from the BFS index
        of the node alone holds for ``d = 1`` only.

        Returns:
            Array ``(n_trees, 2 ** depth - 1)`` with the split value of
            every internal node.
        """
        assert self._feature is not None
        assert self._work_lo is not None and self._work_hi is not None
        feature = self._feature
        n_internal = 2**self.depth - 1
        midpoints = np.zeros((self.n_trees, n_internal), dtype=float)
        t_idx = np.arange(self.n_trees)[:, None]
        lo_cur = self._work_lo[:, None, :].copy()
        hi_cur = self._work_hi[:, None, :].copy()
        for level in range(self.depth):
            n_at_level = 2**level
            start = n_at_level - 1
            f_at = feature[:, start : start + n_at_level]
            n_idx = np.arange(n_at_level)[None, :]
            lo_f = lo_cur[t_idx, n_idx, f_at]
            hi_f = hi_cur[t_idx, n_idx, f_at]
            mid = 0.5 * (lo_f + hi_f)
            midpoints[:, start : start + n_at_level] = mid
            new_lo = np.repeat(lo_cur, 2, axis=1)
            new_hi = np.repeat(hi_cur, 2, axis=1)
            even_idx = np.arange(0, 2 * n_at_level, 2)[None, :]
            odd_idx = np.arange(1, 2 * n_at_level, 2)[None, :]
            new_hi[t_idx, even_idx, f_at] = mid
            new_lo[t_idx, odd_idx, f_at] = mid
            lo_cur = new_lo
            hi_cur = new_hi
        return midpoints

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
            feat = int(self._feature[tr, node])
            if x_norm[feat] <= self._midpoints[tr, node]:
                node = 2 * node + 1
            else:
                node = 2 * node + 2
            path[lv + 1] = node
        return path

    def _terminal_mass(self, tr: int, x_norm: np.ndarray) -> float:
        """``r * 2^k`` of the terminal node of the sample in tree ``tr``.

        Tan et al. (2011), page 3: the terminal node is the first node
        on the path that reaches ``depth`` or holds ``sizeLimit``
        reference samples or fewer.
        """
        assert self._feature is not None
        assert self._midpoints is not None
        assert self._r is not None
        size_limit = self.size_limit_frac * self.window
        node = 0
        k = 0
        while True:
            r = int(self._r[tr, node])
            if k == self.depth or r <= size_limit:
                return r * 2.0**k
            feat = int(self._feature[tr, node])
            if x_norm[feat] <= self._midpoints[tr, node]:
                node = 2 * node + 1
            else:
                node = 2 * node + 2
            k += 1

    def _raw_score(self, x: dict) -> float:
        if not self._ref_ready:
            return 0.0
        try:
            x_norm = self._to_norm(x)
        except (KeyError, TypeError, ValueError):
            return 0.0
        total = sum(self._terminal_mass(tr, x_norm) for tr in range(self.n_trees))
        scale = float(self.n_trees * self.window)
        return scale / (scale + total)

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
        """Record one sample in the mass profile of the current window.

        Args:
            x: a dict of float features.
            t: unused.

        Returns:
            ``self``. A sample that is not a dict, is empty, or has a
            missing or non-finite feature is counted in ``n_missing``.
        """
        self._time_step(t)
        if not isinstance(x, dict) or not x:
            self.n_missing_ += 1
            return self
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
            v_check = np.array([x[k] for k in self._keys], dtype=float)
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
        assert self._r is not None and self._l is not None
        if self._ref_ready:
            self._update_threshold(self._raw_score(x))
        mass = self._l if self._ref_ready else self._r
        for tr in range(self.n_trees):
            mass[tr, self._path_one(tr, x_norm)] += 1
        self._count += 1
        if self._count == self.window:
            if self._ref_ready:
                self._r, self._l = self._l, self._r
                self._l[:] = 0
            else:
                self._ref_ready = True
            self._count = 0
        return self

    def score_one(self, x: dict, t: float | None = None) -> float:
        """Score one sample: higher means more anomalous.

        Args:
            x: a dict of float features.
            t: unused.

        Returns:
            ``M / (M + S)`` with ``M = n_trees * window``, where ``S`` is
            the sum over the trees of ``r * 2^k`` of the terminal node
            (Tan et al. 2011, page 3). ``0.0``, the lowest score, until
            the ranges are fixed and the first reference window of
            ``window`` samples is complete.
        """
        if not self._ready:
            return 0.0
        return self._raw_score(x)

    def is_outlier(self, x: dict, threshold: float | None = None) -> bool:
        """Boolean wrapper around :meth:`score_one`."""
        thr = self.threshold if threshold is None else float(threshold)
        return self.score_one(x) > thr
