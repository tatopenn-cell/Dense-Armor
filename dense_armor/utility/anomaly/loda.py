"""LODA: Lightweight Online Detector of Anomalies.

LODA projects the data onto many one-dimensional random directions
and, for each projection, keeps a histogram over a sliding window.
The score of a point is the average negative log-likelihood of its bin
count across the projections: rare bins flag anomalies.

This module follows Lou et al. (2024), fSEAD, section on Loda: random
projection of the input, one histogram per projection, and the score
``-mean(log2(c / W))`` where ``c`` is the count in the bin and ``W`` is
the window size. The original paper is Pevny (2016).

Details not fixed by Lou et al. (2024), stated explicitly:

- Projection: sparse random Gaussian vector with ``k`` non-zero
  entries, chosen uniformly without replacement; ``k = max(1, d // 2)``
  by default, matching LODA's sparse projection. The vector is
  L2-normalised.
- Histogram: fixed bin edges in ``[lo, hi]``; when ``feature_ranges``
  is not given, the ranges are learned online from the first
  ``range_init`` samples (min and max per feature, with a small
  margin); until then the score is 0.
- Score: ``mean over projections of -log2((c + 1) / (W + 1))``;
  ``+1`` smooths empty bins so the score stays finite.

References
----------
Lou, B., Boland, D., Leong, P. H. W. (2024). fSEAD: a Composable
    FPGA-based Streaming Ensemble Anomaly Detection Library.
    arXiv:2406.05999. Section 2.1 (Streaming Anomaly Detection, Loda).
Pevny, T. (2016). LODA: Lightweight on-line detector of anomalies.
    Machine Learning 102(2), 275-304.
"""

from collections.abc import Sequence

import numpy as np

from dense_armor.roles import AnomalyDetector


class LODA(AnomalyDetector):
    """Lightweight Online Detector of Anomalies.

    Args:
        n_projections: number of random one-dimensional projections.
        n_bins: number of histogram bins per projection.
        window: size of the sliding window.
        sparsity: number of non-zero entries per projection. ``None``
            uses ``max(1, d // 2)`` where ``d`` is the number of
            features.
        feature_ranges: ``[(lo, hi), ...]`` per feature. ``None``
            learns the ranges from the first ``range_init`` samples.
        range_init: number of samples used to learn the working ranges
            when ``feature_ranges`` is ``None``. Default 20.
        feature_keys: names of the dict keys, in order. ``None`` uses
            the sorted keys of the first sample seen.
        threshold: score above which ``is_outlier`` returns True. The
            score lives in ``[0, ~log2(window)]``; default ``4.0``.
        seed: seed of the random generator.
        eps: numerical guard.

    Raises:
        ValueError: on a non-positive parameter.

    Examples:
        >>> import numpy as np
        >>> from dense_armor.utility.anomaly.loda import LODA
        >>> ds = LODA(n_projections=50, n_bins=10, window=200,
        ...           range_init=10, seed=0)
        >>> rng = np.random.default_rng(0)
        >>> for _ in range(300):
        ...     _ = ds.learn_one({
        ...         "a": float(rng.normal(0.5, 0.05)),
        ...         "b": float(rng.normal(0.5, 0.05)),
        ...     })
        >>> s = ds.score_one({"a": 0.5, "b": 0.5})
        >>> isinstance(s, float)
        True
    """

    budget_s = 0.002
    memory_class = "O(1)"

    def __init__(
        self,
        n_projections: int = 100,
        n_bins: int = 10,
        window: int = 256,
        sparsity: int | None = None,
        feature_ranges: Sequence[tuple[float, float]] | None = None,
        range_init: int = 20,
        feature_keys: Sequence[str] | None = None,
        threshold: float = 4.0,
        seed: int | None = None,
        eps: float = 1e-9,
    ) -> None:
        if n_projections < 1:
            raise ValueError(f"n_projections must be >= 1, got {n_projections}")
        if n_bins < 2:
            raise ValueError(f"n_bins must be >= 2, got {n_bins}")
        if window < 1:
            raise ValueError(f"window must be >= 1, got {window}")
        if sparsity is not None and sparsity < 1:
            raise ValueError(f"sparsity must be >= 1, got {sparsity}")
        if range_init < 1:
            raise ValueError(f"range_init must be >= 1, got {range_init}")
        self.n_projections = int(n_projections)
        self.n_bins = int(n_bins)
        self.window = int(window)
        self.sparsity = sparsity
        self.feature_ranges = feature_ranges
        self.range_init = int(range_init)
        self.feature_keys = feature_keys
        self.threshold = float(threshold)
        self.seed = seed
        self.eps = eps
        self.n_missing_ = 0
        self._ready = False
        self._keys_ready = False
        self._keys: list[str] = []
        self._lo: np.ndarray | None = None
        self._hi: np.ndarray | None = None
        self._W: np.ndarray | None = None
        self._counts: np.ndarray | None = None
        self._buf: np.ndarray | None = None
        self._buf_head = 0
        self._buf_size = 0
        self._rng: np.random.Generator | None = None
        self._range_buf: list[np.ndarray] = []

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
            self._finalise()

    def _finalise(self) -> None:
        assert self._lo is not None and self._hi is not None
        d = len(self._keys)
        self._rng = np.random.default_rng(self.seed)
        k = self.sparsity if self.sparsity is not None else max(1, d // 2)
        k = min(k, d)
        W = np.zeros((self.n_projections, d), dtype=float)
        for j in range(self.n_projections):
            idx = self._rng.choice(d, size=k, replace=False)
            w = self._rng.normal(0.0, 1.0, size=k)
            W[j, idx] = w
            nrm = float(np.linalg.norm(W[j]))
            if nrm > self.eps:
                W[j] /= nrm
        self._W = W
        self._counts = np.zeros((self.n_projections, self.n_bins), dtype=np.int64)
        self._buf = np.zeros((self.window, self.n_projections), dtype=np.intp)
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

    def _bin_index(self, proj: np.ndarray) -> np.ndarray:
        u = np.clip(proj, 0.0, 1.0)
        idx = np.floor(u * self.n_bins).astype(np.intp)
        return np.clip(idx, 0, self.n_bins - 1)

    def learn_one(self, x: dict, t: float | None = None) -> "LODA":
        """Learn one sample and slide the window forward.

        Args:
            x: a dict of float features.
            t: unused.

        Returns:
            ``self``.
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
                    self._finalise()
                return self
        try:
            v = np.array([float(x[k]) for k in self._keys], dtype=float)
        except (KeyError, TypeError, ValueError):
            self.n_missing_ += 1
            return self
        if not np.isfinite(v).all():
            self.n_missing_ += 1
            return self
        try:
            x_norm = self._to_norm(x)
        except (KeyError, TypeError, ValueError):
            self.n_missing_ += 1
            return self
        assert self._W is not None
        assert self._counts is not None
        assert self._buf is not None
        proj = self._W @ x_norm
        b = self._bin_index(proj)
        if self._buf_size == self.window:
            old = self._buf[self._buf_head]
            for j in range(self.n_projections):
                self._counts[j, old[j]] -= 1
        for j in range(self.n_projections):
            self._counts[j, b[j]] += 1
        self._buf[self._buf_head] = b
        self._buf_head = (self._buf_head + 1) % self.window
        self._buf_size = min(self._buf_size + 1, self.window)
        return self

    def score_one(self, x: dict, t: float | None = None) -> float:
        """Score one sample: higher means more anomalous.

        Args:
            x: a dict of float features.
            t: unused.

        Returns:
            The mean over projections of
            ``-log2((c_bin + 1) / (W + 1))``. ``0.0`` while the ranges
            are being learned or the model has not seen any sample yet.
        """
        if not self._ready:
            return 0.0
        try:
            x_norm = self._to_norm(x)
        except (KeyError, TypeError, ValueError):
            return 0.0
        assert self._W is not None
        assert self._counts is not None
        n = self._buf_size
        if n == 0:
            return 0.0
        proj = self._W @ x_norm
        b = self._bin_index(proj)
        scores = np.empty(self.n_projections, dtype=float)
        for j in range(self.n_projections):
            c = int(self._counts[j, b[j]])
            scores[j] = -np.log2((c + 1.0) / (n + 1.0))
        return float(scores.mean())

    def is_outlier(self, x: dict, threshold: float | None = None) -> bool:
        """Boolean wrapper around :meth:`score_one`."""
        thr = self.threshold if threshold is None else float(threshold)
        return self.score_one(x) > thr
