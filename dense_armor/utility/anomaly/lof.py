"""Local Outlier Factor (LOF) over a sliding window of the recent stream.

LOF assigns each sample a degree of being an outlier, based on how
isolated the sample is with respect to its local neighborhood. The
score of an inlier is about 1; the score of a local outlier is
greater than 1. It is "local" in that the comparison is against the
neighbors of the sample, not against the whole dataset, so an outlier
near a dense cluster is caught even when its distance to the cluster
is small.

The implementation follows the definitions of Breunig et al. (2000),
Section 4:

    Definition 3   k-distance(p)
    Definition 4   N_k(p), the k-distance neighborhood of p
    Definition 5   reach-dist_k(p, o), the reachability distance
    Definition 6   lrd_k(p), the local reachability density
    Definition 7   LOF_k(p), the local outlier factor

The reference set D is a sliding window of the ``window`` most recent
samples. ``learn_one(x)`` appends x to the window and drops the oldest
sample when the window is full. ``score_one(x)`` returns the LOF of x
against the current window. The neighborhood search is exact: the
pairwise distances inside the window are computed on the fly, and the
cost per score is O(window^2 * d) in the window size and the number
of features d. The incremental insertion and
deletion of Pokrajac, Lazarevic and Latecki (CIDM 2007) is not
implemented: the paper is not in this folder, and recomputing the
neighborhoods touched by the added and the removed sample would not
change the asymptotic cost measured here.

Degenerate cases:

- A window with fewer than ``n_neighbors + 1`` samples: the LOF is
  not defined. ``score_one`` returns 1.0.
- A malformed or non-finite input: ``score_one`` returns 0.0, like the
  other detectors of the library, and ``learn_one`` counts it in
  ``n_missing`` without touching the window.
- Duplicates: the reachability distances in a neighborhood of
  duplicates are 0 and the density of Definition 6 is infinite. As
  a general-purpose machine-learning library does, the mean reachability distance gets a term
  1e-10, so the density is at most 1e10 and every score is finite.
  A window of identical samples gives 1.0 to the same point and a
  value of the order of 1e10 times the distance to a far point.
  This is the one deviation from the paper, which assumes no
  duplicates (page 4 of the PDF).

References:
    Breunig, M. M., Kriegel, H.-P., Ng, R. T., Sander, J. (2000).
    LOF: Identifying Density-Based Local Outliers. In Proceedings of
    the 2000 ACM SIGMOD International Conference on Management of
    Data, pages 93-104. Definitions 3 to 7 in Section 4 (pages 3-4 of
    the PDF).
"""

from collections import deque

import numpy as np

from dense_armor.roles import AnomalyDetector

_EPS = 1e-10


class LocalOutlierFactor(AnomalyDetector):
    """Local Outlier Factor over a sliding window of the last samples.

    ``learn_one(x)`` appends x to the window of the ``window`` most
    recent samples. ``score_one(x)`` returns the LOF of x against the
    current window, using the definitions of Breunig et al. (2000).

    Args:
        n_neighbors: number of neighbors, k in the LOF definitions.
            Also called MinPts in the paper. Must be at least 1.
        window: number of recent samples kept as the reference set.
            Must be at least ``n_neighbors + 1``.
        threshold: LOF value above which a sample is flagged as
            anomalous. Default 1.5 (the paper suggests a rule of
            thumb around 1; the exact value depends on the data).
        seed: unused, kept for interface stability with the other
            detectors of the library.
        feature_keys: optional list of dict keys, one per channel.
            ``None`` takes the sorted keys of the first dict seen.

    Raises:
        ValueError: on a non-positive ``n_neighbors``, a ``window``
            smaller than ``n_neighbors + 1``, or a non-positive
            ``threshold``.

    Cost and memory. ``score_one`` computes the pairwise distances
    of the whole window, O(window^2 * d) for the distance matrix,
    plus O(window * n_neighbors * d) for the two local densities.
    ``learn_one`` is O(1): it appends to the deque and drops the
    oldest when the window is full. The state is a deque of
    ``window`` vectors of ``d`` floats, so it is constant in the
    stream length n and linear in window and in d. During one
    ``score_one`` the pairwise difference array is transient and
    costs window^2 * d floats: peak 6.6 MB at window=300, d=4 and
    168 MB at window=1000, d=10 (tracemalloc). Measured on
    3000 samples (Python 3.13, one core), p50 / p99 / max of
    one ``score_one`` plus one ``learn_one``:

        window=100, n_neighbors=20:  p50 1 120 us   p99 2 130 us   max 3 747 us
        window=200, n_neighbors=20:  p50 3 768 us   p99 6 570 us   max 8 160 us
        window=300, n_neighbors=20:  p50 7 540 us   p99 12 466 us  max 15 041 us

    The declared budget_s = 0.01 s (10 000 us) holds at window up
    to about 200 at n_neighbors=20. At window=300 the p99 exceeds
    the budget. A caller who needs a bigger window is over the
    declared budget and should either lower the window or raise
    the budget.

    On the benchmark streams of ``benchmarks/anomaly_streaming.py``
    (SyntheticArm payload fault, joints 0-3, 200 samples, 100 healthy
    then 100 faulty; DriftStream sudden drift, 4 features, 2000
    samples, one step at half), at the default ``n_neighbors=20``,
    ``window=100``, ``threshold=1.5``:

        stream   protocol          ROC-AUC   FA@thr   ms/sample
        arm      learn-on-healthy    1.000    0.000       1.675
        arm      learn-on-all        0.778    0.000       0.855
        drift    learn-on-healthy    0.997    0.007       2.277
        drift    learn-on-all        0.519    0.007       1.153

    The large gap between learn-on-healthy and learn-on-all is the
    expected behaviour of LOF on a stream where half the samples are
    anomalies: the window is filled with the anomalous samples
    during the second half, the local density around the fault is no
    longer anomalous, and the score drops. The detector is meant to
    be trained on the healthy phase and used afterwards, which is
    the deployment the library targets.

    Protection. The same detector, wrapped in AnomalyGate + Protected
    on the arm residuals, keeps the offset of the least-squares model
    at err_prot = 0.0015 against err_bare = 3.2886 (bare model), with
    100 of the 100 fault samples flagged. See
    test/protect/test_protected_anomaly_stream.py.

    No pure form. The exact k-NN search inside a sliding window is
    not naturally expressible as a pure function of a small state
    that fits ``jax.lax.scan``: the state is the window itself, and
    the neighborhood structure changes at every sample. A pure form
    is not provided.

    Examples:
        >>> import numpy as np
        >>> from dense_armor.utility.anomaly.lof import LocalOutlierFactor
        >>> rng = np.random.default_rng(0)
        >>> model = LocalOutlierFactor(n_neighbors=10, window=100)
        >>> for _ in range(200):
        ...     x = rng.standard_normal(2) * 0.1
        ...     _ = model.learn_one({"a": float(x[0]), "b": float(x[1])})
        >>> inlier = model.score_one({"a": 0.0, "b": 0.0})
        >>> outlier = model.score_one({"a": 5.0, "b": 5.0})
        >>> bool(outlier > inlier)
        True

    References:
        Breunig, M. M., Kriegel, H.-P., Ng, R. T., Sander, J. (2000).
        LOF: Identifying Density-Based Local Outliers. SIGMOD 2000.
        Definitions 3, 4, 5, 6, 7 in Section 4.
    """

    budget_s = 0.01
    memory_class = "O(window)"

    def __init__(
        self,
        n_neighbors: int = 20,
        window: int = 100,
        threshold: float = 1.5,
        seed: int | None = None,
        feature_keys: list | None = None,
    ) -> None:
        if n_neighbors < 1:
            raise ValueError(f"n_neighbors must be >= 1, got {n_neighbors}")
        if window < n_neighbors + 1:
            raise ValueError(
                f"window must be >= n_neighbors + 1, "
                f"got window={window}, n_neighbors={n_neighbors}"
            )
        if threshold <= 0:
            raise ValueError(f"threshold must be > 0, got {threshold}")
        self.n_neighbors = int(n_neighbors)
        self.window = int(window)
        self.threshold = float(threshold)
        self.seed = seed
        self.feature_keys = feature_keys
        self._keys: list | None = None
        self._buf: deque = deque(maxlen=window)
        self.n_missing_: int = 0

    @property
    def n_missing(self) -> int:
        return self.n_missing_

    def _unit_test_skips(self) -> set:
        return set()

    def _resolve_keys(self, x) -> list:
        if self._keys is not None:
            return self._keys
        if self.feature_keys is not None:
            self._keys = list(self.feature_keys)
        else:
            self._keys = sorted(x)
        return self._keys

    def _to_vec(self, x) -> np.ndarray:
        keys = self._resolve_keys(x)
        return np.array([float(x[k]) for k in keys], dtype=float)

    def learn_one(self, x: dict, t: float | None = None) -> "LocalOutlierFactor":
        """Append one sample to the window.

        Args:
            x: dict with one entry per feature.
            t: optional timestamp, currently unused.

        Returns:
            ``self``.
        """
        self._time_step(t)
        if not isinstance(x, dict) or not x:
            self.n_missing_ += 1
            return self
        try:
            v = self._to_vec(x)
        except (KeyError, TypeError, ValueError):
            self.n_missing_ += 1
            return self
        if not np.isfinite(v).all():
            self.n_missing_ += 1
            return self
        self._buf.append(v)
        return self

    def score_one(self, x: dict, t: float | None = None) -> float:
        """LOF of x against the current window.

        Returns 1.0 on a window with fewer than ``n_neighbors + 1``
        samples and 0.0 on a non-finite or malformed input. Duplicates
        give finite values (see the module docstring).

        Args:
            x: dict with one entry per feature.
            t: optional timestamp, currently unused.

        Returns:
            The LOF value, or the degenerate values above.
        """
        self._time_step(t)
        if len(self._buf) < self.n_neighbors + 1:
            return 1.0
        try:
            v = self._to_vec(x)
        except (KeyError, TypeError, ValueError):
            return 0.0
        if not np.isfinite(v).all():
            return 0.0
        return self._lof(v)

    def _lof(self, x: np.ndarray) -> float:
        W = np.array(self._buf, dtype=float)
        N = len(W)
        k = self.n_neighbors

        diff = W[:, None, :] - W[None, :, :]
        D = np.sqrt((diff * diff).sum(axis=-1))

        D_sorted = np.sort(D, axis=1)
        k_dist_W = D_sorted[:, k]

        lrd_W = np.empty(N, dtype=float)
        for i in range(N):
            mask_i = D[i] <= k_dist_W[i]
            mask_i[i] = False
            nbrs_i = np.where(mask_i)[0]
            reach_i = np.maximum(k_dist_W[nbrs_i], D[i, nbrs_i])
            lrd_W[i] = nbrs_i.size / (float(reach_i.sum()) + nbrs_i.size * _EPS)

        dx = np.sqrt(((W - x) ** 2).sum(axis=1))
        kd_x = float(np.sort(dx)[k - 1])
        nbrs_x = np.where(dx <= kd_x)[0]
        reach_x = np.maximum(k_dist_W[nbrs_x], dx[nbrs_x])
        lrd_x = nbrs_x.size / (float(reach_x.sum()) + nbrs_x.size * _EPS)
        return float(lrd_W[nbrs_x].mean() / lrd_x)
