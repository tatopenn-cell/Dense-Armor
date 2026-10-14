"""Exact k-nearest-neighbour classifiers and regressors.

Both estimators keep a sliding window of the ``window_size`` most
recent samples. A query is answered by an exact scan of the window
(``LazySearch``): the pairwise distances are computed on the fly,
the ``n_neighbors`` closest samples are selected, and the answer is
the majority vote (classifier) or the mean of the targets
(regressor). The distance is the Minkowski distance with parameter
``p``, or an optional learned metric exposing
``distance(x, x2) -> float`` (see :class:`MetricLearner`).

The definition of the k-nearest-neighbour rule is standard and is
not the subject of a paper in this folder; the docstrings state it
in words, with no equation number.

Cost per query is ``O(window_size * d)`` for the distances plus
``O(window_size log window_size)`` for the selection. The state is
a deque of ``window_size`` samples of ``d`` features; removing the
oldest sample (``forget``) is exact, no re-computation.
"""

from collections import deque
from typing import Any

import numpy as np

from dense_armor.roles import Classifier, Regressor


class LazySearch:
    """Sliding window of recent samples with an exact k-NN scan.

    Args:
        window_size: maximum number of samples kept, strictly
            positive.
        n_neighbors: number of neighbours used by a query, strictly
            positive and not larger than ``window_size``.
        p: Minkowski parameter, ``p >= 1``.
        metric: optional object with ``distance(x, x2) -> float``.
            When given, ``p`` is ignored.

    Raises:
        ValueError: on a non-positive ``window_size`` or
            ``n_neighbors``, an ``n_neighbors`` larger than
            ``window_size``, or a ``p`` smaller than 1.
    """

    def __init__(
        self,
        window_size: int = 100,
        n_neighbors: int = 5,
        p: float = 2.0,
        metric: Any = None,
    ) -> None:
        if window_size < 1:
            raise ValueError(f"window_size must be >= 1, got {window_size}")
        if n_neighbors < 1:
            raise ValueError(f"n_neighbors must be >= 1, got {n_neighbors}")
        if n_neighbors > window_size:
            raise ValueError(
                f"n_neighbors ({n_neighbors}) cannot exceed window_size ({window_size})"
            )
        if metric is None and p < 1.0:
            raise ValueError(f"p must be >= 1, got {p}")
        self.window_size = int(window_size)
        self.n_neighbors = int(n_neighbors)
        self.p = float(p)
        self.metric = metric
        self._keys: list | None = None
        self._X: deque = deque(maxlen=window_size)
        self._y: deque = deque(maxlen=window_size)

    def __len__(self) -> int:
        return len(self._X)

    def _resolve_keys(self, x: dict) -> list:
        if self._keys is not None:
            return self._keys
        if isinstance(x, dict):
            self._keys = sorted(x, key=str)
        return self._keys or []

    def _to_vec(self, x: dict) -> np.ndarray:
        keys = self._resolve_keys(x)
        return np.array([float(x.get(k, 0.0)) for k in keys], dtype=float)

    def add(self, x: dict, y: Any) -> None:
        """Add one sample. If the window is full the oldest is dropped."""
        keys = self._resolve_keys(x)
        vec = np.array([float(x.get(k, 0.0)) for k in keys], dtype=float)
        self._X.append(vec)
        self._y.append(y)

    def forget(self) -> Any | None:
        """Drop the oldest sample and return its label, or None if empty."""
        if not self._X:
            return None
        y = self._y.popleft()
        self._X.popleft()
        return y

    def _distance_to_window(self, xv: np.ndarray) -> np.ndarray:
        if not self._X:
            return np.zeros(0)
        W = np.array(self._X, dtype=float)
        if self.metric is not None:
            keys = self._keys or []
            d = np.empty(len(W))
            for i in range(len(W)):
                xi = {k: float(W[i, j]) for j, k in enumerate(keys)}
                xq = {k: float(xv[j]) for j, k in enumerate(keys)}
                d[i] = float(self.metric.distance(xq, xi))
            return d
        diff = np.abs(W - xv[None, :])
        if self.p == 2.0:
            return np.sqrt((diff * diff).sum(axis=1))
        if self.p == 1.0:
            return diff.sum(axis=1)
        return np.power(np.power(diff, self.p).sum(axis=1), 1.0 / self.p)

    def neighbours(self, x: dict) -> tuple[list, list]:
        """Return the labels and distances of the k nearest samples.

        The window is scanned in full. Ties at the k-th distance are
        broken by the order in the window.
        """
        if not self._X:
            return [], []
        xv = self._to_vec(x)
        d = self._distance_to_window(xv)
        k = min(self.n_neighbors, len(d))
        idx = np.argsort(d, kind="stable")[:k]
        ys = [self._y[int(i)] for i in idx]
        ds = [float(d[int(i)]) for i in idx]
        return ys, ds


class KNNClassifier(Classifier):
    """k-nearest-neighbour classifier on a sliding window.

    A query is classified by the majority vote of its ``n_neighbors``
    nearest window samples. With ``weights="distance"`` each neighbour
    votes with weight ``1 / max(distance, eps)``; with
    ``weights="uniform"`` every neighbour votes with weight 1. The
    distance is the Minkowski distance with parameter ``p``, or the
    distance of an optional learned metric.

    Args:
        window_size: sliding window length. Default 100.
        n_neighbors: number of neighbours per query. Default 5.
        p: Minkowski parameter. Default 2.0.
        weights: ``"uniform"`` or ``"distance"``. Default
            ``"uniform"``.
        metric: optional learned metric with ``distance``. When set,
            ``p`` is ignored.
        eps: denominator floor for the distance weighting.

    Raises:
        ValueError: on any invalid parameter.

    Examples:
        >>> from dense_armor.utility.neighbors import KNNClassifier
        >>> clf = KNNClassifier(window_size=20, n_neighbors=3)
        >>> for _ in range(10):
        ...     _ = clf.learn_one({"a": 0.0, "b": 0.0}, 0)
        ...     _ = clf.learn_one({"a": 1.0, "b": 1.0}, 1)
        >>> clf.predict_one({"a": 0.05, "b": 0.05})
        0
        >>> clf.predict_one({"a": 0.95, "b": 0.95})
        1
    """

    budget_s = 0.001
    memory_class = "O(window_size * d)"

    def __init__(
        self,
        window_size: int = 100,
        n_neighbors: int = 5,
        p: float = 2.0,
        weights: str = "uniform",
        metric: Any = None,
        eps: float = 1e-9,
    ) -> None:
        if weights not in ("uniform", "distance"):
            raise ValueError(
                f"weights must be 'uniform' or 'distance', got {weights!r}"
            )
        if eps <= 0.0:
            raise ValueError(f"eps must be > 0, got {eps}")
        self.window_size = int(window_size)
        self.n_neighbors = int(n_neighbors)
        self.p = float(p)
        self.weights = weights
        self.metric = metric
        self.eps = float(eps)
        self._search = LazySearch(
            window_size=window_size,
            n_neighbors=n_neighbors,
            p=p,
            metric=metric,
        )
        self._seen_classes: set = set()
        self.n_missing_: int = 0

    @property
    def n_missing(self) -> int:
        return self.n_missing_

    @property
    def _multiclass(self) -> bool:
        return True

    def _unit_test_skips(self) -> set:
        """No pure form is provided: the state is not a fixed-size array.

        The state of these estimators is a dict of counters (Bayes),
        a sliding window (neighbors), or N independent models
        (multiclass, multioutput). None of them fits the
        ``init_state`` / ``step(state, v)`` shape and none would pass
        ``jax.lax.scan``. The two checks are skipped explicitly.
        """
        return {"check_step_is_pure", "check_step_is_jittable"}

    def learn_one(self, x: dict, y: Any, t: float | None = None) -> "KNNClassifier":
        """Add one sample with label ``y``."""
        self._time_step(t)
        if not isinstance(x, dict) or not x:
            self.n_missing_ += 1
            return self
        for v in x.values():
            try:
                fv = float(v)
            except (TypeError, ValueError):
                self.n_missing_ += 1
                return self
            if not np.isfinite(fv):
                self.n_missing_ += 1
                return self
        self._search.add(x, y)
        self._seen_classes.add(y)
        return self

    def predict_proba_one(self, x: dict, t: float | None = None) -> dict:
        """Distance or uniform weighted vote of the neighbours."""
        self._time_step(t)
        if not isinstance(x, dict) or not x or len(self._search) == 0:
            return {}
        for v in x.values():
            try:
                fv = float(v)
            except (TypeError, ValueError):
                return {}
            if not np.isfinite(fv):
                return {}
        ys, ds = self._search.neighbours(x)
        if not ys:
            return {}
        if self.weights == "uniform":
            w = np.ones(len(ys))
        else:
            w = 1.0 / np.maximum(np.array(ds), self.eps)
        out: dict = {}
        for yi, wi in zip(ys, w):
            out[yi] = out.get(yi, 0.0) + float(wi)
        s = sum(out.values())
        return {k: v / s for k, v in out.items()}

    @classmethod
    def _unit_test_params(cls):
        yield {}


class KNNRegressor(Regressor):
    """k-nearest-neighbour regressor on a sliding window.

    A query is predicted by the (optionally distance-weighted) mean
    of the targets of its ``n_neighbors`` nearest window samples.

    Args:
        window_size: sliding window length. Default 100.
        n_neighbors: number of neighbours per query. Default 5.
        p: Minkowski parameter. Default 2.0.
        weights: ``"uniform"`` or ``"distance"``. Default
            ``"uniform"``.
        metric: optional learned metric with ``distance``.
        eps: denominator floor for the distance weighting.

    Raises:
        ValueError: on any invalid parameter.

    Examples:
        >>> from dense_armor.utility.neighbors import KNNRegressor
        >>> reg = KNNRegressor(window_size=20, n_neighbors=3)
        >>> for i in range(20):
        ...     _ = reg.learn_one({"x": float(i)}, float(i) * 2.0)
        >>> round(reg.predict_one({"x": 10.1}), 2)
        20.0
    """

    budget_s = 0.001
    memory_class = "O(window_size * d)"

    def __init__(
        self,
        window_size: int = 100,
        n_neighbors: int = 5,
        p: float = 2.0,
        weights: str = "uniform",
        metric: Any = None,
        eps: float = 1e-9,
    ) -> None:
        if weights not in ("uniform", "distance"):
            raise ValueError(
                f"weights must be 'uniform' or 'distance', got {weights!r}"
            )
        if eps <= 0.0:
            raise ValueError(f"eps must be > 0, got {eps}")
        self.window_size = int(window_size)
        self.n_neighbors = int(n_neighbors)
        self.p = float(p)
        self.weights = weights
        self.metric = metric
        self.eps = float(eps)
        self._search = LazySearch(
            window_size=window_size,
            n_neighbors=n_neighbors,
            p=p,
            metric=metric,
        )
        self.n_missing_: int = 0

    @property
    def n_missing(self) -> int:
        return self.n_missing_

    def _unit_test_skips(self) -> set:
        """No pure form is provided: the state is not a fixed-size array.

        The state of these estimators is a dict of counters (Bayes),
        a sliding window (neighbors), or N independent models
        (multiclass, multioutput). None of them fits the
        ``init_state`` / ``step(state, v)`` shape and none would pass
        ``jax.lax.scan``. The two checks are skipped explicitly.
        """
        return {"check_step_is_pure", "check_step_is_jittable"}

    def learn_one(self, x: dict, y: Any, t: float | None = None) -> "KNNRegressor":
        """Add one sample with target ``y``."""
        self._time_step(t)
        if not isinstance(x, dict) or not x:
            self.n_missing_ += 1
            return self
        for v in x.values():
            try:
                fv = float(v)
            except (TypeError, ValueError):
                self.n_missing_ += 1
                return self
            if not np.isfinite(fv):
                self.n_missing_ += 1
                return self
        try:
            fv = float(y)
        except (TypeError, ValueError):
            self.n_missing_ += 1
            return self
        if not np.isfinite(fv):
            self.n_missing_ += 1
            return self
        self._search.add(x, y)
        return self

    def predict_one(
        self,
        x: dict,
        t: float | None = None,
        return_std: bool = False,
    ) -> Any:
        """(Weighted) mean of the targets of the k nearest neighbours."""
        self._time_step(t)
        if not isinstance(x, dict) or not x or len(self._search) == 0:
            return 0.0
        for v in x.values():
            try:
                fv = float(v)
            except (TypeError, ValueError):
                return 0.0
            if not np.isfinite(fv):
                return 0.0
        ys, ds = self._search.neighbours(x)
        if not ys:
            return 0.0
        y_arr = np.asarray([float(yv) for yv in ys], dtype=float)
        if self.weights == "uniform":
            out = float(y_arr.mean())
        else:
            w = 1.0 / np.maximum(np.array(ds), self.eps)
            out = float((y_arr * w).sum() / w.sum())
        if return_std:
            var = float(y_arr.var())
            return out, float(np.sqrt(max(var, 0.0)))
        return out

    @classmethod
    def _unit_test_params(cls):
        yield {}
