"""
Online metric learning: OASIS, LEGO and POLA, as river-compatible estimators.

Needs the optional dependency: pip install dense-armor[river].
"""
from __future__ import annotations

import random
from collections import deque

import numpy as np

try:
    from river import base
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "dense_armor.utility.metric_learning needs river: pip install dense-armor[river]"
    ) from exc


class MetricLearner(base.Base):
    """Base class for online metric learners.

    A metric learner is trained from pairs or triplets and exposes a
    `distance(x, x2)` method. Unlike river's `Classifier`, `Regressor`, and
    `Transformer`, it learns from relative information (two or three samples
    at a time) and does not predict labels or transform features directly.

    Subclasses implement `learn_triplet`, `learn_pair`, and `distance`.
    Features are dicts; the first call fixes the feature order (sorted keys),
    and any feature not seen at that time counts as 0 in later calls.
    """

    def _reset_features(self) -> None:
        self._feature_names = None
        self._feature_index = None

    def _to_array(self, x: dict) -> np.ndarray:
        if self._feature_names is None:
            self._feature_names = sorted(x.keys(), key=str)
            self._feature_index = {k: i for i, k in enumerate(self._feature_names)}
        out = np.zeros(len(self._feature_names))
        idx = self._feature_index
        for k, v in x.items():
            i = idx.get(k)
            if i is not None:
                out[i] = float(v)
        return out

    def learn_triplet(self, x: dict, x_pos: dict, x_neg: dict) -> None:
        """Update from a triplet (query, positive, negative)."""
        raise NotImplementedError

    def learn_pair(self, x: dict, x2: dict, y) -> None:
        """Update from a pair with a target (distance, or +/-1 label)."""
        raise NotImplementedError

    def distance(self, x: dict, x2: dict) -> float:
        """Return the learned distance between two samples."""
        raise NotImplementedError


class OASIS(MetricLearner):
    """Online metric learning via passive-aggressive bilinear similarity.

    Learns a bilinear similarity `S(x, y) = x^T W y` from triplets
    `(query, positive, negative)` such that `S(x, x_pos) > S(x, x_neg) + 1`.
    `W` is not constrained to be symmetric or positive semi-definite; the
    induced distance uses `M = W^T W`, which is always positive semi-definite.

    Parameters
    ----------
    C
        Aggressiveness. The passive-aggressive step is `min(C, loss / ‖V‖²)`.
        The paper recommends a small value; `0.01` to `0.1` works well in
        practice on standard benchmarks.

    Examples
    --------
    >>> from dense_armor.utility.metric_learning import OASIS
    >>> oasis = OASIS(C=0.1)
    >>> x     = {"a": 1.0, "b": 0.0}
    >>> x_pos = {"a": 0.9, "b": 0.1}
    >>> x_neg = {"a": 0.1, "b": 0.9}
    >>> _ = oasis.learn_triplet(x, x_pos, x_neg)
    >>> oasis.distance(x, x_pos) < oasis.distance(x, x_neg)
    True

    References
    ----------
    [^1]: Chechik, G., Sharma, V., Shalit, U. and Bengio, S., 2009. An online
    algorithm for large scale image similarity learning. In NIPS.
    """

    def __init__(self, C: float = 0.01):
        self.C = C
        self.W = None
        self._reset_features()

    def _ensure_W(self, n: int) -> None:
        if self.W is None:
            self.W = np.identity(n)

    def learn_triplet(self, x: dict, x_pos: dict, x_neg: dict) -> None:
        xa = self._to_array(x)
        xp = self._to_array(x_pos)
        xn = self._to_array(x_neg)
        self._ensure_W(len(xa))

        loss = 1.0 - xa @ self.W @ xp + xa @ self.W @ xn
        if loss <= 0.0:
            return
        V = np.outer(xa, xp - xn)
        tau = min(self.C, loss / (np.sum(V * V) + 1e-12))
        self.W = self.W + tau * V

    def distance(self, x: dict, x2: dict) -> float:
        xa = self._to_array(x)
        ya = self._to_array(x2)
        self._ensure_W(len(xa))
        z = xa - ya
        M = self.W.T @ self.W
        return float(np.sqrt(max(z @ M @ z, 1e-12)))


class LEGO(MetricLearner):
    """Online metric learning via LogDet regularization and exact gradient.

    Learns a Mahalanobis matrix `A` from pairs `(u, v)` with a **target
    squared distance** `y`. Same-class pairs should have small targets,
    different-class pairs large ones; use `percentile_targets` to compute
    the 5th and 95th percentile of the pairwise distances of a training set,
    as in Section 4 of the paper.

    Parameters
    ----------
    eta
        Learning rate of the LogDet gradient step. The paper uses 0.1.

    Examples
    --------
    >>> from dense_armor.utility.metric_learning import LEGO
    >>> lego = LEGO(eta=0.1)
    >>> u = {"a": 1.0, "b": 0.0}
    >>> v = {"a": 0.0, "b": 1.0}
    >>> d_before = lego.distance(u, v)
    >>> _ = lego.learn_pair(u, v, 0.1)
    >>> lego.distance(u, v) <= d_before
    True

    References
    ----------
    [^1]: Jain, P., Kulis, B., Dhillon, I.S. and Grauman, K., 2008. Online
    metric learning and fast similarity search. In NIPS.
    """

    def __init__(self, eta: float = 0.1):
        self.eta = eta
        self.A = None
        self._reset_features()

    def _ensure_A(self, n: int) -> None:
        if self.A is None:
            self.A = np.identity(n)

    def learn_pair(self, x: dict, x2: dict, y: float) -> None:
        u = self._to_array(x)
        v = self._to_array(x2)
        self._ensure_A(len(u))
        z = u - v
        y_hat = float(z @ self.A @ z)
        if y_hat <= 0.0:
            return
        eta = self.eta
        a = eta * float(y) * y_hat - 1.0
        y_bar = (a + np.sqrt(a * a + 4.0 * eta * y_hat * y_hat)) / (2.0 * eta * y_hat)
        Az = self.A @ z
        self.A = self.A - eta * (y_bar - float(y)) * np.outer(Az, Az) / (
            1.0 + eta * (y_bar - float(y)) * y_hat
        )

    def distance(self, x: dict, x2: dict) -> float:
        u = self._to_array(x)
        v = self._to_array(x2)
        self._ensure_A(len(u))
        z = u - v
        return float(np.sqrt(max(z @ self.A @ z, 1e-12)))

    @staticmethod
    def percentile_targets(X, y, low: float = 5.0, high: float = 95.0):
        """5th/95th percentile of same-class / different-class squared distances.

        X is an array-like of shape (n, d), y an array-like of labels.
        Returns (near, far).
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        D2 = np.sum((X[:, None, :] - X[None, :, :]) ** 2, axis=-1)
        same = y[:, None] == y[None, :]
        np.fill_diagonal(same, False)
        off = ~np.eye(len(X), dtype=bool)
        near = float(np.percentile(D2[same], low))
        far = float(np.percentile(D2[~same & off], high))
        return near, far


class POLA(MetricLearner):
    """Online pseudo-metric learning with PSD projection.

    Learns `(A, b)` from labelled pairs `(x, x2, y)` with `y = +1` for similar
    and `y = -1` for dissimilar. After every update, `A` is projected onto the
    positive semi-definite cone and `b` onto `[1, +inf)`.

    Examples
    --------
    >>> from dense_armor.utility.metric_learning import POLA
    >>> pola = POLA()
    >>> a = {"x": 0.0}
    >>> b = {"x": 1.0}
    >>> _ = pola.learn_pair(a, b, +1)
    >>> _ = pola.learn_pair(a, b, -1)
    >>> pola.b >= 1.0
    True

    References
    ----------
    [^1]: Shalev-Shwartz, S., Singer, Y. and Ng, A.Y., 2004. Online and batch
    learning of pseudo-metrics. In ICML.
    """

    def __init__(self, b_init: float = 1.0):
        self.b_init = b_init
        self.A = None
        self.b = b_init
        self._reset_features()

    def _ensure_A(self, n: int) -> None:
        if self.A is None:
            self.A = np.zeros((n, n))

    def learn_pair(self, x: dict, x2: dict, y: int) -> None:
        u = self._to_array(x)
        v = self._to_array(x2)
        self._ensure_A(len(u))
        y = 1 if y > 0 else -1
        z = u - v
        loss = max(0.0, y * (float(z @ self.A @ z) - self.b) + 1.0)
        if loss <= 0.0:
            return
        alpha = loss / (1.0 + float(z @ z) ** 2)
        A_new = self.A - y * alpha * np.outer(z, z)
        b_new = self.b + y * alpha
        if y == 1:
            w, U = np.linalg.eigh(A_new)
            if w[0] < 0:
                A_new = A_new - w[0] * np.outer(U[:, 0], U[:, 0])
            self.A = A_new
            self.b = b_new
        else:
            self.A = A_new
            self.b = max(b_new, 1.0)

    def distance(self, x: dict, x2: dict) -> float:
        u = self._to_array(x)
        v = self._to_array(x2)
        self._ensure_A(len(u))
        z = u - v
        return float(np.sqrt(max(z @ self.A @ z, 1e-12)))


class MetricKNNClassifier(base.Classifier):
    """K-nearest-neighbours classifier that learns its metric online.

    Stores the last `window_size` samples and classifies a new sample by
    majority vote of its `n_neighbors` nearest neighbours under the metric
    learned by `metric_learner`. When `learn_metric=True`, every `learn_one`
    (every `update_every` steps) also feeds the metric learner with a pair
    or triplet built from the current window, so the metric adapts as the
    stream unfolds.

    Parameters
    ----------
    metric_learner
        A `MetricLearner` (OASIS, LEGO or POLA).
    n_neighbors
        Number of nearest neighbours to vote.
    window_size
        Maximum number of stored samples (FIFO).
    learn_metric
        If True, `learn_one` also updates the metric learner. If False, the
        learner is only used as-is (frozen metric).
    seed
        Seed for the internal `random.Random` used to pick window samples.
    update_every
        Update the metric once every `update_every` calls to `learn_one`.

    Examples
    --------
    >>> from dense_armor.utility.metric_learning import MetricKNNClassifier, OASIS
    >>> knn = MetricKNNClassifier(OASIS(C=1e6), n_neighbors=1, learn_metric=False)
    >>> _ = knn.learn_one({"x": 1.0}, "A")
    >>> _ = knn.learn_one({"x": 0.0}, "B")
    >>> knn.predict_one({"x": 0.95})
    'A'
    """

    def __init__(self, metric_learner: MetricLearner, n_neighbors: int = 5,
                 window_size: int = 1000, learn_metric: bool = True,
                 seed: int = 42, update_every: int = 1):
        self.metric_learner = metric_learner
        self.n_neighbors = n_neighbors
        self.window_size = window_size
        self.learn_metric = learn_metric
        self.seed = seed
        self.update_every = update_every
        self._window = deque(maxlen=window_size)
        self.classes = set()
        self._rng = random.Random(seed)
        self._step = 0
        self._lego_targets_fixed = None

    @property
    def _multiclass(self) -> bool:
        return True

    @classmethod
    def _unit_test_params(cls):
        yield {"metric_learner": OASIS(C=0.1)}

    def _euclidean_d2(self, x: dict, xi: dict) -> float:
        keys = sorted(set(x.keys()) | set(xi.keys()), key=str)
        z = np.array([float(x.get(k, 0)) - float(xi.get(k, 0)) for k in keys])
        return float(z @ z)

    def _ensure_lego_targets(self) -> None:
        if self._lego_targets_fixed is not None:
            return
        if len(self._window) < 30:
            return
        items = list(self._window)
        same, diff = [], []
        for _ in range(50):
            i = self._rng.randrange(len(items))
            j = self._rng.randrange(len(items))
            if i == j:
                continue
            xi, yi = items[i]
            xj, yj = items[j]
            d2 = self._euclidean_d2(xi, xj)
            if yi == yj:
                same.append(d2)
            else:
                diff.append(d2)
        if same and diff:
            self._lego_targets_fixed = (
                float(np.percentile(same, 5)),
                float(np.percentile(diff, 95)),
            )

    def _pick_and_learn(self, x: dict, y) -> None:
        same = [(xi, yi) for xi, yi in self._window if yi == y]
        diff = [(xi, yi) for xi, yi in self._window if yi != y]
        if not same or not diff:
            return
        learner = self.metric_learner
        if isinstance(learner, OASIS):
            x_pos = self._rng.choice(same)[0]
            x_neg = self._rng.choice(diff)[0]
            learner.learn_triplet(x, x_pos, x_neg)
        elif isinstance(learner, POLA):
            xi, yi = self._rng.choice(list(self._window))
            learner.learn_pair(x, xi, +1 if yi == y else -1)
        elif isinstance(learner, LEGO):
            self._ensure_lego_targets()
            if self._lego_targets_fixed is None:
                return
            near, far = self._lego_targets_fixed
            xi, yi = self._rng.choice(list(self._window))
            learner.learn_pair(x, xi, near if yi == y else far)

    def learn_one(self, x: dict, y) -> None:
        if self.learn_metric:
            self._step += 1
            if self._step % self.update_every == 0:
                self._pick_and_learn(x, y)
        self._window.append((dict(x), y))
        self.classes.add(y)

    def predict_proba_one(self, x: dict) -> dict:
        if not self._window:
            return {}
        dists = [(self.metric_learner.distance(x, xi), yi) for xi, yi in self._window]
        dists.sort(key=lambda t: t[0])
        k = min(self.n_neighbors, len(dists))
        proba = {c: 0.0 for c in self.classes}
        for _, yi in dists[:k]:
            proba[yi] += 1.0
        total = sum(proba.values())
        return {c: v / total for c, v in proba.items()}