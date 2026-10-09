"""Synthetic binary stream with concept drift.

Follows the taxonomy of Ksieniewicz, P., Zyblewski, P. (2020),
"stream-learn -- open-source Python library for difficult data stream
batch analysis", SoftwareX 11, section 3. Three kinds of drift are
generated, each sample marked with its concept identity:

- ``"sudden"``: the concept is replaced at once at the drift position
  ``d``; this is the limit of the sigmoid
  ``s(p) = 1 / (1 + exp(-spacing * (p - d)))`` of section 3.2.1 for a
  large ``spacing``, so ``spacing`` is not used.
- ``"gradual"``: at each sample the new concept is used with
  probability ``s(p)`` and the old one with ``1 - s(p)`` (section
  3.2.2). Samples are drawn from one of the two, never mixed.
- ``"incremental"``: the concept drifts continuously; the mean moves
  linearly between the two concept means with the same sigmoid weight
  (section 3.2.3). The paper describes this in words; the linear-in-s
  interpolation of the means is the concrete formula used here, and the
  label follows the concept with the larger weight.

Each sample uses the drift nearest to its position ``p`` in ``[0, 1]``,
so a transition is continuous on both sides of its drift.

Each concept is two Gaussian clusters in ``n_features`` dimensions, one
per class. The clusters are placed along a random unit direction with a
fixed separation, and the class labels alternate between concepts: on
even-indexed concepts cluster 0 is class 0, on odd-indexed ones it is
class 1. A classifier trained on the earlier concept therefore becomes
systematically wrong when a new concept starts, which is the point of
the drift.

Examples:
    >>> from dense_armor.utility.datasets import DriftStream
    >>> ds = DriftStream(n_samples=100, n_features=2, n_drifts=1,
    ...                  kind="sudden", spacing=100.0, seed=0)
    >>> X, y = ds.data()
    >>> X.shape, y.shape
    ((100, 2), (100,))
    >>> len(ds.drift_positions)
    1
"""

from collections.abc import Iterator

import numpy as np

from dense_armor.roles import Signal


def _make_concept(
    rng: np.random.Generator,
    n_features: int,
    concept_index: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    d = rng.normal(0.0, 1.0, size=n_features)
    d = d / (np.linalg.norm(d) + 1e-9)
    sep = 6.0
    mu = np.stack([-sep * d / 2.0, +sep * d / 2.0])
    mu = mu + rng.normal(0.0, 0.1, size=mu.shape)
    scale = np.full((2, n_features), 0.5)
    if concept_index % 2 == 0:
        labels = np.array([0.0, 1.0])
    else:
        labels = np.array([1.0, 0.0])
    return mu, scale, labels


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


class DriftStream:
    """Synthetic binary stream with concept drift.

    Args:
        n_samples: number of samples in the stream.
        n_features: number of input dimensions.
        n_drifts: number of concept changes. Zero means stationary.
        kind: one of ``"sudden"``, ``"gradual"``, ``"incremental"``.
        spacing: sharpness of the sigmoid, in inverse position units.
            Higher values give more abrupt transitions.
        seed: seed of the random generator.

    Raises:
        ValueError: on a bad ``kind`` or a non-positive count.

    Examples:
        >>> from dense_armor.utility.datasets import DriftStream
        >>> ds = DriftStream(n_samples=50, n_features=3, n_drifts=2,
        ...                  kind="gradual", spacing=20.0, seed=1)
        >>> X, y = ds.data()
        >>> X.shape
        (50, 3)
        >>> ds.drift_positions
        array([0.33333333, 0.66666667])
    """

    _KINDS = ("sudden", "gradual", "incremental")

    def __init__(
        self,
        n_samples: int,
        n_features: int,
        n_drifts: int,
        kind: str,
        spacing: float = 10.0,
        seed: int | None = 0,
    ) -> None:
        if n_samples < 1:
            raise ValueError(f"n_samples must be >= 1, got {n_samples}")
        if n_features < 1:
            raise ValueError(f"n_features must be >= 1, got {n_features}")
        if n_drifts < 0:
            raise ValueError(f"n_drifts must be >= 0, got {n_drifts}")
        if kind not in self._KINDS:
            raise ValueError(f"kind must be one of {self._KINDS}, got {kind!r}")
        self.n_samples = n_samples
        self.n_features = n_features
        self.n_drifts = n_drifts
        self.kind = kind
        self.spacing = float(spacing)
        self.seed = seed
        rng = np.random.default_rng(seed)
        self._mu = [_make_concept(rng, n_features, j) for j in range(n_drifts + 1)]
        self._scale = self._mu[0][1]
        if n_drifts == 0:
            self.drift_positions = np.zeros(0)
        else:
            self.drift_positions = np.array(
                [(j + 1) / (n_drifts + 1) for j in range(n_drifts)]
            )

    def _segment(self, p: float) -> tuple[int, float]:
        """Nearest drift ``j`` and the weight ``s(p)`` of concept ``j + 1``.

        Using the nearest drift keeps every transition continuous on both
        sides of its position: the weight goes from about 0 to about 1
        across the drift, and the next drift takes over only half way to
        it.
        """
        if self.n_drifts == 0:
            return 0, 0.0
        j = int(np.argmin(np.abs(self.drift_positions - p)))
        w = float(_sigmoid(np.asarray(self.spacing * (p - self.drift_positions[j]))))
        return j, w

    def _sample_one(
        self, rng: np.random.Generator, p: float
    ) -> tuple[np.ndarray, int, int]:
        j, w = self._segment(p)
        nxt = min(j + 1, self.n_drifts)
        k = int(rng.integers(0, 2))
        if self.kind == "sudden":
            past = self.n_drifts > 0 and p >= self.drift_positions[j]
            which = nxt if past else j
        elif self.kind == "gradual":
            which = nxt if rng.random() < w else j
        else:
            which = nxt if w >= 0.5 else j
            mu_k = (1.0 - w) * self._mu[j][0][k] + w * self._mu[nxt][0][k]
            x = rng.normal(mu_k, self._scale[k])
            return x, int(self._mu[which][2][k]), which
        mu, _, cls = self._mu[which]
        x = rng.normal(mu[k], self._scale[k])
        return x, int(cls[k]), which

    def data(self) -> tuple[np.ndarray, np.ndarray]:
        """Generate the whole stream.

        Returns:
            ``(X, y)`` with shapes ``(n_samples, n_features)`` and
            ``(n_samples,)``.
        """
        rng = np.random.default_rng(None if self.seed is None else self.seed + 1)
        X = np.zeros((self.n_samples, self.n_features))
        y = np.zeros(self.n_samples, dtype=int)
        concepts = np.zeros(self.n_samples, dtype=int)
        for i in range(self.n_samples):
            p = i / max(1, self.n_samples - 1)
            x, label, which = self._sample_one(rng, p)
            X[i] = x
            y[i] = label
            concepts[i] = which
        self.concepts_ = concepts
        return X, y

    def stream(self) -> Iterator[tuple[Signal, int]]:
        """Yield ``(Signal, y)`` pairs in time order."""
        X, y = self.data()
        names = [f"x{i}" for i in range(self.n_features)]
        units = ["" for _ in range(self.n_features)]
        for i in range(self.n_samples):
            sig = Signal(
                values=X[i],
                names=names,
                units=units,
                t=float(i),
            )
            yield sig, int(y[i])
