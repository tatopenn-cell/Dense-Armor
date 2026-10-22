"""k-nearest-neighbour models on the whole training table.

References:
    Cover, T., Hart, P. (1967). Nearest neighbor pattern classification,
        IEEE Trans. Information Theory 13(1) (not in the folder,
        bibliographic only). The Minkowski distance of order ``p`` and the
        uniform or inverse-distance vote are textbook definitions.
"""

from __future__ import annotations

import numpy as np

from dense_armor.ml.base import MLBase

__all__ = ["KNNClassifier", "KNNRegressor"]


class _KNN(MLBase):
    def __init__(
        self,
        n_neighbors: int = 5,
        p: float = 2.0,
        weights: str = "uniform",
        eps: float = 1e-9,
    ) -> None:
        if weights not in ("uniform", "distance"):
            raise ValueError(
                f"weights must be 'uniform' or 'distance', got {weights!r}"
            )
        self.n_neighbors = n_neighbors
        self.p = p
        self.weights = weights
        self.eps = eps

    def fit(self, X, y=None):  # type: ignore[override]
        self.X_ = np.asarray(X, dtype=float)
        self.X_ = self.X_[:, None] if self.X_.ndim == 1 else self.X_
        self.y_ = np.asarray(y)
        return self

    def _neighbours(self, X) -> tuple[np.ndarray, np.ndarray]:
        Q = np.asarray(X, dtype=float)
        Q = Q[:, None] if Q.ndim == 1 else Q
        d = (np.abs(Q[:, None, :] - self.X_[None, :, :]) ** self.p).sum(axis=2) ** (
            1.0 / self.p
        )
        k = min(self.n_neighbors, self.X_.shape[0])
        idx = np.argsort(d, axis=1, kind="stable")[:, :k]
        dist = np.take_along_axis(d, idx, axis=1)
        w = np.ones_like(dist) if self.weights == "uniform" else 1.0 / (dist + self.eps)
        return idx, w


class KNNClassifier(_KNN):
    """k-nearest-neighbour classifier with a uniform or inverse-distance vote.

    Args:
        n_neighbors: number of neighbours ``k``.
        p: order of the Minkowski distance.
        weights: ``"uniform"`` or ``"distance"`` (weight ``1 / (d + eps)``).
        eps: constant added to the distance.

    Example:
        >>> m = KNNClassifier(n_neighbors=1).fit([[0.0], [1.0], [5.0]], [0, 0, 1])
        >>> m.predict([[0.4], [4.0]]).tolist()
        [0, 1]
    """

    def fit(self, X, y=None):  # type: ignore[override]
        super().fit(X, y)
        self.classes_ = np.unique(self.y_)
        return self

    def predict_proba(self, X) -> np.ndarray:
        """Share of the vote of each class among the neighbours."""
        idx, w = self._neighbours(X)
        votes = np.zeros((idx.shape[0], len(self.classes_)))
        for j, c in enumerate(self.classes_):
            votes[:, j] = (w * (self.y_[idx] == c)).sum(axis=1)
        return votes / votes.sum(axis=1, keepdims=True)

    def predict(self, X) -> np.ndarray:
        """The class with the largest vote."""
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


class KNNRegressor(_KNN):
    """k-nearest-neighbour regressor: the weighted mean of the neighbours' targets.

    Args:
        n_neighbors: number of neighbours ``k``.
        p: order of the Minkowski distance.
        weights: ``"uniform"`` or ``"distance"``.
        eps: constant added to the distance.

    Example:
        >>> KNNRegressor(n_neighbors=2).fit([[0.0], [1.0], [9.0]], [0.0, 2.0, 10.0]).predict([[0.4]]).tolist()
        [1.0]
    """

    def predict(self, X) -> np.ndarray:
        """The weighted mean target of the ``k`` nearest rows."""
        idx, w = self._neighbours(X)
        return (w * self.y_[idx].astype(float)).sum(axis=1) / w.sum(axis=1)
