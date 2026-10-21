"""Forests of trees grown on bootstrap samples or with random thresholds.

References:
    Breiman, L. (2001). Random forests, Machine Learning 45(1): a tree per
        bootstrap sample, a random subset of features at each split (not in
        the folder, bibliographic only).
    Geurts, P., Ernst, D., Wehenkel, L. (2006). Extremely randomized trees,
        Machine Learning 63(1): the whole sample and a random threshold at
        each split (not in the folder, bibliographic only).
"""

from __future__ import annotations

import numpy as np

from dense_armor.ml.base import MLBase
from dense_armor.ml.tree import DecisionTreeClassifier, DecisionTreeRegressor

__all__ = ["RandomForestClassifier", "ExtraTreesRegressor"]

STOPPED = {
    "AdaptiveRandomForestClassifier": "the batch form is RandomForestClassifier: the drift detectors and the background trees have no batch counterpart",
    "AggregatedMondrianForestClassifier": "the aggregation over the prunings is defined online (Mourtada et al.); the batch Mondrian forest is in the tree module",
    "AggregatedMondrianForestRegressor": "the aggregation over the prunings is defined online (Mourtada et al.); the batch Mondrian forest is in the tree module",
}


class RandomForestClassifier(MLBase):
    """Random forest: trees on bootstrap samples with ``sqrt(d)`` random features per split.

    Args:
        n_trees: number of trees.
        max_features: features tried at each split, or ``None`` for ``sqrt(d)``.
        max_depth: depth limit of each tree, or ``None``.
        seed: seed of the bootstrap and of the feature choices.

    Example:
        >>> X = [[i / 10, i % 2] for i in range(6)] + [[1.0 + i / 10, i % 2] for i in range(6)]
        >>> m = RandomForestClassifier(n_trees=15, seed=2).fit(X, [0] * 6 + [1] * 6)
        >>> m.predict([[0.2, 0.5], [1.3, 0.5]]).tolist()
        [0, 1]
    """

    def __init__(
        self,
        n_trees: int = 10,
        max_features: int | None = None,
        max_depth: int | None = None,
        seed: int = 0,
    ) -> None:
        self.n_trees = n_trees
        self.max_features = max_features
        self.max_depth = max_depth
        self.seed = seed

    def fit(self, X, y=None):  # type: ignore[override]
        A = np.asarray(X, dtype=float)
        t = np.asarray(y)
        self.classes_ = np.unique(t)
        rng = np.random.default_rng(self.seed)
        mf = self.max_features or max(1, int(np.sqrt(A.shape[1])))
        self.trees_ = []
        for i in range(self.n_trees):
            idx = rng.integers(0, A.shape[0], size=A.shape[0])
            tree = DecisionTreeClassifier(
                max_depth=self.max_depth,
                max_features=mf,
                seed=int(rng.integers(1 << 30)),
            )
            self.trees_.append(tree.fit(A[idx], t[idx]))
        return self

    def predict_proba(self, X) -> np.ndarray:
        """Mean over the trees of the class frequencies of their leaves."""
        out = np.zeros((np.asarray(X).shape[0], len(self.classes_)))
        for tree in self.trees_:
            P = tree.predict_proba(X)
            for j, c in enumerate(tree.classes_):
                out[:, int(np.flatnonzero(self.classes_ == c)[0])] += P[:, j]
        return out / len(self.trees_)

    def predict(self, X) -> np.ndarray:
        """The class with the highest mean probability."""
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


class ExtraTreesRegressor(MLBase):
    """Extremely randomised trees: the whole sample and a random threshold at each split.

    Args:
        n_trees: number of trees.
        max_features: features tried at each split, or ``None`` for all.
        max_depth: depth limit of each tree, or ``None``.
        seed: seed of the thresholds and of the feature choices.

    Example:
        >>> X = [[0.0], [0.2], [1.0], [1.2]]
        >>> m = ExtraTreesRegressor(n_trees=5, seed=3).fit(X, [0.0, 0.0, 4.0, 4.0])
        >>> m.predict([[0.1], [1.1]]).round(1).tolist()
        [0.0, 4.0]
    """

    def __init__(
        self,
        n_trees: int = 20,
        max_features: int | None = None,
        max_depth: int | None = None,
        seed: int = 0,
    ) -> None:
        self.n_trees = n_trees
        self.max_features = max_features
        self.max_depth = max_depth
        self.seed = seed

    def fit(self, X, y=None):  # type: ignore[override]
        A = np.asarray(X, dtype=float)
        t = np.asarray(y, dtype=float)
        rng = np.random.default_rng(self.seed)
        self.trees_ = [
            DecisionTreeRegressor(
                max_depth=self.max_depth,
                max_features=self.max_features,
                splitter="random",
                seed=int(rng.integers(1 << 30)),
            ).fit(A, t)
            for _ in range(self.n_trees)
        ]
        return self

    def predict(self, X) -> np.ndarray:
        """The mean prediction of the trees."""
        return np.mean([t.predict(X) for t in self.trees_], axis=0)
