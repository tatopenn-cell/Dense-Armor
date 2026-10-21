"""Models with several targets, built on a single-target model fitted on the whole table.

``factory`` returns a fresh single-target model with ``fit(X, y)``; a classifier
also has ``predict_proba(X)`` (the probability of the positive class) and a
regressor ``predict(X)``. ``Y`` has one column per target.

References:
    Spyromitros-Xioufis, E. et al. (2016). Multi-target regression via input
        space expansion, Section 2 (not in the folder, bibliographic only).
    Read, J. et al. (2009). Classifier chains for multi-label classification
        (not in the folder, bibliographic only).
    Dembczynski, K., Cheng, W., Hullermeier, E. (2010). Bayes optimal
        multilabel classification via probabilistic classifier chains, Eq. (10)
        to (12) (not in the folder, bibliographic only).
"""

from __future__ import annotations

from collections.abc import Callable
from itertools import product

import numpy as np

from dense_armor.ml.base import MLBase

__all__ = [
    "PerOutputClassifier",
    "PerOutputRegressor",
    "ClassifierChain",
    "RegressorChain",
    "ProbabilisticClassifierChain",
]


def _order(order, q: int, seed: int) -> list[int]:
    if order is not None:
        return list(order)
    return list(np.random.default_rng(seed).permutation(q))


class PerOutputClassifier(MLBase):
    """One binary classifier per target column, trained on ``X`` alone.

    Args:
        factory: callable returning a fresh binary classifier.
        order: unused, kept for symmetry with the chains.
        seed: unused.

    Example:
        >>> from dense_armor.ml.linear import LogisticRegression
        >>> X = [[0.0], [1.0], [2.0], [3.0]]
        >>> Y = [[0, 1], [0, 1], [1, 0], [1, 0]]
        >>> m = PerOutputClassifier(lambda: LogisticRegression(l2=0.1)).fit(X, Y)
        >>> m.predict([[0.0], [3.0]]).tolist()
        [[0, 1], [1, 0]]
    """

    def __init__(
        self, factory: Callable, order: list | None = None, seed: int = 0
    ) -> None:
        self.factory = factory
        self.order = order
        self.seed = seed

    def fit(self, X, y=None):  # type: ignore[override]
        A, Y = np.asarray(X, dtype=float), np.asarray(y)
        self.estimators_ = [self.factory().fit(A, Y[:, j]) for j in range(Y.shape[1])]
        return self

    def predict_proba(self, X) -> np.ndarray:
        """Probability of the positive class of every target."""
        return np.column_stack([e.predict_proba(X) for e in self.estimators_])

    def predict(self, X) -> np.ndarray:
        """The label of every target."""
        return (self.predict_proba(X) >= 0.5).astype(int)


class PerOutputRegressor(MLBase):
    """One regressor per target column, trained on ``X`` alone.

    Args:
        factory: callable returning a fresh regressor.
        order: unused, kept for symmetry with the chains.
        seed: unused.

    Example:
        >>> from dense_armor.ml.linear import LinearRegression
        >>> X = [[0.0], [1.0], [2.0]]
        >>> m = PerOutputRegressor(LinearRegression).fit(X, [[1.0, 0.0], [3.0, -1.0], [5.0, -2.0]])
        >>> m.predict([[3.0]]).round(4).tolist()
        [[7.0, -3.0]]
    """

    def __init__(
        self, factory: Callable, order: list | None = None, seed: int = 0
    ) -> None:
        self.factory = factory
        self.order = order
        self.seed = seed

    def fit(self, X, y=None):  # type: ignore[override]
        A, Y = np.asarray(X, dtype=float), np.asarray(y, dtype=float)
        self.estimators_ = [self.factory().fit(A, Y[:, j]) for j in range(Y.shape[1])]
        return self

    def predict(self, X) -> np.ndarray:
        """The prediction of every target."""
        return np.column_stack([e.predict(X) for e in self.estimators_])


class ClassifierChain(MLBase):
    """Chained binary classifiers: link ``j`` sees ``X`` and the targets of the earlier links.

    At fit time the earlier targets are the true ones; at predict time they are
    the predicted ones (Read et al. 2009).

    Args:
        factory: callable returning a fresh binary classifier.
        order: the order of the targets, or ``None`` for a random permutation.
        seed: seed of the random order.

    Example:
        >>> from dense_armor.ml.linear import LogisticRegression
        >>> X = [[0.0], [1.0], [2.0], [3.0]]
        >>> Y = [[0, 1], [0, 1], [1, 0], [1, 0]]
        >>> m = ClassifierChain(lambda: LogisticRegression(l2=0.1), order=[0, 1]).fit(X, Y)
        >>> m.predict([[0.0], [3.0]]).tolist()
        [[0, 1], [1, 0]]
    """

    def __init__(
        self, factory: Callable, order: list | None = None, seed: int = 0
    ) -> None:
        self.factory = factory
        self.order = order
        self.seed = seed

    def fit(self, X, y=None):  # type: ignore[override]
        A, Y = np.asarray(X, dtype=float), np.asarray(y)
        self.order_ = _order(self.order, Y.shape[1], self.seed)
        self.estimators_ = []
        for pos, j in enumerate(self.order_):
            Z = np.hstack([A, Y[:, self.order_[:pos]]])
            self.estimators_.append(self.factory().fit(Z, Y[:, j]))
        return self

    def predict_proba(self, X) -> np.ndarray:
        """Probability of the positive class of every target, in the columns of ``Y``."""
        A = np.asarray(X, dtype=float)
        out = np.zeros((A.shape[0], len(self.order_)))
        for pos, (j, e) in enumerate(zip(self.order_, self.estimators_)):
            Z = np.hstack([A, (out[:, self.order_[:pos]] >= 0.5).astype(float)])
            out[:, j] = e.predict_proba(Z)
        return out

    def predict(self, X) -> np.ndarray:
        """The label of every target."""
        return (self.predict_proba(X) >= 0.5).astype(int)


class RegressorChain(MLBase):
    """Chained regressors: link ``j`` sees ``X`` and the targets of the earlier links.

    Args:
        factory: callable returning a fresh regressor.
        order: the order of the targets, or ``None`` for a random permutation.
        seed: seed of the random order.

    Example:
        >>> from dense_armor.ml.linear import LinearRegression
        >>> X = [[0.0], [1.0], [2.0]]
        >>> m = RegressorChain(lambda: LinearRegression(l2=1e-9), order=[0, 1]).fit(X, [[1.0, 2.0], [3.0, 6.0], [5.0, 10.0]])
        >>> m.predict([[3.0]]).round(2).tolist()
        [[7.0, 14.0]]
    """

    def __init__(
        self, factory: Callable, order: list | None = None, seed: int = 0
    ) -> None:
        self.factory = factory
        self.order = order
        self.seed = seed

    def fit(self, X, y=None):  # type: ignore[override]
        A, Y = np.asarray(X, dtype=float), np.asarray(y, dtype=float)
        self.order_ = _order(self.order, Y.shape[1], self.seed)
        self.estimators_ = []
        for pos, j in enumerate(self.order_):
            Z = np.hstack([A, Y[:, self.order_[:pos]]])
            self.estimators_.append(self.factory().fit(Z, Y[:, j]))
        return self

    def predict(self, X) -> np.ndarray:
        """The prediction of every target, each link using the earlier predictions."""
        A = np.asarray(X, dtype=float)
        out = np.zeros((A.shape[0], len(self.order_)))
        for pos, (j, e) in enumerate(zip(self.order_, self.estimators_)):
            out[:, j] = e.predict(np.hstack([A, out[:, self.order_[:pos]]]))
        return out


class ProbabilisticClassifierChain(ClassifierChain):
    """Probabilistic classifier chain: the joint mode of ``P(y | x)`` (Dembczynski et al. 2010).

    The joint is ``P(y) = f_1(x) prod_{i >= 2} f_i(x, y_1..y_{i-1})`` (Eq. 11-12)
    and the prediction is its arg max over the ``2^m`` label vectors (Eq. 10).

    Args:
        factory: callable returning a fresh binary classifier.
        order: the order of the targets, or ``None`` for a random permutation.
        seed: seed of the random order.

    Example:
        >>> from dense_armor.ml.linear import LogisticRegression
        >>> X = [[0.0], [1.0], [2.0], [3.0]]
        >>> Y = [[0, 1], [0, 1], [1, 0], [1, 0]]
        >>> m = ProbabilisticClassifierChain(lambda: LogisticRegression(l2=0.1), order=[0, 1]).fit(X, Y)
        >>> m.predict([[0.0], [3.0]]).tolist()
        [[0, 1], [1, 0]]
    """

    def joint_proba(self, X) -> tuple[np.ndarray, np.ndarray]:
        """The label vectors and the joint probability of each, one row per sample."""
        A = np.asarray(X, dtype=float)
        m = len(self.order_)
        combos = np.array(list(product([0, 1], repeat=m)))
        joint = np.ones((A.shape[0], len(combos)))
        for c, y in enumerate(combos):
            for pos, (j, e) in enumerate(zip(self.order_, self.estimators_)):
                prev = np.tile(y[self.order_[:pos]], (A.shape[0], 1)).astype(float)
                p = e.predict_proba(np.hstack([A, prev]))
                joint[:, c] *= p if y[j] == 1 else 1.0 - p
        return combos, joint

    def predict(self, X) -> np.ndarray:
        """The label vector with the highest joint probability."""
        combos, joint = self.joint_proba(X)
        return combos[np.argmax(joint, axis=1)]
