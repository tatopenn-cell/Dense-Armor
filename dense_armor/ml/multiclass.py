"""Multiclass strategies built on a binary classifier fitted on the whole table.

``factory`` returns a fresh binary classifier with ``fit(X, y)`` and
``predict_proba(X)``, the probability of the positive class.

References:
    Rifkin, R., Klautau, A. (2004). In defense of one-vs-all classification:
        one binary problem per class (not in the folder, bibliographic only).
    Furnkranz, J. (2002). Round robin classification, JMLR 2: one binary problem
        per pair of classes, decided by votes (not in the folder,
        bibliographic only).
    Dietterich, T. G., Bakiri, G. (1995). Solving multiclass learning problems
        via error-correcting output codes, JAIR 2: a codeword per class and the
        closest codeword in Hamming distance (not in the folder, bibliographic
        only).
"""

from __future__ import annotations

from collections.abc import Callable
from itertools import combinations

import numpy as np

from dense_armor.ml.base import MLBase

__all__ = ["OneVsRestClassifier", "OneVsOneClassifier", "OutputCodeClassifier"]


class OneVsRestClassifier(MLBase):
    """One binary problem per class: that class against all the others.

    The class with the highest positive probability wins.

    Args:
        binary_factory: callable returning a fresh binary classifier.

    Example:
        >>> from dense_armor.ml.linear import LogisticRegression
        >>> X = [[0.0, 0.0], [0.0, 1.0], [5.0, 0.0], [5.0, 1.0], [0.0, 5.0], [1.0, 5.0]]
        >>> m = OneVsRestClassifier(lambda: LogisticRegression(l2=0.1)).fit(X, [0, 0, 1, 1, 2, 2])
        >>> m.predict([[0.5, 0.5], [5.0, 0.5], [0.5, 5.0]]).tolist()
        [0, 1, 2]
    """

    def __init__(self, binary_factory: Callable) -> None:
        self.binary_factory = binary_factory

    def fit(self, X, y=None):  # type: ignore[override]
        A = np.asarray(X, dtype=float)
        t = np.asarray(y)
        self.classes_ = np.unique(t)
        self.estimators_ = [
            self.binary_factory().fit(A, (t == c).astype(int)) for c in self.classes_
        ]
        return self

    def predict_proba(self, X) -> np.ndarray:
        """Positive probabilities of the binary problems, scaled to sum to 1."""
        P = np.column_stack([e.predict_proba(X) for e in self.estimators_])
        return P / P.sum(axis=1, keepdims=True)

    def predict(self, X) -> np.ndarray:
        """The class with the highest positive probability."""
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


class OneVsOneClassifier(MLBase):
    """One binary problem per pair of classes; each votes for one class.

    The class with the most votes wins; a tie goes to the class that comes first.

    Args:
        binary_factory: callable returning a fresh binary classifier.

    Example:
        >>> from dense_armor.ml.linear import LogisticRegression
        >>> X = [[0.0, 0.0], [0.0, 1.0], [5.0, 0.0], [5.0, 1.0], [0.0, 5.0], [1.0, 5.0]]
        >>> m = OneVsOneClassifier(lambda: LogisticRegression(l2=0.1)).fit(X, [0, 0, 1, 1, 2, 2])
        >>> m.predict([[0.5, 0.5], [0.5, 5.0]]).tolist()
        [0, 2]
    """

    def __init__(self, binary_factory: Callable) -> None:
        self.binary_factory = binary_factory

    def fit(self, X, y=None):  # type: ignore[override]
        A = np.asarray(X, dtype=float)
        t = np.asarray(y)
        self.classes_ = np.unique(t)
        self.pairs_ = list(combinations(range(len(self.classes_)), 2))
        self.estimators_ = []
        for i, j in self.pairs_:
            keep = (t == self.classes_[i]) | (t == self.classes_[j])
            self.estimators_.append(
                self.binary_factory().fit(
                    A[keep], (t[keep] == self.classes_[j]).astype(int)
                )
            )
        return self

    def votes(self, X) -> np.ndarray:
        """Number of pairwise votes of every class, one row per sample."""
        n = np.asarray(X, dtype=float).shape[0]
        v = np.zeros((n, len(self.classes_)))
        for (i, j), e in zip(self.pairs_, self.estimators_):
            pj = e.predict_proba(X) >= 0.5
            v[:, j] += pj
            v[:, i] += ~pj
        return v

    def predict(self, X) -> np.ndarray:
        """The class with the most votes."""
        return self.classes_[np.argmax(self.votes(X), axis=1)]


class OutputCodeClassifier(MLBase):
    """Error-correcting output code: one binary classifier per bit of the class codewords.

    Every class gets a random codeword of ``code_size`` bits (+1 or -1) and one
    classifier is trained per bit; a row goes to the class whose codeword is
    closest in Hamming distance to the predicted bits. The code design (random bits,
    redrawn until every column has both signs) is our choice: Dietterich and Bakiri
    (Section 2) also separate the rows of the code matrix.

    Args:
        binary_factory: callable returning a fresh binary classifier.
        code_size: number of bits, i.e. of binary classifiers.
        seed: seed of the code matrix.

    Example:
        >>> from dense_armor.ml.linear import LogisticRegression
        >>> X = [[0.0, 0.0], [0.0, 1.0], [5.0, 0.0], [5.0, 1.0], [0.0, 5.0], [1.0, 5.0]]
        >>> m = OutputCodeClassifier(lambda: LogisticRegression(l2=0.1), code_size=8).fit(X, [0, 0, 1, 1, 2, 2])
        >>> m.codes_.shape
        (3, 8)
    """

    def __init__(
        self, binary_factory: Callable, code_size: int = 6, seed: int = 0
    ) -> None:
        self.binary_factory = binary_factory
        self.code_size = code_size
        self.seed = seed

    def fit(self, X, y=None):  # type: ignore[override]
        A = np.asarray(X, dtype=float)
        t = np.asarray(y)
        self.classes_ = np.unique(t)
        k = len(self.classes_)
        rng = np.random.default_rng(self.seed)
        for _ in range(1000):
            codes = rng.choice([-1, 1], size=(k, self.code_size))
            if all(len(set(codes[:, b])) == 2 for b in range(self.code_size)):
                break
        self.codes_ = codes
        self.estimators_ = []
        for b in range(self.code_size):
            pos = np.isin(t, self.classes_[codes[:, b] == 1])
            self.estimators_.append(self.binary_factory().fit(A, pos.astype(int)))
        return self

    def predict(self, X) -> np.ndarray:
        """The class whose codeword is closest to the predicted bits."""
        bits = np.column_stack(
            [np.where(e.predict_proba(X) >= 0.5, 1, -1) for e in self.estimators_]
        )
        dist = (bits[:, None, :] != self.codes_[None, :, :]).sum(axis=2)
        return self.classes_[np.argmin(dist, axis=1)]
