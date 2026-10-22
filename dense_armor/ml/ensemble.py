"""Ensembles fitted on the whole table.

A model is any object of this package with ``fit(X, y)``, ``predict_proba(X)``
(one row of class probabilities, columns in ``classes_`` order) and, for the
boosting, ``fit(X, y, sample_weight=w)``.

References:
    Breiman, L. (1996). Bagging predictors, Machine Learning 24(2): bootstrap
        samples and the mean of the models (not in the folder, bibliographic
        only).
    Freund, Y., Schapire, R. E. (1997). A decision-theoretic generalization of
        on-line learning and an application to boosting, JCSS 55(1): the
        Hedge weights ``exp(-eta L)`` and AdaBoost (not in the folder,
        bibliographic only). Zhu, J. et al. (2009). Multi-class AdaBoost
        (SAMME) for the multiclass weights (not in the folder, bibliographic
        only).
    Wolpert, D. H. (1992). Stacked generalization, Neural Networks 5(2): a
        meta-model on out-of-fold predictions (not in the folder,
        bibliographic only).
"""

from __future__ import annotations

import numpy as np

from dense_armor.ml.base import MLBase

__all__ = [
    "BaggingClassifier",
    "BoostingClassifier",
    "ExponentiallyWeightedAverage",
    "VotingClassifier",
    "StackingClassifier",
]


def _align(model, P: np.ndarray, classes: np.ndarray) -> np.ndarray:
    out = np.zeros((P.shape[0], len(classes)))
    for j, c in enumerate(model.classes_):
        out[:, int(np.flatnonzero(classes == c)[0])] = P[:, j]
    return out


class BaggingClassifier(MLBase):
    """Bagging: one model per bootstrap sample, probabilities averaged.

    Args:
        model: an unfitted model; each member is a ``clone`` of it.
        n_models: number of members.
        seed: seed of the bootstrap samples.

    Example:
        >>> from dense_armor.ml.tree import DecisionTreeClassifier
        >>> X = [[0.0], [0.2], [1.0], [1.2]]
        >>> m = BaggingClassifier(DecisionTreeClassifier(), n_models=5, seed=1).fit(X, [0, 0, 1, 1])
        >>> m.predict([[0.1], [1.1]]).tolist()
        [0, 1]
    """

    def __init__(self, model, n_models: int = 10, seed: int = 0) -> None:
        self.model = model
        self.n_models = n_models
        self.seed = seed

    def fit(self, X, y=None):  # type: ignore[override]
        A, t = np.asarray(X, dtype=float), np.asarray(y)
        self.classes_ = np.unique(t)
        rng = np.random.default_rng(self.seed)
        self.models_ = []
        for _ in range(self.n_models):
            idx = rng.integers(0, A.shape[0], size=A.shape[0])
            self.models_.append(self.model.clone().fit(A[idx], t[idx]))
        return self

    def predict_proba(self, X) -> np.ndarray:
        """Mean of the member probabilities."""
        return np.mean(
            [_align(m, m.predict_proba(X), self.classes_) for m in self.models_], axis=0
        )

    def predict(self, X) -> np.ndarray:
        """The class with the highest mean probability."""
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


class BoostingClassifier(MLBase):
    """AdaBoost with the multiclass weights SAMME (Zhu et al. 2009).

    Each round fits a model on the weighted sample; its error ``e`` gives
    ``alpha = ln((1 - e) / e) + ln(K - 1)`` and the weights of the misclassified
    rows are multiplied by ``exp(alpha)``. A member with error at or above
    ``1 - 1/K`` stops the boosting, and a member with no error is the last.

    Args:
        model: an unfitted model that accepts ``sample_weight``.
        n_models: largest number of rounds.
        seed: kept for the interface.

    Example:
        >>> from dense_armor.ml.tree import DecisionTreeClassifier
        >>> X = [[0.0], [1.0], [2.0], [3.0], [4.0], [5.0]]
        >>> m = BoostingClassifier(DecisionTreeClassifier(max_depth=1), n_models=5).fit(X, [0, 0, 1, 1, 0, 0])
        >>> m.predict([[0.5], [2.5], [4.5]]).tolist()
        [0, 1, 0]
    """

    def __init__(self, model, n_models: int = 10, seed: int = 0) -> None:
        self.model = model
        self.n_models = n_models
        self.seed = seed

    def fit(self, X, y=None):  # type: ignore[override]
        A, t = np.asarray(X, dtype=float), np.asarray(y)
        self.classes_ = np.unique(t)
        K = len(self.classes_)
        w = np.full(A.shape[0], 1.0 / A.shape[0])
        self.models_, self.alphas_ = [], []
        for _ in range(self.n_models):
            m = self.model.clone().fit(A, t, sample_weight=w)
            miss = m.predict(A) != t
            err = float(np.sum(w * miss) / w.sum())
            if err >= 1.0 - 1.0 / K:
                if not self.models_:
                    self.models_.append(m)
                    self.alphas_.append(1.0)
                break
            alpha = (
                np.log((1.0 - err) / max(err, 1e-12)) + np.log(K - 1.0)
                if K > 1
                else 1.0
            )
            self.models_.append(m)
            self.alphas_.append(float(alpha))
            if err == 0.0:
                break
            w = w * np.exp(alpha * miss)
            w = w / w.sum()
        return self

    def predict_proba(self, X) -> np.ndarray:
        """Share of the weighted votes of each class."""
        v = np.zeros((np.asarray(X).shape[0], len(self.classes_)))
        for m, a in zip(self.models_, self.alphas_):
            idx = np.searchsorted(self.classes_, m.predict(X))
            v[np.arange(v.shape[0]), idx] += a
        return v / v.sum(axis=1, keepdims=True)

    def predict(self, X) -> np.ndarray:
        """The class with the largest weighted vote."""
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


class ExponentiallyWeightedAverage(MLBase):
    """Weights ``exp(-eta L_m)`` from the total loss of each model on the data (Hedge).

    Args:
        eta: learning rate of the weights.

    Example:
        >>> P = [[1.0, 5.0], [2.0, 5.0], [3.0, 5.0]]
        >>> m = ExponentiallyWeightedAverage(eta=1.0).fit(P, [1.0, 2.0, 3.0])
        >>> m.weights_.round(3).tolist()
        [1.0, 0.0]
    """

    def __init__(self, eta: float = 0.5) -> None:
        self.eta = eta

    def fit(self, X, y=None):  # type: ignore[override]
        P = np.asarray(X, dtype=float)
        loss = ((P - np.asarray(y, dtype=float)[:, None]) ** 2).sum(axis=0)
        w = np.exp(-self.eta * (loss - loss.min()))
        self.weights_ = w / w.sum()
        return self

    def predict(self, X) -> np.ndarray:
        """The weighted average of the model predictions in the columns of ``X``."""
        return np.asarray(X, dtype=float) @ self.weights_


class VotingClassifier(MLBase):
    """A fixed list of models; the hard vote or the mean of the probabilities.

    Args:
        models: unfitted models; each is cloned.
        mode: ``"hard"`` or ``"soft"``.
        weights: weight of every model, or ``None``.

    Example:
        >>> from dense_armor.ml.tree import DecisionTreeClassifier
        >>> from dense_armor.ml.neighbors import KNNClassifier
        >>> X = [[0.0], [0.2], [1.0], [1.2]]
        >>> m = VotingClassifier([DecisionTreeClassifier(), KNNClassifier(n_neighbors=1)], mode="soft").fit(X, [0, 0, 1, 1])
        >>> m.predict([[0.1], [1.1]]).tolist()
        [0, 1]
    """

    def __init__(self, models, mode: str = "hard", weights=None) -> None:
        if mode not in ("hard", "soft"):
            raise ValueError(f"mode must be 'hard' or 'soft', got {mode!r}")
        self.models = models
        self.mode = mode
        self.weights = weights

    def fit(self, X, y=None):  # type: ignore[override]
        A, t = np.asarray(X, dtype=float), np.asarray(y)
        self.classes_ = np.unique(t)
        self.models_ = [m.clone().fit(A, t) for m in self.models]
        self.w_ = (
            np.ones(len(self.models))
            if self.weights is None
            else np.asarray(self.weights, dtype=float)
        )
        return self

    def predict_proba(self, X) -> np.ndarray:
        """Weighted mean of the probabilities (soft) or the share of the votes (hard)."""
        n = np.asarray(X).shape[0]
        out = np.zeros((n, len(self.classes_)))
        for m, w in zip(self.models_, self.w_):
            if self.mode == "soft":
                out += w * _align(m, m.predict_proba(X), self.classes_)
            else:
                out[np.arange(n), np.searchsorted(self.classes_, m.predict(X))] += w
        return out / out.sum(axis=1, keepdims=True)

    def predict(self, X) -> np.ndarray:
        """The class with the highest weighted vote or probability."""
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


class StackingClassifier(MLBase):
    """A meta-model fitted on the out-of-fold probabilities of the base models (Wolpert 1992).

    The rows are split in ``cv`` folds; the meta-features of a row come from base
    models fitted on the other folds, and the base models are then refitted on
    all the rows for prediction.

    Args:
        models: unfitted base models; each is cloned.
        meta_model: unfitted model fitted on the stacked probabilities.
        cv: number of folds.

    Example:
        >>> from dense_armor.ml.tree import DecisionTreeClassifier
        >>> from dense_armor.ml.neighbors import KNNClassifier
        >>> X = [[0.0], [0.1], [0.2], [0.3], [1.0], [1.1], [1.2], [1.3]]
        >>> m = StackingClassifier([KNNClassifier(n_neighbors=1)], DecisionTreeClassifier(), cv=2).fit(X, [0, 0, 0, 0, 1, 1, 1, 1])
        >>> m.predict([[0.15], [1.15]]).tolist()
        [0, 1]
    """

    def __init__(self, models, meta_model, cv: int = 5) -> None:
        self.models = models
        self.meta_model = meta_model
        self.cv = cv

    def fit(self, X, y=None):  # type: ignore[override]
        A, t = np.asarray(X, dtype=float), np.asarray(y)
        self.classes_ = np.unique(t)
        folds = np.arange(A.shape[0]) % self.cv
        Z = np.zeros((A.shape[0], len(self.models) * len(self.classes_)))
        for f in range(self.cv):
            tr, te = folds != f, folds == f
            for j, m in enumerate(self.models):
                fit = m.clone().fit(A[tr], t[tr])
                Z[te, j * len(self.classes_) : (j + 1) * len(self.classes_)] = _align(
                    fit, fit.predict_proba(A[te]), self.classes_
                )
        self.models_ = [m.clone().fit(A, t) for m in self.models]
        self.meta_ = self.meta_model.clone().fit(Z, t)
        return self

    def _meta_features(self, X) -> np.ndarray:
        return np.hstack(
            [_align(m, m.predict_proba(X), self.classes_) for m in self.models_]
        )

    def predict_proba(self, X) -> np.ndarray:
        """Probabilities of the meta-model on the base-model probabilities."""
        return _align(
            self.meta_, self.meta_.predict_proba(self._meta_features(X)), self.classes_
        )

    def predict(self, X) -> np.ndarray:
        """The class with the highest meta-model probability."""
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]
