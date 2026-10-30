"""scikit-learn bridge for Dense-Armor.

Wraps a Dense-Armor online estimator in a scikit-learn estimator
(``SklearnClassifier``, ``SklearnRegressor``, ``SklearnTransformer``),
and a scikit-learn estimator in a Dense-Armor ``Estimator`` (``from_sklearn``).

Needs the optional dependency: ``pip install dense-armor[sklearn]``.
"""

from __future__ import annotations

import copy
from typing import Any

import numpy as np

try:
    from sklearn.base import (
        BaseEstimator,
        ClassifierMixin,
        RegressorMixin,
        TransformerMixin,
    )
    from sklearn.utils.validation import check_array, check_is_fitted, column_or_1d
    from sklearn.utils.multiclass import check_classification_targets
    from scipy import sparse as _sparse
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "dense_armor.bridges.sklearn needs scikit-learn: "
        "pip install dense-armor[sklearn]"
    ) from exc

from dense_armor.roles import Estimator

__all__ = [
    "SklearnClassifier",
    "SklearnRegressor",
    "SklearnTransformer",
    "from_sklearn",
]


def _fresh(estimator: Any) -> Any:
    """Return a fresh Dense-Armor estimator from a factory or an instance."""
    if isinstance(estimator, type):
        return estimator()
    if callable(estimator) and not hasattr(estimator, "learn_one"):
        return estimator()
    if hasattr(estimator, "clone"):
        return estimator.clone()
    return copy.deepcopy(estimator)


def _as_array(X: Any, *, min_samples: int = 1) -> np.ndarray:
    """Convert X to a 2D float64 array using the sklearn validators.

    Sparse, complex, empty, 1D and NaN/inf inputs are rejected with the
    message scikit-learn expects from a compliant estimator.
    """
    if _sparse.issparse(X):
        raise ValueError(
            "Sparse data was passed, but dense data is required by "
            "dense_armor.bridges.sklearn"
        )
    return check_array(
        X,
        dtype=float,
        ensure_2d=True,
        accept_sparse=False,
        input_name="X",
        ensure_min_samples=min_samples,
        ensure_all_finite=True,
    )


def _y_classifier(y: Any) -> np.ndarray:
    if y is None:
        raise ValueError(
            "This estimator requires y to be passed, but the target y is None"
        )
    y = column_or_1d(y, warn=True)
    y = check_array(
        y,
        ensure_2d=False,
        dtype=None,
        input_name="y",
        ensure_min_samples=2,
        ensure_all_finite=True,
    )
    check_classification_targets(y)
    return y


def _y_regressor(y: Any) -> np.ndarray:
    if y is None:
        raise ValueError(
            "This estimator requires y to be passed, but the target y is None"
        )
    y = column_or_1d(y, warn=True)
    return check_array(
        y,
        ensure_2d=False,
        dtype=float,
        input_name="y",
        ensure_min_samples=2,
        ensure_all_finite=True,
    )


class _DenseWrapper:
    """Common machinery of the three wrappers; not a BaseEstimator."""

    def __init__(self, estimator: Any, feature_names: Any = None) -> None:
        self.estimator = estimator
        self.feature_names = feature_names

    def _reset(self) -> None:
        for attr in (
            "model_",
            "n_features_in_",
            "_names_",
            "classes_",
            "_out_keys_",
            "feature_names_out_",
        ):
            self.__dict__.pop(attr, None)

    def _setup(self, X: Any, reset: bool, min_samples: int = 1) -> np.ndarray:
        arr = _as_array(X, min_samples=min_samples)
        if reset:
            self._reset()
            self.model_ = _fresh(self.estimator)
        if not hasattr(self, "model_"):
            self.model_ = _fresh(self.estimator)
        if hasattr(self, "n_features_in_") and arr.shape[1] != self.n_features_in_:
            raise ValueError(
                "X has {} features, but {} is expecting {} features as input".format(
                    arr.shape[1],
                    type(self).__name__,
                    self.n_features_in_,
                )
            )
        self.n_features_in_ = arr.shape[1]
        if not hasattr(self, "_names_"):
            if self.feature_names is not None:
                self._names_ = list(self.feature_names)
            elif hasattr(X, "columns"):
                self._names_ = list(X.columns)
            else:
                self._names_ = list(range(arr.shape[1]))
        return arr

    def _iter(self, arr: np.ndarray) -> Any:
        names = self._names_
        for row in arr:
            yield {name: float(v) for name, v in zip(names, row)}


class SklearnClassifier(ClassifierMixin, _DenseWrapper, BaseEstimator):
    """A Dense-Armor classifier behind the scikit-learn classifier interface.

    Example:
        >>> import numpy as np
        >>> from dense_armor.bridges.sklearn import SklearnClassifier
        >>> from dense_armor.utility.learn.online_classifiers import OnlineGaussianNB
        >>> X = np.array([[0.0], [1.0], [2.0], [3.0]])
        >>> y = np.array([0, 0, 1, 1])
        >>> m = SklearnClassifier(OnlineGaussianNB).fit(X, y)
        >>> m.predict(np.array([[0.5], [2.5]])).tolist()
        [0, 1]
    """

    def fit(self, X: Any, y: Any) -> "SklearnClassifier":
        arr = self._setup(X, reset=True, min_samples=2)
        y_arr = _y_classifier(y)
        if arr.shape[0] != len(y_arr):
            raise ValueError(
                "dense_armor.bridges.sklearn: X and y differ in length"
            )
        self.classes_ = np.unique(y_arr)
        for x, yi in zip(self._iter(arr), y_arr):
            self.model_.learn_one(x, yi)
        return self

    def partial_fit(
        self, X: Any, y: Any, classes: Any = None
    ) -> "SklearnClassifier":
        if not hasattr(self, "model_"):
            if classes is not None:
                self.classes_ = np.asarray(classes)
            return self.fit(X, y)
        if y is None:
            raise ValueError("dense_armor.bridges.sklearn: y is required")
        arr = self._setup(X, reset=False)
        y_arr = np.asarray(y)
        for x, yi in zip(self._iter(arr), y_arr):
            self.model_.learn_one(x, yi)
        return self

    def predict(self, X: Any) -> np.ndarray:
        check_is_fitted(self, "model_")
        proba = self.predict_proba(X)
        idx = np.argmax(proba, axis=1)
        return np.asarray(self.classes_)[idx]

    def predict_proba(self, X: Any) -> np.ndarray:
        check_is_fitted(self, "model_")
        arr = self._setup(X, reset=False)
        classes = list(self.classes_)
        proba = []
        for x in self._iter(arr):
            p = self.model_.predict_proba_one(x) or {}
            proba.append([float(p.get(c, 0.0)) for c in classes])
        return np.asarray(proba, dtype=float)


class SklearnRegressor(RegressorMixin, _DenseWrapper, BaseEstimator):
    """A Dense-Armor regressor behind the scikit-learn regressor interface."""

    def fit(self, X: Any, y: Any) -> "SklearnRegressor":
        arr = self._setup(X, reset=True, min_samples=2)
        y_arr = _y_regressor(y)
        if arr.shape[0] != len(y_arr):
            raise ValueError(
                "dense_armor.bridges.sklearn: X and y differ in length"
            )
        for x, yi in zip(self._iter(arr), y_arr):
            self.model_.learn_one(x, float(yi))
        return self

    def partial_fit(self, X: Any, y: Any) -> "SklearnRegressor":
        if not hasattr(self, "model_"):
            return self.fit(X, y)
        if y is None:
            raise ValueError("dense_armor.bridges.sklearn: y is required")
        arr = self._setup(X, reset=False)
        y_arr = np.asarray(y, dtype=float)
        for x, yi in zip(self._iter(arr), y_arr):
            self.model_.learn_one(x, float(yi))
        return self

    def predict(self, X: Any) -> np.ndarray:
        check_is_fitted(self, "model_")
        arr = self._setup(X, reset=False)
        return np.asarray(
            [self.model_.predict_one(x) for x in self._iter(arr)], dtype=float
        )


class SklearnTransformer(TransformerMixin, _DenseWrapper, BaseEstimator):
    """A Dense-Armor transformer behind the scikit-learn transformer interface."""

    def fit(self, X: Any, y: Any = None) -> "SklearnTransformer":
        arr = self._setup(X, reset=True, min_samples=1)
        for x in self._iter(arr):
            self.model_.learn_one(x, y)
        sample = next(self._iter(arr))
        out = self.model_.transform_one(sample) or {}
        self._out_keys_ = list(out.keys())
        self.feature_names_out_ = np.asarray([str(k) for k in out.keys()])
        return self

    def partial_fit(self, X: Any, y: Any = None) -> "SklearnTransformer":
        if not hasattr(self, "model_"):
            return self.fit(X, y)
        arr = self._setup(X, reset=False)
        for x in self._iter(arr):
            self.model_.learn_one(x, y)
        return self

    def transform(self, X: Any) -> np.ndarray:
        check_is_fitted(self, "model_")
        arr = self._setup(X, reset=False)
        keys = self._out_keys_
        rows = []
        for x in self._iter(arr):
            d = self.model_.transform_one(x) or {}
            rows.append([float(d.get(k, np.nan)) for k in keys])
        return np.asarray(rows, dtype=float)

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        check_is_fitted(self, "model_")
        return np.asarray(self.feature_names_out_)


def from_sklearn(
    est: Any,
    refit_every: int | None = None,
    classes: Any = None,
) -> Estimator:
    """Wrap a scikit-learn estimator as a Dense-Armor ``Estimator``.

    If the wrapped estimator has ``partial_fit`` (a scikit-learn online
    estimator), ``learn_one`` calls ``partial_fit`` on the current row.
    The first call to ``partial_fit`` of a scikit-learn classifier must
    carry the list of classes; pass ``classes`` explicitly, or leave it
    ``None`` and the adapter buffers the rows until it has seen at least
    two distinct classes, then calls ``partial_fit`` with them.

    If the estimator has no ``partial_fit`` (a batch estimator),
    ``refit_every`` must be given and the rows are collected and refit
    every ``refit_every`` rows.

    Args:
        est: a scikit-learn estimator.
        refit_every: refit interval for a batch estimator. Required when
            the estimator has no ``partial_fit``, ignored otherwise.
        classes: the class list for a ``partial_fit`` classifier. When
            omitted, the adapter buffers rows until it sees two classes.

    Returns:
        A Dense-Armor ``Estimator`` with ``learn_one``, ``predict_one``,
        and ``predict_proba_one`` if the wrapped estimator has
        ``predict_proba``.
    """
    has_partial = hasattr(est, "partial_fit")
    if has_partial and refit_every is not None:
        raise ValueError(
            "from_sklearn: the wrapped estimator has partial_fit, "
            "learn_one calls it, so refit_every must be None"
        )
    if not has_partial and refit_every is None:
        raise ValueError(
            "from_sklearn: the wrapped estimator has no partial_fit, "
            "refit_every must be given (rows are collected and refit "
            "every refit_every rows)"
        )
    return _DenseArmorAdapter(est, refit_every=refit_every, classes=classes)


class _DenseArmorAdapter(Estimator):
    """A Dense-Armor Estimator that delegates to a scikit-learn estimator."""

    def __init__(
        self,
        est: Any,
        refit_every: int | None = None,
        classes: Any = None,
    ) -> None:
        self.est = est
        self.refit_every = refit_every
        self._rows: list = []
        self._y: list = []
        self._model: Any = None
        self._classes = None if classes is None else list(classes)
        self._has_partial = hasattr(est, "partial_fit")
        self._partial_ready = False
        self._buffer_rows: list = []
        self._buffer_y: list = []
        if self._has_partial:
            self._model = copy.deepcopy(est)

    def _refit(self) -> None:
        model = copy.deepcopy(self.est)
        if self._rows:
            model.fit(np.asarray(self._rows, dtype=float), np.asarray(self._y))
        self._model = model

    def _partial_first(self, row: list, y: Any) -> None:
        self._buffer_rows.append(row)
        self._buffer_y.append(y)
        distinct = sorted(set(self._buffer_y))
        if len(distinct) < 2:
            return
        X = np.asarray(self._buffer_rows, dtype=float)
        y_arr = np.asarray(self._buffer_y)
        self._model.partial_fit(X, y_arr, classes=np.asarray(distinct))
        self._classes = list(distinct)
        self._partial_ready = True
        self._buffer_rows = []
        self._buffer_y = []

    def learn_one(self, x: Any, y: Any, t: Any = None) -> "_DenseArmorAdapter":
        row = [float(v) for v in x.values()]
        if self._has_partial:
            if not self._partial_ready:
                if self._classes is not None:
                    self._model.partial_fit(
                        np.asarray([row], dtype=float),
                        np.asarray([y]),
                        classes=np.asarray(self._classes),
                    )
                    self._partial_ready = True
                else:
                    self._partial_first(row, y)
                return self
            if y not in self._classes:
                self._classes.append(y)
            self._model.partial_fit(np.asarray([row], dtype=float), np.asarray([y]))
        else:
            self._rows.append(row)
            self._y.append(y)
            if len(self._rows) % int(self.refit_every) == 0:
                self._refit()
        return self

    def predict_one(self, x: Any, t: Any = None) -> Any:
        if self._model is None:
            self._refit()
        X = np.asarray([[float(v) for v in x.values()]])
        return self._model.predict(X)[0]

    def predict_proba_one(self, x: Any, t: Any = None) -> dict:
        if self._model is None:
            self._refit()
        X = np.asarray([[float(v) for v in x.values()]])
        proba = self._model.predict_proba(X)[0]
        classes = list(getattr(self._model, "classes_", range(len(proba))))
        return dict(zip(classes, (float(p) for p in proba)))
