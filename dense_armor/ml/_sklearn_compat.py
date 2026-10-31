"""Compatibility of the batch classes of ``dense_armor.ml`` with scikit-learn tools.

The classes of ``dense_armor.ml`` have ``fit``, ``predict`` and ``transform``
on the whole table, but a ``Pipeline``, ``cross_val_score`` or
``GridSearchCV`` of scikit-learn also asks for three things:

- ``__sklearn_tags__``: what kind of estimator the object is (classifier,
  regressor, transformer). scikit-learn is imported inside the method, so
  ``import dense_armor.ml`` works without scikit-learn.
- ``score``: the accuracy of a classifier and the coefficient of
  determination of a regressor, as ``ClassifierMixin.score`` and
  ``RegressorMixin.score`` do.
- ``fit_transform(X, y=None, **fit_params)``: a ``Pipeline`` calls it with
  the target.

The mixin changes no algorithm and no default. The kind of a class comes
from its name and its methods: ``predict_proba`` or a name ending in
``Classifier`` makes a classifier, a name ending in ``Regressor`` or
``Regression`` a regressor, and ``transform`` a transformer.

References:
    scikit-learn documentation, "Developing scikit-learn estimators", the
    ``__sklearn_tags__`` method (scikit-learn 1.6 and later), and the
    ``r2_score`` convention for a constant target: the score is 1.0 for a
    perfect prediction and 0.0 otherwise.
"""

from __future__ import annotations

from typing import Any

import numpy as np

_CLASSIFIER = "classifier"
_REGRESSOR = "regressor"
_TRANSFORMER = "transformer"


def estimator_kind(cls: type) -> str | None:
    """Return ``"classifier"``, ``"regressor"``, ``"transformer"`` or ``None``.

    Example:
        >>> class SomeClassifier:
        ...     def predict(self, X): ...
        >>> estimator_kind(SomeClassifier)
        'classifier'
    """
    name = cls.__name__
    if hasattr(cls, "predict_proba") or name.endswith("Classifier"):
        return _CLASSIFIER
    if name.endswith(("Regressor", "Regression")):
        return _REGRESSOR
    if hasattr(cls, "transform"):
        return _TRANSFORMER
    return None


class _AvailableIf:
    """A method that exists on the instance only when ``check`` is true."""

    def __init__(self, func: Any, check: Any) -> None:
        self.func = func
        self.check = check
        self.__doc__ = func.__doc__

    def __set_name__(self, owner: type, name: str) -> None:
        self.name = name

    def __get__(self, obj: Any, objtype: type | None = None) -> Any:
        if obj is None:
            return self.func
        if not self.check(obj):
            raise AttributeError(
                f"{type(obj).__name__!r} object has no attribute {self.name!r}"
            )
        return self.func.__get__(obj, objtype)


def _is_predictor(obj: Any) -> bool:
    return estimator_kind(type(obj)) in (_CLASSIFIER, _REGRESSOR)


def _has_transform(obj: Any) -> bool:
    return hasattr(type(obj), "transform")


def _score(self: Any, X: Any, y: Any, sample_weight: Any = None) -> float:
    """Accuracy of a classifier, coefficient of determination of a regressor."""
    pred = np.asarray(self.predict(X))
    true = np.asarray(y)
    w = None if sample_weight is None else np.asarray(sample_weight, dtype=float)
    if estimator_kind(type(self)) == _CLASSIFIER:
        return float(np.average(pred == true, weights=w))
    true = true.astype(float)
    pred = pred.astype(float)
    if w is None:
        w = np.ones(true.shape[0])
    w = w.reshape((-1,) + (1,) * (true.ndim - 1))
    ss_res = np.sum(w * (true - pred) ** 2, axis=0)
    mean = np.sum(w * true, axis=0) / np.sum(w)
    ss_tot = np.sum(w * (true - mean) ** 2, axis=0)
    ok = ss_tot > 0
    r2 = np.where(
        ok, 1.0 - ss_res / np.where(ok, ss_tot, 1.0), np.where(ss_res == 0, 1.0, 0.0)
    )
    return float(np.mean(r2))


def _fit_transform(self: Any, X: Any, y: Any = None, **fit_params: Any) -> Any:
    """``fit`` then ``transform`` on the same table."""
    return self.fit(X, y, **fit_params).transform(X)


class SklearnCompat:
    """Mixin: ``__sklearn_tags__``, ``score`` and ``fit_transform`` for scikit-learn tools."""

    def __sklearn_tags__(self) -> Any:
        from sklearn.utils import (  # type: ignore[import-untyped]
            ClassifierTags,
            InputTags,
            RegressorTags,
            Tags,
            TargetTags,
            TransformerTags,
        )

        kind = estimator_kind(type(self))
        predictor = kind in (_CLASSIFIER, _REGRESSOR)
        return Tags(
            estimator_type=kind if predictor else None,
            target_tags=TargetTags(required=predictor),
            transformer_tags=TransformerTags() if kind == _TRANSFORMER else None,
            classifier_tags=ClassifierTags() if kind == _CLASSIFIER else None,
            regressor_tags=RegressorTags() if kind == _REGRESSOR else None,
            input_tags=InputTags(),
        )

    score = _AvailableIf(_score, _is_predictor)
    fit_transform = _AvailableIf(_fit_transform, _has_transform)
