"""Wrapper role: an estimator that delegates to a wrapped model."""

from __future__ import annotations

from dense_armor.base.estimator import Estimator


class Wrapper(Estimator):
    """Delegates ``_supervised`` and ``_multiclass`` to the wrapped
    model. Subclasses must implement ``_wrapped_model``.
    """

    @property
    def _wrapped_model(self) -> Estimator:
        raise NotImplementedError(f"{type(self).__name__} must define _wrapped_model")

    @property
    def _supervised(self) -> bool:  # type: ignore[override]
        return getattr(self._wrapped_model, "_supervised", True)

    @property
    def _multiclass(self) -> bool:  # type: ignore[override]
        return getattr(self._wrapped_model, "_multiclass", False)
