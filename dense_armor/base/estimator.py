"""Estimator base with pipeline composition and tags.

Estimators share dict-in / dict-out semantics (Montiel et al.,
arXiv:2012.04740) and the sklearn parameter / clone / inspect contract
(Buitinck et al., arXiv:1309.0238).
"""

from __future__ import annotations

from typing import Any

from dense_armor.base.base import Base


class _PipelineAwareMeta(type):
    """Instance check that follows the last step of a pipeline."""

    def __instancecheck__(cls, instance: Any) -> bool:
        if getattr(instance, "_is_pipeline", False):
            steps = getattr(instance, "steps", ())
            if not steps:
                return False
            return type.__instancecheck__(cls, steps[-1][1])
        return type.__instancecheck__(cls, instance)


class _Pipeline(Base):
    """Sequential composition; the last step defines the pipeline role."""

    _is_pipeline = True

    def __init__(self, steps: list[tuple[str, Base]]):
        self.steps = list(steps)

    def __or__(self, other) -> _Pipeline:
        if not hasattr(other, "learn_one"):
            return NotImplemented
        return _Pipeline(self.steps + [(type(other).__name__, other)])

    def _forward(self, x: dict) -> dict:
        for _, s in self.steps[:-1]:
            s_any: Any = s
            x = s_any.transform_one(x)
        return x

    def learn_one(self, x: dict, y: Any = None, t: float | None = None) -> _Pipeline:
        h = self._forward(x)
        last: Any = self.steps[-1][1]
        if y is None:
            last.learn_one(h, t=t) if t is not None else last.learn_one(h)
        else:
            last.learn_one(h, y, t=t) if t is not None else last.learn_one(h, y)
        return self

    def predict_one(self, x: dict, t: float | None = None):
        h = self._forward(x)
        last: Any = self.steps[-1][1]
        return last.predict_one(h, t=t) if t is not None else last.predict_one(h)

    def predict_proba_one(self, x: dict, t: float | None = None):
        h = self._forward(x)
        last: Any = self.steps[-1][1]
        return (
            last.predict_proba_one(h, t=t)
            if t is not None
            else last.predict_proba_one(h)
        )

    def transform_one(self, x: dict, t: float | None = None) -> dict:
        for _, s in self.steps:
            s_any: Any = s
            x = s_any.transform_one(x)
        return x

    def __repr__(self) -> str:
        return " | ".join(f"{name}({type(s).__name__})" for name, s in self.steps)


class Estimator(Base, metaclass=_PipelineAwareMeta):
    """Base class for every Dense-Armor estimator."""

    _supervised: bool = True

    def __or__(self, other) -> _Pipeline:
        if not hasattr(other, "learn_one"):
            return NotImplemented
        return _Pipeline([(type(self).__name__, self), (type(other).__name__, other)])

    @property
    def _more_tags(self) -> dict[str, Any]:
        return {}

    def _get_tags(self) -> dict[str, Any]:
        tags: dict[str, Any] = {}
        for klass in reversed(type(self).__mro__):
            mt = klass.__dict__.get("_more_tags")
            if isinstance(mt, property):
                fget = mt.fget
                mt = fget(self) if fget is not None else {}
            if callable(mt) and not isinstance(mt, dict):
                mt = mt(self)
            if isinstance(mt, dict):
                tags.update(mt)
        return tags

    @property
    def _tags(self) -> set:
        """Names of the tags that are set, as a set (the interoperable form)."""
        return {k for k, v in self._get_tags().items() if v}
