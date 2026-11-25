"""Estimator base with pipeline composition and tags.

Estimators share dict-in / dict-out semantics (Montiel et al.,
arXiv:2012.04740) and the parameter / clone / inspect contract of a general-purpose machine-learning library
(Buitinck et al., arXiv:1309.0238).
"""

from typing import Any

from dense_armor.roles._util import (
    call_learn_one,
    call_with_t,
    sum_budgets,
    worst_memory_class,
)
from dense_armor.roles.root import Root


class _PipelineAwareMeta(type):
    """Instance check that follows the last step of a pipeline."""

    def __instancecheck__(cls, instance: Any) -> bool:
        if getattr(instance, "_is_pipeline", False):
            steps = getattr(instance, "steps", ())
            if not steps:
                return False
            return type.__instancecheck__(cls, steps[-1][1])
        return type.__instancecheck__(cls, instance)


class _Pipeline(Root):
    """Sequential composition; the last step defines the pipeline role."""

    _is_pipeline = True

    def __init__(self, steps: list[tuple[str, Root]]):
        self.steps = list(steps)

    @staticmethod
    def _through(step: Any, x: dict, t: float | None) -> dict:
        """``x`` after ``step``; a step with no ``transform_one`` lets it pass."""
        fn = getattr(step, "transform_one", None)
        return x if fn is None else call_with_t(fn, x, t=t)

    def _capability(self, name: str):
        """The method ``name`` of the pipeline, if the last step has it.

        A pipeline is a classifier, a regressor or a transformer according
        to its last step, so ``hasattr(pipeline, name)`` is true only when
        the last step has ``name``.
        """
        last = self.steps[-1][1] if self.steps else None
        if last is None or not hasattr(last, name):
            raise AttributeError(f"the last step of this pipeline has no {name}")
        return getattr(self, "_" + name)

    def __or__(self, other) -> '_Pipeline':
        if not hasattr(other, "learn_one"):
            return NotImplemented
        return _Pipeline(self.steps + [(type(other).__name__, other)])

    def _forward(self, x: dict, t: float | None = None) -> dict:
        for _, s in self.steps[:-1]:
            x = self._through(s, x, t)
        return x

    @property
    def budget_s(self) -> float | None:
        """Latency of ``learn_one``: ``transform_one`` and ``learn_one`` of every transformer, and ``learn_one`` of the last step.

        ``None`` if a step declares no ``budget_s``.
        """
        weights = [2.0] * (len(self.steps) - 1) + [1.0]
        return sum_budgets(zip(weights, (s for _, s in self.steps)))

    @property
    def memory_class(self) -> str:
        """The largest memory class of the steps."""
        return worst_memory_class([s for _, s in self.steps])

    def learn_one(self, x: dict, y: Any = None, t: float | None = None) -> '_Pipeline':
        if not self.steps:
            return self
        h = x
        for _, s in self.steps[:-1]:
            s_any: Any = s
            out = self._through(s_any, h, t)
            if hasattr(s_any, "learn_one"):
                call_learn_one(s_any.learn_one, h, y, t=t)
            h = out
        last: Any = self.steps[-1][1]
        if not hasattr(last, "learn_one"):
            return self
        if y is None:
            call_with_t(last.learn_one, h, t=t)
        else:
            call_with_t(last.learn_one, h, y, t=t)
        return self

    def _predict_one(self, x: dict, t: float | None = None):
        h = self._forward(x, t=t)
        last: Any = self.steps[-1][1]
        return call_with_t(last.predict_one, h, t=t)

    def _predict_proba_one(self, x: dict, t: float | None = None):
        h = self._forward(x, t=t)
        last: Any = self.steps[-1][1]
        return call_with_t(last.predict_proba_one, h, t=t)

    def _transform_one(self, x: dict, t: float | None = None) -> dict:
        for _, s in self.steps:
            x = self._through(s, x, t)
        return x

    @property
    def predict_one(self):
        """``predict_one`` of the last step, after the transformers."""
        return self._capability("predict_one")

    @property
    def predict_proba_one(self):
        """``predict_proba_one`` of the last step, after the transformers."""
        return self._capability("predict_proba_one")

    @property
    def transform_one(self):
        """``transform_one`` of every step; only if the last step has it."""
        return self._capability("transform_one")

    def __repr__(self) -> str:
        return " | ".join(f"{name}({type(s).__name__})" for name, s in self.steps)


class Estimator(Root, metaclass=_PipelineAwareMeta):
    """Root class for every Dense-Armor estimator."""

    _supervised: bool = True

    def __or__(self, other) -> '_Pipeline':
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
