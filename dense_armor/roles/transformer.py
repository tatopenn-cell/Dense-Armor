"""Transformer role: feature union (``+``) and feature product (``*``)."""

from typing import Any

from dense_armor.roles._util import call_with_t, sum_budgets, worst_memory_class
from dense_armor.roles.estimator import Estimator


class Transformer(Estimator):
    """Unsupervised transformer: ``transform_one(x) -> dict``.

    ``a + b`` is a transformer whose output is the union of the outputs of
    ``a`` and ``b``; ``a * b`` is a transformer whose output has one
    feature ``"ka*kb"`` for every pair of features of ``a`` and ``b``. Both
    results are transformers too, so the operators chain:
    ``(a + b) + c``, ``(a * b) * c`` and ``a + (b * c)`` are valid.

    Examples:
        >>> from dense_armor.roles.transformer import Transformer
        >>> class Shift(Transformer):
        ...     def __init__(self, name, by):
        ...         self.name, self.by = name, by
        ...     def transform_one(self, x, t=None):
        ...         return {self.name: x["v"] + self.by}
        >>> t = (Shift("a", 1.0) + Shift("b", 2.0)) + Shift("c", 3.0)
        >>> t.transform_one({"v": 0.0})
        {'a': 1.0, 'b': 2.0, 'c': 3.0}
        >>> (Shift("a", 1.0) * Shift("b", 2.0)).transform_one({"v": 1.0})
        {'a*b': 6.0}
    """

    _supervised = False

    def learn_one(
        self, x: dict, y: Any = None, t: float | None = None
    ) -> "Transformer":
        return self

    def transform_one(self, x: dict, t: float | None = None) -> dict[str, Any]:
        raise NotImplementedError

    def __add__(self, other: "Transformer") -> "Transformer":
        return _UnionTransformer(self, other)

    def __mul__(self, other: "Transformer") -> "Transformer":
        return _ProductTransformer(self, other)

    def learn_many(self, X, t=None) -> "Transformer":
        for x in X:
            call_with_t(self.learn_one, x, t=t)
        return self

    def transform_many(self, X, t=None):
        return [call_with_t(self.transform_one, x, t=t) for x in X]


class _PairTransformer(Transformer):
    """Two transformers learned together; the subclasses combine their outputs."""

    def __init__(self, a, b):
        self.a = a
        self.b = b

    @property
    def budget_s(self) -> float | None:
        """Latency of one sample: both transformers. ``None`` if one declares none."""
        return sum_budgets([(1.0, self.a), (1.0, self.b)])

    @property
    def memory_class(self) -> str:
        """The larger memory class of the two transformers."""
        return worst_memory_class([self.a, self.b])

    def learn_one(self, x, y=None, t=None):
        for tr in (self.a, self.b):
            if y is None:
                call_with_t(tr.learn_one, x, t=t)
            else:
                call_with_t(tr.learn_one, x, y, t=t)
        return self


class _UnionTransformer(_PairTransformer):
    """The union of the features of two transformers; ``b`` wins on a shared key."""

    def transform_one(self, x, t=None):
        out = dict(call_with_t(self.a.transform_one, x, t=t))
        out.update(call_with_t(self.b.transform_one, x, t=t))
        return out


class _ProductTransformer(_PairTransformer):
    """The product of every pair of features of two transformers."""

    def transform_one(self, x, t=None):
        a = call_with_t(self.a.transform_one, x, t=t)
        b = call_with_t(self.b.transform_one, x, t=t)
        return {f"{ka}*{kb}": va * vb for ka, va in a.items() for kb, vb in b.items()}


class TransformerSupervised(Transformer):
    """Supervised transformer: ``learn_one(x, y)``."""

    _supervised = True

    def learn_one(
        self, x: dict, y: Any = None, t: float | None = None
    ) -> "Transformer":
        raise NotImplementedError
