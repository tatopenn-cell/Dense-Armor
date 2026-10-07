"""Transformer role: feature union (+) and product (*)."""

from __future__ import annotations

from typing import Any

from dense_armor.base._util import call_with_t
from dense_armor.base.estimator import Estimator


class _UnionTransformer(Estimator):
    _supervised = False

    def __init__(self, a, b):
        self.a = a
        self.b = b

    def learn_one(self, x, y=None, t=None):
        for tr in (self.a, self.b):
            if y is None:
                call_with_t(tr.learn_one, x, t=t)
            else:
                call_with_t(tr.learn_one, x, y, t=t)
        return self

    def transform_one(self, x, t=None):
        out = dict(call_with_t(self.a.transform_one, x, t=t))
        out.update(call_with_t(self.b.transform_one, x, t=t))
        return out


class _ProductTransformer(Estimator):
    _supervised = False

    def __init__(self, a, b):
        self.a = a
        self.b = b

    def learn_one(self, x, y=None, t=None):
        for tr in (self.a, self.b):
            if y is None:
                call_with_t(tr.learn_one, x, t=t)
            else:
                call_with_t(tr.learn_one, x, y, t=t)
        return self

    def transform_one(self, x, t=None):
        a = call_with_t(self.a.transform_one, x, t=t)
        b = call_with_t(self.b.transform_one, x, t=t)
        return {f"{ka}*{kb}": va * vb for ka, va in a.items() for kb, vb in b.items()}


class Transformer(Estimator):
    """Unsupervised transformer: ``transform_one(x) -> dict``.

    ``a + b`` unions the two outputs, ``a * b`` takes the product of
    every pair of features.
    """

    _supervised = False

    def learn_one(self, x: dict, y: Any = None, t: float | None = None) -> Transformer:
        return self

    def transform_one(self, x: dict, t: float | None = None) -> dict[str, Any]:
        raise NotImplementedError

    def __add__(self, other: Transformer) -> Estimator:
        return _UnionTransformer(self, other)

    def __mul__(self, other: Transformer) -> Estimator:
        return _ProductTransformer(self, other)

    def learn_many(self, X, t=None) -> Transformer:
        for x in X:
            call_with_t(self.learn_one, x, t=t)
        return self

    def transform_many(self, X, t=None):
        return [call_with_t(self.transform_one, x, t=t) for x in X]


class TransformerSupervised(Transformer):
    """Supervised transformer: ``learn_one(x, y)``."""

    _supervised = True

    def learn_one(self, x: dict, y: Any = None, t: float | None = None) -> Transformer:
        raise NotImplementedError
