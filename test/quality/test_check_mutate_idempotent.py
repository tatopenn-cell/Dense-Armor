"""check_mutate_idempotent passes an idempotent mutate and rejects one that is not."""
import pytest

from dense_armor.checks.check_estimator import check_mutate_idempotent
from dense_armor.roles import Regressor


class _Shrink(Regressor):
    _mutable_attributes = frozenset({"shrink"})

    def __init__(self, shrink: float = 0.5):
        self.shrink = shrink

    def learn_one(self, x, y, t=None):
        return self

    def predict_one(self, x, t=None, return_std=False):
        return 0.0


class _Accumulating(_Shrink):
    def mutate(self, new_attrs):
        for k, v in new_attrs.items():
            setattr(self, k, getattr(self, k) + v)
        return self


class _NoMutable(_Shrink):
    _mutable_attributes = frozenset()


def test_idempotent_mutate_passes():
    est = _Shrink(0.5)
    check_mutate_idempotent(est)
    assert est.shrink == 0.5


def test_accumulating_mutate_is_rejected():
    with pytest.raises(AssertionError, match="applied twice"):
        check_mutate_idempotent(_Accumulating(0.5))


def test_no_mutable_attributes_returns_without_mutating():
    est = _NoMutable(0.5)
    check_mutate_idempotent(est)
    assert est.shrink == 0.5
