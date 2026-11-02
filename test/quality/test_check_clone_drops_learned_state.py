"""check_clone_drops_learned_state rejects a clone that carries learned state."""

import pytest

from dense_armor.checks.check_estimator import check_clone_drops_learned_state
from dense_armor.roles import Regressor


class _Counts(Regressor):
    """A regressor whose learned state is a sample counter ``n_``."""

    def learn_one(self, x, y, t=None):
        self.n_ = getattr(self, "n_", 0) + 1
        return self

    def predict_one(self, x, t=None, return_std=False):
        return 0.0


class _BrokenClone(_Counts):
    """Clone that carries the learned counter over, which is a bug."""

    def clone(self, new_params=None, include_attributes=False):
        c = type(self)()
        if hasattr(self, "n_"):
            c.n_ = self.n_
        return c


class _ConstructorSetsLearned(Regressor):
    """Constructor already sets a trailing-underscore attribute."""

    def __init__(self):
        self.n_ = 0

    def learn_one(self, x, y, t=None):
        return self

    def predict_one(self, x, t=None, return_std=False):
        return 0.0


class _NotRebuildable(Regressor):
    """``__init__`` keeps its argument under a name ``get_params`` cannot see."""

    def __init__(self, required):
        self._hidden = required

    def learn_one(self, x, y, t=None):
        return self

    def predict_one(self, x, t=None, return_std=False):
        return 0.0


def test_good_clone_passes():
    est = _Counts()
    est.learn_one({"a": 1.0}, 0.0)
    assert est._has_learned()
    check_clone_drops_learned_state(est)


def test_fresh_estimator_returns_without_testing_clone():
    est = _Counts()
    assert not est._has_learned()
    check_clone_drops_learned_state(est)


def test_broken_clone_is_rejected():
    est = _BrokenClone()
    est.learn_one({"a": 1.0}, 0.0)
    assert est._has_learned()
    with pytest.raises(AssertionError, match=r"clone\(\) carried learned state"):
        check_clone_drops_learned_state(est)


def test_constructor_with_underscore_attribute_is_still_checked():
    # The constructor sets ``n_ = 0``, which violates the "no trailing
    # underscore in the constructor" convention, but clone() still
    # resets it to the constructor default, so the check passes.
    est = _ConstructorSetsLearned()
    check_clone_drops_learned_state(est)


def test_clone_that_copies_constructor_underscore_attribute_is_rejected():
    class _Broken(_ConstructorSetsLearned):
        def learn_one(self, x, y, t=None):
            self.n_ = getattr(self, "n_", 0) + 1
            return self

        def clone(self, new_params=None, include_attributes=False):
            c = type(self)()
            c.n_ = self.n_
            return c

    est = _Broken()
    est.learn_one({"a": 1.0}, 0.0)
    with pytest.raises(AssertionError, match="carried learned state in n_"):
        check_clone_drops_learned_state(est)


def test_constructor_not_rebuildable_from_params_is_not_applicable():
    with pytest.raises(NotImplementedError, match="cannot be rebuilt from get_params"):
        check_clone_drops_learned_state(_NotRebuildable(required=1))
