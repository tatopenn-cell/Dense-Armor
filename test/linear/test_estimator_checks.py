"""Tests that prove each check in check_estimator does its job.

For each check there is one deliberately broken estimator that the
check must reject, and one test that confirms the rejection with the
check name in the assertion message. A check that never fires is not a
check. The five core checks (clone_independent, pickle_roundtrip,
state_dict_roundtrip, memory_growth_bounded, p99_within_budget) are the
ones most often silently no-op'd by a guard, so each has its own broken
estimator here.

The file also proves that the two skips still declared by the library
are honoured and honest: the corresponding check fails when called
directly on the estimator, so the skip hides a real limitation, not a
passing check.
"""

import pickle
import time

import pytest

from dense_armor.checks.check_estimator import check_estimator
from dense_armor.dynamics.urdf_dynamics import RigidBodyModel
from dense_armor.roles import Regressor
from dense_armor.utility.linear.glm import LinearRegression, LogisticRegression
from dense_armor.utility.linear.payload import PayloadEstimator
from dense_armor.utility.optim import SGD

URDF = "test/fixtures/urdf/panda.urdf"


class _BrokenRepr(Regressor):
    """__str__ does not match the class name."""

    def __str__(self) -> str:
        return "not-the-class-name"

    def learn_one(self, x, y, t=None):
        return self

    def predict_one(self, x, t=None, return_std=False):
        return 0.0


class _NonPicklable(Regressor):
    """__reduce__ raises PicklingError, so pickle cannot round-trip it."""

    def __reduce__(self):
        raise pickle.PicklingError("cannot serialize this estimator")

    def learn_one(self, x, y, t=None):
        return self

    def predict_one(self, x, t=None, return_std=False):
        return 0.0


class _BrokenStateDict(Regressor):
    """state_dict and load_state_dict do not round-trip."""

    def state_dict(self):
        return {"__unloadable__": 42}

    def load_state_dict(self, state):
        raise AssertionError("state dict rejected")

    def learn_one(self, x, y, t=None):
        return self

    def predict_one(self, x, t=None, return_std=False):
        return 0.0


class _CloneReturnsSelf(Regressor):
    """clone() returns the same object, so state is not independent."""

    def __init__(self) -> None:
        self._state_ = 0

    def clone(self, new_params=None, include_attributes=False):
        return self

    def learn_one(self, x, y, t=None):
        return self

    def predict_one(self, x, t=None, return_std=False):
        return 0.0


class _GrowingMemory(Regressor):
    """Declares O(1) but grows its reported memory by 1 MB per sample."""

    memory_class = "O(1)"

    def __init__(self) -> None:
        self._mem_ = 0

    @property
    def _raw_memory_usage(self):
        return self._mem_

    def learn_one(self, x, y=None, t=None):
        self._mem_ += 1_000_000
        return self

    def predict_one(self, x, t=None, return_std=False):
        return 0.0


class _SlowLearner(Regressor):
    """Declares a 1e-3 s budget but spends 5e-3 s per sample."""

    budget_s = 1e-3

    def learn_one(self, x, y=None, t=None):
        time.sleep(5e-3)
        return self

    def predict_one(self, x, t=None, return_std=False):
        return 0.0


def test_check_repr_rejects_wrong_repr():
    with pytest.raises(AssertionError, match="check_repr"):
        check_estimator(_BrokenRepr())


def test_check_pickle_roundtrip_rejects_lambda_state():
    with pytest.raises(AssertionError, match="check_pickle_roundtrip"):
        check_estimator(_NonPicklable())


def test_check_state_dict_roundtrip_rejects_broken_dict():
    with pytest.raises(AssertionError, match="check_state_dict_roundtrip"):
        check_estimator(_BrokenStateDict())


def test_check_clone_independent_rejects_shared_clone():
    with pytest.raises(AssertionError, match="check_clone_independent"):
        check_estimator(_CloneReturnsSelf())


def test_check_memory_growth_bounded_rejects_growing_estimator():
    with pytest.raises(AssertionError, match="check_memory_growth_bounded"):
        check_estimator(_GrowingMemory())


def test_check_p99_within_budget_rejects_slow_learner():
    with pytest.raises(AssertionError, match="check_p99_within_budget"):
        check_estimator(_SlowLearner())


def test_step_is_pure_skip_is_honoured_for_linear():
    from dense_armor.checks.robot import check_step_is_pure

    est = LinearRegression(optimizer=SGD(lr=0.1))
    assert "check_step_is_pure" in est._unit_test_skips()
    with pytest.raises(TypeError, match="scalar"):
        check_step_is_pure(est)


def test_step_is_pure_skip_is_honoured_for_logistic():
    from dense_armor.checks.robot import check_step_is_pure

    est = LogisticRegression(optimizer=SGD(lr=0.1))
    assert "check_step_is_pure" in est._unit_test_skips()
    with pytest.raises(TypeError, match="scalar"):
        check_step_is_pure(est)


def test_clone_skip_is_honoured_for_payload():
    from dense_armor.checks.check_estimator import check_clone

    est = PayloadEstimator(RigidBodyModel(URDF), link="panda_link7")
    assert "check_clone" in est._unit_test_skips()
    with pytest.raises(AssertionError):
        check_clone(est)


def test_repr_clone_equal_skip_is_honoured_for_payload():
    from dense_armor.checks.check_estimator import check_repr_clone_equal

    est = PayloadEstimator(RigidBodyModel(URDF), link="panda_link7")
    assert "check_repr_clone_equal" in est._unit_test_skips()
    with pytest.raises(AssertionError):
        check_repr_clone_equal(est)
