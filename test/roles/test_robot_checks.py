"""Tests for dense_armor/checks/robot.py."""
import pickle
from pathlib import Path

import jax.numpy as jnp
import pytest

from dense_armor.checks import robot


class _Step:
    def __init__(self, pure=True):
        self.pure = pure
        self.calls = 0

    def init_state(self):
        return jnp.array(0.0)

    def step(self, state, v):
        self.calls += 1
        k = 0.0 if self.pure else float(self.calls)
        return state + v + k, v * 2.0 + k


class _Learner:
    budget_s = 1.0
    memory_class = "O(1)"

    def __init__(self):
        self.n = 0
        self._raw_memory_usage = 0

    def learn_one(self, x):
        self.n += 1


class _Grow:
    memory_class = "O(1)"

    def __init__(self):
        self.buf = []

    @property
    def _raw_memory_usage(self):
        return 1000 * len(self.buf)

    def learn_one(self, x):
        self.buf.append(x)


class _Est:
    def __init__(self, var):
        self.var = var


class _Pred:
    def __init__(self, var):
        self.v = var

    def predict_one(self, x, return_estimate=False):
        return _Est(self.v)


class _Ckpt:
    def __init__(self, check=True):
        self.n = 0
        self.check = check

    def learn_one(self, x, y):
        self.n += 1

    def clone(self):
        return _Ckpt(self.check)

    def save(self, p):
        Path(p).write_bytes(pickle.dumps(self.n) + b"\x01" * 16)

    def restore(self, p):
        if self.check and not Path(p).read_bytes().endswith(b"\x01" * 8):
            raise ValueError("corrupted")


def test_step_checks_pass_on_pure_step():
    robot.check_step_is_pure(_Step())
    robot.check_step_is_jittable(_Step())


def test_step_is_pure_detects_impure_step():
    with pytest.raises(AssertionError, match="not pure"):
        robot.check_step_is_pure(_Step(pure=False))


def test_step_checks_skip_without_step():
    robot.check_step_is_pure(object())
    robot.check_step_is_jittable(object())


def test_learn_passes_target_when_accepted():
    got = []

    class _Y:
        def learn_one(self, x, y):
            got.append(y)

    robot._learn(_Y(), {"x": 1.0})
    assert got == [0.0]


def test_p99_within_budget_passes():
    est = _Learner()
    robot.check_p99_within_budget(est)
    assert est.n == 32


def test_p99_within_budget_fails_on_zero_budget():
    est = _Learner()
    est.budget_s = 0.0
    est.learn_one = lambda x: sum(range(10000))
    with pytest.raises(AssertionError, match="exceeds"):
        robot.check_p99_within_budget(est)


def test_p99_skips_without_budget_or_learn():
    class _B:
        budget_s = 1.0

    robot.check_p99_within_budget(object())
    robot.check_p99_within_budget(_B())


def test_memory_growth_passes_when_flat():
    robot.check_memory_growth_bounded(_Learner())


def test_memory_growth_fails_when_growing():
    with pytest.raises(AssertionError, match="memory grew"):
        robot.check_memory_growth_bounded(_Grow())


def test_memory_growth_skips():
    class _M:
        memory_class = "O(1)"

    class _N:
        memory_class = "O(1)"

        def learn_one(self, x):
            pass

    robot.check_memory_growth_bounded(object())
    robot.check_memory_growth_bounded(_M())
    robot.check_memory_growth_bounded(_N())


def test_variance_check():
    robot.check_estimate_variance_non_negative(_Pred(0.5))
    with pytest.raises(AssertionError, match="negative"):
        robot.check_estimate_variance_non_negative(_Pred(-1.0))


def test_variance_check_skips():
    class _NoKw:
        def predict_one(self, x):
            return 0.0

    class _Plain:
        def predict_one(self, x, return_estimate=False):
            return 0.0

    robot.check_estimate_variance_non_negative(object())
    robot.check_estimate_variance_non_negative(_NoKw())
    robot.check_estimate_variance_non_negative(_Plain())


def test_health_check():
    class _H:
        health = "nominal"

    class _E:
        health = ""

    robot.check_health_is_reachable(_H())
    robot.check_health_is_reachable(object())
    with pytest.raises(AssertionError, match="empty"):
        robot.check_health_is_reachable(_E())


def test_checkpoint_roundtrip_passes():
    robot.check_checkpoint_roundtrip(_Ckpt())


def test_checkpoint_roundtrip_detects_accepted_corruption():
    with pytest.raises(AssertionError, match="corrupted"):
        robot.check_checkpoint_roundtrip(_Ckpt(check=False))


def test_checkpoint_roundtrip_skips():
    class _Bad(_Ckpt):
        def learn_one(self, x, y, z):
            pass

    robot.check_checkpoint_roundtrip(object())
    robot.check_checkpoint_roundtrip(_Bad())


def test_schema_json_runs_on_estimator():
    from dense_armor.roles import Regressor

    class _R(Regressor):
        def __init__(self, a=1.0):
            self.a = a

        def learn_one(self, x, y):
            return self

        def predict_one(self, x):
            return 0.0

    robot.check_schema_json(_R())
