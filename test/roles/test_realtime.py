"""Tests for dense_armor/roles/realtime.py."""
import doctest
import time

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import dense_armor.roles.realtime as rt_mod
from dense_armor.roles.realtime import (
    DEFAULT_WARMUP,
    ProfileResult,
    PureEW,
    RealtimePipeline,
    profile,
)
from dense_armor.roles.signal import Signal
from dense_armor.utility.stats.moments import RunningMoments


def _stream(n: int = 100):
    return [
        Signal(
            values=jnp.array([float(i)]),
            names=["x"],
            units=[""],
            t=i * 0.01,
        )
        for i in range(n)
    ]


class _Budgeted(RunningMoments):
    """RunningMoments with the robot contract declared."""

    budget_s = 1e-4
    memory_class = "O(1)"


def test_profile_returns_dataclass():
    est = RunningMoments()
    res = profile(est, _stream(50))
    assert isinstance(res, ProfileResult)
    assert res.n_samples == 50 - DEFAULT_WARMUP
    assert res.n_warmup == DEFAULT_WARMUP
    assert res.p50_s > 0
    assert res.p99_s >= res.p50_s
    assert res.max_s >= res.p99_s


def test_profile_fits_none_without_budget():
    est = RunningMoments()
    res = profile(est, _stream(30))
    assert res.budget_s is None
    assert res.fits is None
    assert res.memory_class == "unknown"


def test_profile_fits_with_budget():
    est = _Budgeted()
    res = profile(est, _stream(200))
    assert res.budget_s == 1e-4
    assert res.fits is not None
    assert res.memory_class == "O(1)"


def test_profile_memory_growth_is_bounded_for_running_moments():
    est = RunningMoments()
    res = profile(est, _stream(200))
    assert res.memory_growth_b < 4096


def test_realtime_pipeline_accepts_fitting_steps():
    pipe = RealtimePipeline(
        steps=[_Budgeted(), _Budgeted()],
        period_s=1e-3,
    )
    assert pipe.total_budget_s == pytest.approx(2e-4)
    assert pipe.headroom_s == pytest.approx(8e-4)


def test_realtime_pipeline_rejects_over_budget():
    with pytest.raises(ValueError, match="exceeds period"):
        RealtimePipeline(
            steps=[_Budgeted(), _Budgeted(), _Budgeted()],
            period_s=1e-4,
        )


def test_realtime_pipeline_rejects_missing_budget():
    with pytest.raises(ValueError, match="do not declare budget_s"):
        RealtimePipeline(
            steps=[RunningMoments()],
            period_s=1.0,
        )


def test_realtime_pipeline_allows_missing_budget_when_bypassed():
    pipe = RealtimePipeline(
        steps=[RunningMoments()],
        period_s=1.0,
        require_budget=False,
    )
    assert pipe.total_budget_s == 0.0


def test_realtime_pipeline_rejects_bad_period():
    with pytest.raises(ValueError, match="period_s"):
        RealtimePipeline(steps=[_Budgeted()], period_s=0.0)


def test_pure_ew_step_matches_python_loop():
    alpha = 0.1
    stream = jnp.linspace(0.0, 1.0, 50)
    ew = PureEW(alpha=alpha)
    means = ew.scan(stream)
    manual = []
    m = 0.0
    for i, v in enumerate(np.asarray(stream)):
        m = float(v) if i == 0 else m + alpha * (float(v) - m)
        manual.append(m)
    np.testing.assert_allclose(np.asarray(means), manual, rtol=1e-6)


def test_pure_ew_scan_is_jittable():
    stream = jnp.linspace(0.0, 1.0, 50)
    ew = PureEW(alpha=0.1)
    means_eager = ew.scan(stream)
    means_jit = jax.jit(ew.scan)(stream)
    np.testing.assert_allclose(
        np.asarray(means_eager), np.asarray(means_jit), rtol=1e-6
    )


def test_pure_ew_has_robot_contract():
    ew = PureEW()
    assert ew.budget_s is not None
    assert ew.memory_class == "O(1)"


def test_doctests():
    assert doctest.testmod(rt_mod).failed == 0
