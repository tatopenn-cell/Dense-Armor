"""Tests for the flash crash generator."""

import numpy as np
import pytest

from pipelines.finance.stress import (
    FlashCrashResult,
    crash_metrics,
    synthetic_flash_crash,
)


def test_deterministic_for_a_seed():
    a = synthetic_flash_crash(n=200, seed=7).prices
    b = synthetic_flash_crash(n=200, seed=7).prices
    np.testing.assert_array_equal(a, b)


def test_different_seed_different_path():
    a = synthetic_flash_crash(n=200, seed=1).prices
    b = synthetic_flash_crash(n=200, seed=2).prices
    assert not np.array_equal(a, b)


def test_drop_and_recovery_have_the_planted_size():
    r = synthetic_flash_crash(n=300, seed=0, crash_at=100,
                              duration_ticks=5, drop=0.05,
                              recovery_fraction=0.6, vol=0.0)
    m = crash_metrics(r)
    assert abs(m["realised_drop"] - 0.05) < 1e-9
    assert abs(m["realised_recovery"] - 0.6) < 1e-9


def test_result_dataclass_fields():
    r = synthetic_flash_crash(n=100, seed=0, crash_at=50, drop=0.03,
                              recovery_fraction=0.4)
    assert isinstance(r, FlashCrashResult)
    assert r.n_ticks == 100
    assert r.crash_at == 50
    assert r.drop == 0.03
    assert r.recovery_fraction == 0.4
    assert len(r.prices) == 100


def test_bad_drop():
    with pytest.raises(ValueError, match="drop"):
        synthetic_flash_crash(drop=-0.01)


def test_bad_recovery():
    with pytest.raises(ValueError, match="recovery_fraction"):
        synthetic_flash_crash(recovery_fraction=1.5)


def test_bad_duration():
    with pytest.raises(ValueError, match="duration_ticks"):
        synthetic_flash_crash(duration_ticks=0)


def test_crash_near_the_end_does_not_overflow():
    r = synthetic_flash_crash(n=10, seed=0, crash_at=8,
                              duration_ticks=5, drop=0.1)
    assert len(r.prices) == 10
    m = crash_metrics(r)
    assert np.isfinite(m["realised_drop"])
