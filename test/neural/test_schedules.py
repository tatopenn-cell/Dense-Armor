"""Tests for schedules.py."""

from __future__ import annotations

import pytest

from dense_armor.utility.neural.schedules import (
    ConstantLR,
    CosineWarmRestarts,
    StepLR,
)


def test_constant_lr_at_many_epochs():
    c = ConstantLR(0.1)
    for e in (0.0, 3.5, 100.0, 1e6):
        assert c(e) == 0.1


def test_constant_lr_multiplier_is_one():
    assert ConstantLR(0.1).multiplier(0.0) == 1.0
    assert ConstantLR(0.1).multiplier(42.0) == 1.0


def test_step_lr_hand_values():
    s = StepLR(0.1, 10, 0.5)
    assert s(9.99) == pytest.approx(0.1)
    assert s(10.0) == pytest.approx(0.05)
    assert s(25.0) == pytest.approx(0.025)
    assert s(100.0) == pytest.approx(0.1 * 0.5**10)


def test_cosine_t0_10_hand_values():
    c = CosineWarmRestarts(0.1, 0.0, 10)
    assert c(0.0) == pytest.approx(0.1)
    assert c(5.0) == pytest.approx(0.05)
    assert c(9.9) < 0.0025
    assert c(10.0) == pytest.approx(0.1)
    assert c(20.0) == pytest.approx(0.1)


def test_cosine_t_mult_2_hand_values():
    c = CosineWarmRestarts(0.1, 0.0, 1, t_mult=2)
    assert c(1.0) == pytest.approx(0.1)
    assert c(3.0) == pytest.approx(0.1)
    assert c(7.0) == pytest.approx(0.1)
    assert c(2.0) == pytest.approx(0.05)


def test_cosine_never_below_lr_min():
    c = CosineWarmRestarts(0.1, 0.02, 7)
    for e in [i / 10.0 for i in range(0, 200)]:
        assert c(e) >= 0.02 - 1e-12


def test_cosine_multiplier_at_zero_is_one():
    for c in [
        CosineWarmRestarts(0.1, 0.0, 10),
        CosineWarmRestarts(0.1, 0.02, 7),
        CosineWarmRestarts(0.1, 0.0, 1, t_mult=2),
    ]:
        assert c.multiplier(0.0) == pytest.approx(1.0)


def test_value_errors():
    with pytest.raises(ValueError):
        ConstantLR(0.0)
    with pytest.raises(ValueError):
        ConstantLR(-0.1)
    with pytest.raises(ValueError):
        ConstantLR(0.1)(-0.5)

    with pytest.raises(ValueError):
        StepLR(0.0, 10, 0.5)
    with pytest.raises(ValueError):
        StepLR(0.1, 0.0, 0.5)
    with pytest.raises(ValueError):
        StepLR(0.1, 10, 0.0)
    with pytest.raises(ValueError):
        StepLR(0.1, 10, 1.5)
    with pytest.raises(ValueError):
        StepLR(0.1, 10, 0.5)(-1.0)

    with pytest.raises(ValueError):
        CosineWarmRestarts(0.1, 0.0, 0.0)
    with pytest.raises(ValueError):
        CosineWarmRestarts(0.1, 0.0, 10, t_mult=0.0)
    with pytest.raises(ValueError):
        CosineWarmRestarts(0.0, 0.1, 10)
    with pytest.raises(ValueError):
        CosineWarmRestarts(0.1, 0.0, 10)(-0.5)


def test_cosine_at_the_restart_boundary_is_exact():
    c = CosineWarmRestarts(0.1, 0.0, 10)
    assert abs(c(10.0) - 0.1) < 1e-12
    assert abs(c(20.0) - 0.1) < 1e-12
    assert abs(c(30.0) - 0.1) < 1e-12


def test_cosine_t_mult_2_at_boundary_is_exact():
    c = CosineWarmRestarts(0.1, 0.0, 1, t_mult=2)
    for restart in (1.0, 3.0, 7.0, 15.0, 31.0):
        assert abs(c(restart) - 0.1) < 1e-12


def test_cosine_at_end_of_run_is_lr_min():
    c = CosineWarmRestarts(0.1, 0.01, 10)
    assert c(9.999) == pytest.approx(0.01, abs=1e-3)
    c2 = CosineWarmRestarts(0.1, 0.0, 10)
    assert c2(9.999) < 1e-6


def test_cosine_continuous_at_restart():
    c = CosineWarmRestarts(0.1, 0.0, 10)
    left = c(9.9999)
    right = c(10.0)
    assert left < right
    assert right == pytest.approx(0.1, abs=1e-9)


def test_step_lr_multiplier():
    s = StepLR(0.1, 10, 0.5)
    assert s.multiplier(0.0) == 1.0
    assert s.multiplier(10.0) == 0.5
    assert s.multiplier(25.0) == 0.25


def test_cosine_multiplier_with_zero_lr_max():
    c = CosineWarmRestarts(0.0, 0.0, 10)
    assert c.multiplier(1.0) == 0.0
    assert c.multiplier(5.0) == 0.0
    assert c(5.0) == 0.0
