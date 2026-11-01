"""Tests for the linear scaling rule and the data-parallel step."""

from __future__ import annotations

import pytest

from dense_armor.utility.neural.data_parallel import linear_scaling
from dense_armor.utility.neural.schedules_more import LinearWarmup


def test_linear_scaling_hand_case():
    """Goyal 2017, page 2: a batch of 256 with a base of 32 scales lr by 8."""
    assert linear_scaling(0.1, 256, 32) == pytest.approx(0.8)


def test_linear_scaling_identity_at_base_batch():
    assert linear_scaling(0.05, 32, 32) == pytest.approx(0.05)


def test_linear_scaling_rejects_invalid_arguments():
    with pytest.raises(ValueError):
        linear_scaling(0.1, 256, 0)
    with pytest.raises(ValueError):
        linear_scaling(0.1, 16, 32)
    with pytest.raises(ValueError):
        linear_scaling(0.0, 256, 32)
    with pytest.raises(ValueError):
        linear_scaling(-0.1, 256, 32)


def test_linear_warmup_reaches_the_scaled_rate():
    """The warmup reaches the scaled learning rate at ``warmup`` epochs."""
    lr0 = 0.1
    lr_end = linear_scaling(lr0, 256, 32)
    s = LinearWarmup(lr_start=lr0, lr_end=lr_end, warmup=5)
    assert s(0.0) == pytest.approx(lr0)
    assert s(2.5) == pytest.approx((lr0 + lr_end) / 2.0)
    assert s(5.0) == pytest.approx(lr_end)
    assert s(7.0) == pytest.approx(lr_end)
