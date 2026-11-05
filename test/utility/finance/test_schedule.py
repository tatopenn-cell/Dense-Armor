"""Tests for the TWAP and VWAP schedules."""

import numpy as np
import pytest

from dense_armor.utility.finance.schedule import twap_schedule, vwap_schedule


def test_vwap_hand_case_from_the_item():
    """3.6 shares, two buckets, two sessions: bucket 1 = 1.6, bucket 2 = 2.0."""
    h = np.array([[1.0, 2.0], [4.0, 2.0]])
    out = vwap_schedule(3.6, h)
    assert out.shape == (2,)
    assert abs(float(out[0]) - 1.6) < 1e-12
    assert abs(float(out[1]) - 2.0) < 1e-12
    assert abs(float(out.sum()) - 3.6) < 1e-12


def test_constant_history_equals_twap():
    """Remark 3, PDF page 5: constant volume means TWAP."""
    h = np.full((3, 4), 2.5)
    out = vwap_schedule(10.0, h)
    ref = twap_schedule(10.0, 4)
    assert np.allclose(out, ref, atol=1e-12)


def test_twap_sums_to_total():
    for total, n in [(0.0, 1), (1.0, 3), (7.3, 5), (1e6, 17)]:
        s = twap_schedule(total, n)
        assert s.shape == (n,)
        assert abs(float(s.sum()) - total) < 1e-9


def test_vwap_sums_to_total():
    rng = np.random.default_rng(0)
    for _ in range(5):
        h = rng.uniform(0.1, 10.0, size=(4, 6))
        out = vwap_schedule(1.234, h)
        assert abs(float(out.sum()) - 1.234) < 1e-12


def test_arithmetic_mean_gives_a_different_schedule():
    """Two histories with the same arithmetic mean per bucket but a
    different harmonic mean per bucket must produce different schedules."""
    h1 = np.array([[1.0, 4.0], [9.0, 6.0]])
    h2 = np.array([[2.0, 2.0], [8.0, 8.0]])
    s1 = vwap_schedule(1.0, h1)
    s2 = vwap_schedule(1.0, h2)
    assert np.allclose(s1, s2) is False
    assert abs(float(s1[0]) - float(s2[0])) > 1e-6


def test_twap_errors():
    with pytest.raises(ValueError):
        twap_schedule(1.0, 0)
    with pytest.raises(ValueError):
        twap_schedule(-0.1, 3)


def test_vwap_errors():
    with pytest.raises(ValueError):
        vwap_schedule(-1.0, np.ones((1, 2)))
    with pytest.raises(ValueError):
        vwap_schedule(1.0, np.ones(3))
    with pytest.raises(ValueError):
        vwap_schedule(1.0, np.array([[1.0, np.nan]]))
    with pytest.raises(ValueError):
        vwap_schedule(1.0, np.array([[1.0, 0.0]]))
    with pytest.raises(ValueError):
        vwap_schedule(1.0, np.array([[1.0, -0.5]]))
