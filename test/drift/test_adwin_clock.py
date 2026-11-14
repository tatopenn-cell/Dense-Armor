"""ADWIN: the default flags exactly where the previous version did, and ``clock`` delays the test."""

import numpy as np
import pytest

from dense_armor.core.online.drift.adwin import ADWIN


def _normal_steps():
    g = np.random.default_rng(0)
    return (
        list(g.normal(0, 1, 400)) + list(g.normal(3, 1, 400)) + list(g.normal(0, 2, 300))
    )


def _bernoulli_step():
    g = np.random.default_rng(9)
    return [float(v) for v in np.concatenate([g.random(1000) < 0.1, g.random(1000) < 0.5])]


def _flags(det, xs):
    out = []
    for i, x in enumerate(xs):
        det.update(x)
        if det.drift_detected:
            out.append(i)
    return out


def test_the_default_flags_where_the_previous_version_did():
    assert _flags(ADWIN(), _normal_steps()) == [444, 449, 839, 840, 890, 931]
    assert _flags(ADWIN(), _bernoulli_step()) == [1010, 1021, 1023, 1105]


def test_clock_one_is_the_default():
    assert ADWIN().clock == 1
    assert _flags(ADWIN(clock=1), _normal_steps()) == _flags(ADWIN(), _normal_steps())


@pytest.mark.parametrize("clock", [4, 16, 32])
def test_a_larger_clock_tests_only_every_clock_updates(clock):
    flags = _flags(ADWIN(clock=clock), _normal_steps())
    assert flags
    assert all((i + 1) % clock == 0 for i in flags)


@pytest.mark.parametrize("clock", [4, 16, 32])
def test_a_larger_clock_delays_the_first_flag_by_less_than_clock(clock):
    first = _flags(ADWIN(), _normal_steps())[0]
    delayed = _flags(ADWIN(clock=clock), _normal_steps())[0]
    assert 0 <= delayed - first < clock


def test_clock_below_one_is_rejected():
    with pytest.raises(ValueError, match="clock must be at least 1"):
        ADWIN(clock=0)


def test_reset_restarts_the_clock():
    det = ADWIN(clock=4)
    for x in _normal_steps()[:10]:
        det.update(x)
    det._reset()
    assert det._ticks == 0
