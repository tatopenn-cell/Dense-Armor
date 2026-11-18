"""ADWIN with ``max_window=None``: the window is cut only when a change is found."""

import numpy as np
import pytest

from dense_armor.core.online.drift.adwin import ADWIN


def _feed(det, values):
    flags = []
    for i, v in enumerate(values):
        det.update(float(v))
        if det.drift_detected:
            flags.append(i)
    return flags


def test_the_default_window_is_bounded_and_none_is_not():
    values = np.random.default_rng(0).normal(0.0, 1.0, 3000)
    bounded = ADWIN()
    free = ADWIN(max_window=None)
    _feed(bounded, values)
    _feed(free, values)
    assert bounded._n <= 200
    assert free._n > 1000


def test_the_unbounded_window_keeps_its_buckets_logarithmic():
    det = ADWIN(max_window=None)
    _feed(det, np.random.default_rng(1).normal(0.0, 1.0, 20000))
    assert sum(len(level) for level in det._buckets) < 120


def test_max_window_too_small_still_raises():
    with pytest.raises(ValueError):
        ADWIN(max_window=4, min_sub=2)
