# -*- coding: utf-8 -*-
"""
Unit tests for dense_armor/utility/one_sided.py -- previously only
exercised indirectly through test_benchmark_v2_1_ablation.py's real
Qwen telemetry analysis.
"""
import numpy as np
import pytest

from dense_armor.utility.one_sided import one_sided_upper_filter


def test_keeps_flags_above_median_drops_flags_below():
    x = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 10.0, -10.0, 1.0, 1.0, 1.0], dtype=float)
    flags = np.array([False] * 5 + [True, True] + [False] * 3)
    out = one_sided_upper_filter(x, flags, radius=2, ref_mult=1)
    assert out[5], "the high point's flag must survive (10.0 is above its causal median)"
    assert not out[6], "the low point's flag must be dropped (-10.0 is below its causal median)"


def test_never_introduces_a_new_flag():
    rng = np.random.default_rng(0)
    x = rng.normal(0.0, 1.0, 200)
    flags = rng.random(200) > 0.5  # arbitrary, not from a real detector -- just exercising the filter
    out = one_sided_upper_filter(x, flags, radius=5, ref_mult=2)
    assert np.all(out <= flags), "filter must only ever remove flags, never add one"


def test_all_false_input_stays_all_false():
    x = np.arange(50, dtype=float)
    flags = np.zeros(50, dtype=bool)
    out = one_sided_upper_filter(x, flags)
    assert not np.any(out)


def test_mismatched_shapes_raise():
    with pytest.raises(ValueError):
        one_sided_upper_filter(np.zeros(10), np.zeros(5, dtype=bool))


def test_real_detector_composition_smoke():
    """Composes with an actual detector (classify_segments), not just a
    synthetic flags array -- the real intended usage pattern."""
    from dense_armor.utility.arbiter import classify_segments

    rng = np.random.default_rng(1)
    x = rng.normal(2.0, 0.3, 200)
    x[100] = 20.0   # a large HIGH outlier -- should survive the filter
    x[150] = -20.0  # a large LOW outlier -- should be dropped by the filter
    labels, _, _ = classify_segments(x, radius=10, ref_mult=3)
    flags = labels != "clean"
    assert flags[100], "sanity: the high outlier must be flagged by classify_segments itself"
    assert flags[150], "sanity: the low outlier must be flagged by classify_segments itself"

    filtered = one_sided_upper_filter(x, flags, radius=10, ref_mult=3)
    assert filtered[100], "high outlier's flag must survive the one-sided filter"
    assert not filtered[150], "low outlier's flag must be dropped by the one-sided filter"
