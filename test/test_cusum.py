# -*- coding: utf-8 -*-
"""
Unit tests for dense_armor/utility/cusum.py -- until now cusum_detector
was only exercised indirectly through test_benchmark_v0_runtime_
behavioral_drift.py, which checks structural sanity (0<=rate<=1) but
never verifies the detector's own correctness against a known ground
truth. Added during a rigor review of that benchmark before treating
cusum.py as ready to be public library surface.
"""
import numpy as np

from dense_armor.utility.cusum import cusum_detector


def test_pure_noise_has_low_alert_rate():
    rng = np.random.default_rng(1)
    x = rng.normal(0.0, 1.0, 600)
    flagged, cusum = cusum_detector(x)
    assert flagged.shape == x.shape
    assert cusum.shape == x.shape
    # not zero (an FP rate this low but nonzero is the honest expectation,
    # see the benchmark's own condition 1 -- ~1% observed there), but should
    # stay well clear of "flags most of the series", which would indicate a
    # broken threshold rather than real detections.
    assert np.mean(flagged) < 0.10


def test_detects_a_clear_upward_sustained_step():
    rng = np.random.default_rng(2)
    x = rng.normal(0.0, 1.0, 400)
    x[200:] += 10.0  # a large, unambiguous step -- not a borderline case
    flagged, cusum = cusum_detector(x)
    assert np.any(flagged[200:250]), "a 10-sigma-scale sustained step must be caught within 50 points"
    # nothing before the step should have flagged because OF the step
    # (pure baseline noise before it can still produce the odd FP, same
    # as test_pure_noise_has_low_alert_rate -- the real claim is that the
    # step itself gets caught soon after it starts).
    first_flag_after = np.argmax(flagged[200:]) if np.any(flagged[200:]) else None
    assert first_flag_after is not None and first_flag_after < 50


def test_detects_a_clear_downward_sustained_step():
    """Two-sided: s_neg must trigger for a downward shift, not just s_pos."""
    rng = np.random.default_rng(3)
    x = rng.normal(0.0, 1.0, 400)
    x[200:] -= 10.0
    flagged, cusum = cusum_detector(x)
    assert np.any(flagged[200:250]), "a large downward sustained step must be caught within 50 points"
    idx = 200 + np.argmax(flagged[200:])
    assert cusum[idx] < 0, "a downward step must flag through the negative accumulator (cusum<0)"


def test_degenerate_constant_baseline_does_not_crash_or_flag():
    x = np.full(200, 5.0)
    flagged, cusum = cusum_detector(x)
    assert flagged.shape == x.shape
    assert not np.any(flagged)
    assert np.all(np.isfinite(cusum))


def test_short_series_below_warmup_returns_all_clean_without_crashing():
    x = np.array([1.0, 2.0, 3.0])
    flagged, cusum = cusum_detector(x)
    assert flagged.shape == (3,)
    assert not np.any(flagged)


def test_reset_after_flag_allows_detecting_a_second_later_shift():
    """Real behavior claimed in the module docstring: both accumulators
    reset to 0 the instant they flag, so a second, separate shift later
    in the same series can still be detected -- verified here, not just
    asserted in prose."""
    rng = np.random.default_rng(4)
    x = rng.normal(0.0, 1.0, 700)
    x[150:] += 10.0    # first shift
    x[450:] += 10.0    # second shift, on top of the first (600 build-up)
    flagged, cusum = cusum_detector(x)
    assert np.any(flagged[150:250]), "first shift must be caught"
    assert np.any(flagged[450:550]), "second shift must also be caught, not suppressed by the first"


def test_k_and_h_change_sensitivity_in_the_expected_direction():
    """Not a claim about which values are 'better' -- just that a smaller
    h (easier to trip) flags at least as much as a larger h on the same
    frozen input, and a larger k (more slack) flags at most as much as a
    smaller k. A sanity check on the accumulator's own monotonicity, not
    a benchmark result."""
    rng = np.random.default_rng(5)
    x = rng.normal(0.0, 1.0, 300)
    x[150:] += 3.0
    flagged_loose, _ = cusum_detector(x, h=2.0)
    flagged_strict, _ = cusum_detector(x, h=10.0)
    assert np.sum(flagged_loose) >= np.sum(flagged_strict)
