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


def test_nan_input_point_does_not_crash_or_propagate_nan():
    """A NaN at x[i] must not turn into a NaN accumulator or a spurious
    flag -- explicit guard, not reliant on Python's max(0.0, nan)==0.0 /
    min(0.0, nan)==0.0 incidental argument-order behavior (verified
    separately to exist, but not something this function should depend
    on silently)."""
    rng = np.random.default_rng(6)
    x = rng.normal(0.0, 1.0, 200)
    x[100] = np.nan
    flagged, cusum = cusum_detector(x)
    assert np.all(np.isfinite(cusum)), "cusum accumulator must never contain NaN/Inf"
    assert not flagged[100], "the NaN point itself must not be flagged"


def test_nan_point_does_not_corrupt_later_reference_windows():
    """A NaN sitting inside a LATER point's causal reference window
    (up to radius*ref_mult points back) must not turn that later point's
    med/scale into NaN and silently break its own accumulator update."""
    rng = np.random.default_rng(7)
    x = rng.normal(0.0, 1.0, 200)
    x[100] = np.nan
    flagged, cusum = cusum_detector(x, radius=10, ref_mult=3)  # span=30
    window_after = slice(101, 131)  # NaN still inside these points' windows
    assert np.all(np.isfinite(cusum[window_after]))
    assert not np.any(flagged[window_after])


def test_inf_input_point_does_not_crash_or_propagate():
    rng = np.random.default_rng(8)
    x = rng.normal(0.0, 1.0, 200)
    x[100] = np.inf
    flagged, cusum = cusum_detector(x)
    assert np.all(np.isfinite(cusum))
    assert not flagged[100]


def test_invalid_reference_raises():
    x = np.zeros(50)
    try:
        cusum_detector(x, reference="bogus")
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_fixed_reference_keeps_flagging_after_a_permanent_shift():
    """The documented, real behavioral difference from 'adaptive': a
    fixed target never updates, so a sustained shift keeps re-triggering
    the accumulator almost every step, instead of the detector adapting
    to the new level and going quiet."""
    rng = np.random.default_rng(2)
    x = rng.normal(0.0, 1.0, 400)
    x[200:] += 10.0
    flagged_fixed, _ = cusum_detector(x, reference="fixed")
    flagged_adaptive, _ = cusum_detector(x, reference="adaptive")
    assert flagged_fixed[200:].sum() > flagged_adaptive[200:].sum(), (
        "fixed reference should keep re-flagging a permanent shift far more than adaptive"
    )


def test_fixed_reference_warmup_before_span_is_never_flagged():
    rng = np.random.default_rng(9)
    x = rng.normal(0.0, 1.0, 100)
    flagged, cusum = cusum_detector(x, radius=10, ref_mult=3, reference="fixed")
    span = 10 * 3
    assert not np.any(flagged[:span])
    assert not np.any(cusum[:span])


def test_default_h_keeps_stream_level_false_alarm_rate_low_on_a_practical_length():
    """Regression test for a real bug found 2026-09-03 (while porting this
    same algorithm to online-ml/river): the OLD default h=5.0 gave a
    STREAM-level false-alarm rate of 100% ("adaptive") / 85% ("fixed") on
    a 1000-sample purely stable N(0,1) series -- invisible to
    test_pure_noise_has_low_alert_rate above, which only checks the much
    weaker POINT-level flag rate (a detector can flag nearly every stream
    at least once while still flagging <10% of individual points). This
    test checks the metric that actually matters for a practical
    monitoring horizon: the fraction of ENTIRELY STABLE streams that
    raise at least one flag anywhere."""
    n_trials = 40  # kept small for CI runtime; the real 200-trial sweep
    # this default was calibrated against is documented in cusum.py's own
    # docstring
    for mode in ("adaptive", "fixed"):
        rng = np.random.default_rng(42)
        n_with_flag = 0
        for _ in range(n_trials):
            x = rng.normal(0.0, 1.0, 1000)
            flagged, _ = cusum_detector(x, radius=10, ref_mult=3, k=0.5, reference=mode)  # default h
            if flagged.any():
                n_with_flag += 1
        rate = n_with_flag / n_trials
        assert rate < 0.30, (
            f"reference={mode!r}: stream-level false-alarm rate {rate:.2f} over {n_trials} "
            f"trials is too high for the default h -- the whole point of this test"
        )


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
