# -*- coding: utf-8 -*-
"""Unit tests for dense_armor/utility/drift/cusum.py.

cusum_detector was only exercised indirectly through
test_benchmark_v0_runtime_behavioral_drift.py before this file (that
test checks structural sanity, 0 <= rate <= 1, and never the detector's
own correctness). Tests below target cusum_detector directly.

Organized by the behavioral claim each test verifies, so a removed
guard, reset, or default produces a failing test rather than a silent
pass. Guard tests use `warnings.simplefilter("error")` because the
branches they cover exist to prevent a RuntimeWarning -- an output-only
check passes without them, since Python's max(0.0, nan) == 0.0 hides
the corruption.
"""
import warnings

import numpy as np
import pytest

from dense_armor.utility.drift.cusum import cusum_detector


# ---------------------------------------------------------------------------
# Core behavior
# ---------------------------------------------------------------------------
def test_pure_noise_low_point_level_alert_rate():
    rng = np.random.default_rng(1)
    x = rng.normal(0.0, 1.0, 400)
    flagged, cusum = cusum_detector(x)
    assert flagged.shape == x.shape
    assert cusum.shape == x.shape
    assert np.mean(flagged) < 0.10


def test_clear_upward_step_caught_within_50_points():
    rng = np.random.default_rng(2)
    x = rng.normal(0.0, 1.0, 400)
    x[200:] += 10.0
    flagged, _ = cusum_detector(x)
    assert np.any(flagged[200:250])


def test_clear_downward_step_flags_through_negative_accumulator():
    rng = np.random.default_rng(3)
    x = rng.normal(0.0, 1.0, 400)
    x[200:] -= 10.0
    flagged, cusum = cusum_detector(x)
    assert np.any(flagged[200:250])
    idx = 200 + int(np.argmax(flagged[200:]))
    assert cusum[idx] < 0


def test_invalid_reference_raises():
    with pytest.raises(ValueError):
        cusum_detector(np.zeros(50), reference="bogus")


# ---------------------------------------------------------------------------
# [guard] Branches inside the loop. Each one fails when its branch is
# removed: the corresponding RuntimeWarning becomes an exception, or the
# accumulator no longer resets.
# ---------------------------------------------------------------------------
def test_constant_series_does_not_flag_or_warn():
    """[guard scale<eps] Without `if not isfinite(scale) or scale < eps:
    continue`, a constant series computes 0/0 and numpy emits
    RuntimeWarning. Output stays clean either way (max(0.0, nan) == 0.0
    in Python), so only the warning check catches the removed branch."""
    x = np.full(200, 5.0)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        flagged, cusum = cusum_detector(x)
    assert not flagged.any()
    assert np.all(np.isfinite(cusum))


def test_short_series_does_not_warn():
    """[guard w_ref.size < 4] Without the warmup `continue`,
    np.median([]) runs on the empty first window and numpy emits
    RuntimeWarning. The next guard (isfinite(scale)) cleans up the
    resulting nan, so only the warning check catches the removed
    branch."""
    x = np.array([1.0, 2.0, 3.0])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        flagged, cusum = cusum_detector(x)
    assert flagged.shape == (3,)
    assert not flagged.any()


def test_nan_input_does_not_warn_and_shifts_first_flag():
    """[guard isfinite(x[i])] With the guard removed, z = nan and Python's
    max(0.0, nan) resets the accumulator to 0 -- the NaN silently
    delays the first flag. Without the guard the flag index moves; with
    it the flag index is unchanged from the clean run."""
    rng = np.random.default_rng(0)
    clean = rng.normal(0.0, 0.1, 200)
    clean[50:] += 0.3
    nan = clean.copy()
    nan[70] = np.nan

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        f_clean, _ = cusum_detector(clean, radius=10, ref_mult=3,
                                    reference="fixed", h=100.0, k=0.5)
        f_nan, _ = cusum_detector(nan, radius=10, ref_mult=3,
                                  reference="fixed", h=100.0, k=0.5)
    assert f_clean.any() and f_nan.any()
    assert int(np.argmax(f_clean)) == int(np.argmax(f_nan))


def test_inf_input_does_not_warn_or_propagate():
    """[guard isfinite(x[i])] Without the guard, +inf flows into s_pos
    and flags the current point with a corrupted accumulator."""
    rng = np.random.default_rng(0)
    clean = rng.normal(0.0, 0.1, 200)
    clean[50:] += 0.3
    inf = clean.copy()
    inf[70] = np.inf

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        f_clean, _ = cusum_detector(clean, radius=10, ref_mult=3,
                                    reference="fixed", h=100.0, k=0.5)
        f_inf, c_inf = cusum_detector(inf, radius=10, ref_mult=3,
                                      reference="fixed", h=100.0, k=0.5)
    assert np.all(np.isfinite(c_inf))
    assert int(np.argmax(f_clean)) == int(np.argmax(f_inf))


# ---------------------------------------------------------------------------
# [guard] Reset on both accumulator branches. Without an instantaneous
# reset, |cusum| grows with the length of a sustained shift instead of
# being bounded near h.
# ---------------------------------------------------------------------------
def test_positive_reset_bounds_accumulator():
    """[guard s_pos = 0.0 after flag]"""
    rng = np.random.default_rng(0)
    base = rng.normal(0.0, 0.1, 400)

    def _peak(shift_len):
        x = base.copy()
        x[100:100 + shift_len] = 5.0
        _, c = cusum_detector(x, radius=10, ref_mult=1,
                              reference="fixed", h=5.0, k=0.5)
        return float(np.max(c[100:100 + shift_len]))

    assert abs(_peak(100) - _peak(10)) < 5.0


def test_negative_reset_bounds_accumulator():
    """[guard s_neg = 0.0 after flag] Symmetric to the positive test;
    the positive one never exercises s_neg, so removing the negative
    reset passes it unnoticed."""
    rng = np.random.default_rng(0)
    base = rng.normal(0.0, 0.1, 400)

    def _trough(shift_len):
        x = base.copy()
        x[100:100 + shift_len] = -5.0
        _, c = cusum_detector(x, radius=10, ref_mult=1,
                              reference="fixed", h=5.0, k=0.5)
        return float(np.min(c[100:100 + shift_len]))

    assert abs(_trough(100) - _trough(10)) < 5.0


def test_reset_allows_a_second_later_shift():
    """The reset is what lets a second, later shift be detected after the
    first one flagged."""
    rng = np.random.default_rng(4)
    x = rng.normal(0.0, 1.0, 500)
    x[100:] += 10.0
    x[350:] += 10.0
    flagged, _ = cusum_detector(x)
    assert flagged[100:200].any()
    assert flagged[350:450].any()


# ---------------------------------------------------------------------------
# Parameter behavior
# ---------------------------------------------------------------------------
def test_larger_k_flags_at_most_as_much_as_smaller_k():
    rng = np.random.default_rng(5)
    x = rng.normal(0.0, 1.0, 400)
    x[100:200] += 1.5
    x[250:350] -= 1.5
    small_k, _ = cusum_detector(x, h=20.0, k=0.1)
    large_k, _ = cusum_detector(x, h=20.0, k=2.0)
    assert small_k.sum() >= large_k.sum()


def test_larger_h_flags_at_most_as_much_as_smaller_h():
    rng = np.random.default_rng(5)
    x = rng.normal(0.0, 1.0, 300)
    x[150:] += 3.0
    loose, _ = cusum_detector(x, h=2.0)
    strict, _ = cusum_detector(x, h=10.0)
    assert loose.sum() >= strict.sum()


def test_fixed_reference_re_flags_permanent_shift_more_than_adaptive():
    """Documented difference between the two reference modes: a fixed
    target never updates, so a permanent shift keeps re-tripping the
    accumulator; an adaptive target catches up and goes quiet."""
    rng = np.random.default_rng(2)
    x = rng.normal(0.0, 1.0, 400)
    x[200:] += 10.0
    f_fixed, _ = cusum_detector(x, reference="fixed")
    f_adaptive, _ = cusum_detector(x, reference="adaptive")
    assert f_fixed[200:].sum() > f_adaptive[200:].sum()


def test_fixed_reference_warmup_before_span_is_never_flagged():
    rng = np.random.default_rng(9)
    x = rng.normal(0.0, 1.0, 100)
    flagged, cusum = cusum_detector(x, radius=10, ref_mult=3,
                                    reference="fixed")
    span = 10 * 3
    assert not flagged[:span].any()
    assert not cusum[:span].any()


# ---------------------------------------------------------------------------
# Regression on the shipped default h. h=5.0 gave 100 % (adaptive) /
# 85 % (fixed) stream-level FA rate on a stable 1000-sample series; the
# shipped h=20.0 gives 3.5 % / 15.5 % over 200 trials (see cusum.py
# docstring). Kept to 20 trials for CI runtime: the property asserted is
# qualitative (rate clearly below the broken default's), not a
# reproduction of the calibration sweep.
# ---------------------------------------------------------------------------
def test_default_h_stream_level_false_alarm_rate_is_low():
    n_trials = 20
    for mode in ("adaptive", "fixed"):
        rng = np.random.default_rng(42)
        n_with_flag = 0
        for _ in range(n_trials):
            x = rng.normal(0.0, 1.0, 800)
            flagged, _ = cusum_detector(x, radius=10, ref_mult=3,
                                        k=0.5, reference=mode)
            if flagged.any():
                n_with_flag += 1
        rate = n_with_flag / n_trials
        assert rate < 0.40, (
            f"reference={mode!r}: stream-level FA rate {rate:.2f} over "
            f"{n_trials} trials -- default h may have regressed"
        )
