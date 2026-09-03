# -*- coding: utf-8 -*-
"""
test/test_arl_theory.py
========================
Tests for the ARL (Average Run Length) theory functions promoted from
Dense-Evolution-Discovery's cusum_detectability_theory experiment:
one_sided_arl, two_sided_arl, detectability_report (utility/cusum.py).
"""
import numpy as np
import pytest

from dense_armor.utility.cusum import one_sided_arl, two_sided_arl, detectability_report


def test_one_sided_arl_matches_known_monte_carlo_value():
    """delta=0.5, h=5.0 -- Reynolds-corrected formula gives 10.34,
    matching a direct Monte Carlo simulation of the exact idealized
    process (10.26 at 3000 trials, see Discovery's
    validate_arl_theory.py) to within 1%."""
    assert one_sided_arl(delta=0.5, h=5.0, corrected=True) == pytest.approx(10.34, abs=0.05)


def test_one_sided_arl_delta_zero_uses_the_squared_boundary_formula():
    """delta=0 (mu exactly equals k) is a distinct closed-form branch
    (ARL = hh**2, the pure-random-walk boundary-crossing time), not
    covered by the delta!=0 cases above -- exercised directly here."""
    assert one_sided_arl(delta=0.0, h=5.0, corrected=True) == pytest.approx((5.0 + 1.166) ** 2)
    assert one_sided_arl(delta=0.0, h=5.0, corrected=False) == pytest.approx(25.0)


def test_uncorrected_formula_underestimates_at_small_h():
    """Reynolds' 1975 finding, reproduced: the boundary correction is
    not optional at realistic h -- the uncorrected value is ~20% low."""
    corrected = one_sided_arl(delta=0.5, h=5.0, corrected=True)
    uncorrected = one_sided_arl(delta=0.5, h=5.0, corrected=False)
    assert uncorrected < corrected * 0.85


def test_two_sided_arl_reproduces_experiment_44s_reference_value():
    assert two_sided_arl(mu=1.0, k=0.5, h=5.0) == pytest.approx(10.34, abs=0.05)


def test_detectability_report_reproduces_ad_hoc_sigma_ratio():
    report = detectability_report(local_noise_scale=7.72, k=0.5, h=5.0, candidate_shift=10.0)
    assert report["shift_in_sigma"] == pytest.approx(10.0 / 7.72)
    assert report["false_alarm_arl"] == pytest.approx(469.11, rel=0.01)
    assert report["detection_arl"] == pytest.approx(6.96, abs=0.05)


def test_detectability_report_floors_at_one_at_extreme_snr():
    """Real finding from a real accelerometer domain (Dense-Evolution-
    Discovery's validate_against_real_imu.py): at extreme shift_in_sigma
    (real local noise tiny relative to a real candidate shift), the raw
    formula predicts a fractional ARL -- floored at 1.0 here since a
    detector cannot flag in under one real sample."""
    report = detectability_report(local_noise_scale=0.001, k=0.5, h=5.0, candidate_shift=3.0)
    assert report["shift_in_sigma"] == pytest.approx(3000.0)
    assert report["detection_arl"] == 1.0


def test_false_alarm_arl_also_floored():
    report = detectability_report(local_noise_scale=1.0, k=0.5, h=0.01, candidate_shift=0.0)
    assert report["false_alarm_arl"] >= 1.0
