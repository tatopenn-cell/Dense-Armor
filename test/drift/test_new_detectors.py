"""Tests for the four drift detectors and the CUSUM fixed-reference fix."""

import numpy as np
import pytest

from dense_armor.checks import check_estimator
from dense_armor.utility.drift.adwin import ADWIN
from dense_armor.utility.drift.detector import CUSUMDriftDetector
from dense_armor.utility.drift.kswin import KSWIN
from dense_armor.utility.drift.page_hinkley import PageHinkley


def _stream(n0, mu0, n1, mu1, sigma=1.0, seed=0):
    rng = np.random.default_rng(seed)
    return list(rng.normal(mu0, sigma, n0)) + list(rng.normal(mu1, sigma, n1))


def _first_alarm(det, stream):
    for i, x in enumerate(stream):
        det.update(x)
        if det.drift_detected:
            return i
    return None


def _n_alarms(det, stream):
    n = 0
    for x in stream:
        det.update(x)
        if det.drift_detected:
            n += 1
    return n


def test_cusum_fixed_reference_with_nan_prefix():
    det = CUSUMDriftDetector(reference="fixed", radius=5, ref_mult=4)
    stream = [float("nan")] * 10 + _stream(200, 0.0, 200, 3.0, seed=11)
    i = _first_alarm(det, stream)
    assert i is not None, "detector must fire after the step"
    assert det.n_missing == 10


def test_cusum_fixed_reference_with_flat_prefix():
    det = CUSUMDriftDetector(reference="fixed", radius=5, ref_mult=4)
    stream = [0.0] * 40 + [5.0] * 200
    i = _first_alarm(det, stream)
    assert i is not None, "detector must fire after the flat-then-step"
    assert i >= 40


def test_cusum_adaptive_quiet_on_stationary():
    # The detector is an adaptive-reference CUSUM, not the idealised
    # fixed-reference CUSUM of Reynolds 1975 (Technometrics 17(1),
    # Sec. 3): the reference itself drifts with a causal window, the
    # accumulator state is not guaranteed to be zero at an arbitrary
    # point, and the reset on flag is asymmetric. The paper's ARL
    # formula therefore gives an order-of-magnitude expectation for the
    # false-alarm count, not a value the detector is guaranteed to
    # satisfy. At the module's h = 20 the idealised ARL_0 is in the
    # hundreds of samples, so over n = 10 000 the expected point-level
    # false-alarm count is in the tens. The bound 40 holds that order
    # of magnitude; it is a regression check (a broken detector would
    # blow past it), not a verification of the paper's formula.
    det = CUSUMDriftDetector(reference="adaptive")
    n = 10000
    stream = _stream(n, 0.0, 0, 0.0, seed=1)
    fa = _n_alarms(det, stream)
    assert fa <= 40, f"adaptive: {fa} alarms in {n} stationary samples"


def test_cusum_two_sided_catches_downward_step():
    det = CUSUMDriftDetector(reference="fixed", radius=5, ref_mult=4)
    assert _first_alarm(det, _stream(200, 0.0, 200, -3.0, seed=6)) is not None


def test_ph_detects_mean_step():
    det = PageHinkley(delta=0.05, threshold=20.0)
    stream = _stream(500, 0.0, 500, 1.5, seed=12)
    assert _first_alarm(det, stream) is not None


def test_ph_quiet_on_stationary():
    # Same caveat as the CUSUM case above: PageHinkley is the one-sided
    # form of Page's CUSUM, and the paper's ARL formula describes the
    # idealised detector, not this implementation. The bound 40 is an
    # order-of-magnitude regression check at the module's own (k, h),
    # matching the CUSUM test so the two are held to a comparable
    # standard.
    det = PageHinkley(delta=0.05, threshold=20.0)
    n = 10000
    stream = _stream(n, 0.0, 0, 0.0, seed=2)
    fa = _n_alarms(det, stream)
    assert fa <= 40, f"PH: {fa} alarms in {n} stationary samples"


def test_ph_scale_invariant():
    base = _stream(500, 0.0, 500, 3.0, seed=21)
    a = PageHinkley(delta=0.05, threshold=20.0)
    b = PageHinkley(delta=0.05, threshold=20.0)
    i_a = _first_alarm(a, base)
    i_b = _first_alarm(b, [1e-4 * x for x in base])
    assert i_a is not None and i_b is not None
    assert i_a == i_b


def test_ph_nan_skipped():
    det = PageHinkley(delta=0.05, threshold=20.0)
    for x in [1.0, float("nan"), 2.0, float("inf")]:
        det.update(x)
    assert det.n_missing == 2


def test_adwin_detects_mean_step():
    det = ADWIN(delta=0.002, max_window=100)
    stream = _stream(300, 0.0, 300, 4.0, seed=13)
    assert _first_alarm(det, stream) is not None


def test_adwin_quiet_on_stationary():
    # ADWIN's delta is a theoretical bound, not an empirical choice:
    # Bifet & Gavalda 2007 (SIAM SDM) prove that both the false
    # positive and false negative rates are bounded by delta. Over
    # n = 10 000 with delta = 0.002, the expected number of false
    # alarms is at most n * delta = 20. Unlike the CUSUM and PH cases
    # above, this bound is exact for the detector's own configuration,
    # not an order-of-magnitude estimate for an idealised algorithm.
    det = ADWIN(delta=0.002, max_window=200)
    n = 10000
    rng = np.random.default_rng(3)
    stream = list(rng.normal(0.0, 1.0, n))
    fa = _n_alarms(det, stream)
    assert fa <= 20, f"ADWIN: {fa} alarms in {n} stationary samples (delta=0.002)"


def test_adwin_scale_invariant():
    base = _stream(500, 0.0, 500, 3.0, seed=22)
    a = ADWIN(delta=0.002, max_window=200)
    b = ADWIN(delta=0.002, max_window=200)
    i_a = _first_alarm(a, base)
    i_b = _first_alarm(b, [1e-4 * x for x in base])
    assert i_a is not None and i_b is not None
    assert i_a == i_b


def test_adwin_flat_window_does_not_fire():
    assert _n_alarms(ADWIN(), [2.0] * 300) == 0


def test_kswin_detects_mean_step():
    det = KSWIN(seed=0)
    stream = _stream(400, 0.0, 400, 2.5, seed=14)
    assert _first_alarm(det, stream) is not None


def test_kswin_single_alarm_per_change():
    det = KSWIN(seed=0)
    stream = _stream(400, 0.0, 400, 2.5, seed=14)
    n = _n_alarms(det, stream)
    assert n == 1, f"KSWIN: {n} alarms for one change"


def test_kswin_quiet_on_stationary(capsys):
    # Bound derivation. KSWIN (Raab, Heusinger, Schleif 2020,
    # arXiv:2007.05432, Sec. 4.1 and Sec. 5.2) runs a per-dimension KS
    # test every step with a Bonferroni-Dunn-corrected significance
    # level alpha* = 0.05/60 = 0.0001 and window r = 30. Over n = 5000
    # samples the paper's own parameter choice predicts on the order of
    # n * alpha* = 0.5 false alarms, not the dozens a naive uncorrected
    # alpha = 0.01 would give. The mean over 3 seeds is used here
    # instead of a single seed, so the test is not tied to one lucky or
    # unlucky draw. Bound 3 is several times the expected order of
    # magnitude. The 3 x 5000 configuration keeps the test at a few
    # seconds in CI: 10 x 10000 measured 45 s on the development
    # environment, too slow for the fast page.
    n_seeds = 3
    n = 5000
    counts = []
    for seed in range(n_seeds):
        rng = np.random.default_rng(seed)
        stream = list(rng.normal(0.0, 1.0, n))
        det = KSWIN(seed=0)
        fa = 0
        for v in stream:
            det.update(v)
            if det.drift_detected:
                fa += 1
        counts.append(fa)
    mean_fa = float(np.mean(counts))
    with capsys.disabled():
        print(f"[kswin/stationary] mean FA over {n_seeds} seeds of "
              f"{n} samples={mean_fa:.2f}, per-seed counts={counts}")
    assert mean_fa <= 3, (
        f"KSWIN: mean {mean_fa:.2f} alarms over {n_seeds} seeds"
    )


def test_kswin_does_not_fire_with_window_40_stat_30():
    # window_size=40, stat_size=30 leaves only 10 points of "reference"
    # against 30 of "test". Under that split KSWIN emits no alarm on this
    # 0 -> 5 step, with or without noise around the step. The test pins
    # that behavior; the name describes the parameters, not a claim
    # about "small pools" in general.
    det = KSWIN(window_size=40, stat_size=30, seed=0)
    assert _n_alarms(det, _stream(100, 0.0, 100, 5.0, seed=5)) == 0


@pytest.mark.parametrize("cls", [CUSUMDriftDetector, PageHinkley, ADWIN, KSWIN])
def test_check_estimator(cls):
    check_estimator(cls())


@pytest.mark.parametrize("cls", [CUSUMDriftDetector, PageHinkley, ADWIN, KSWIN])
def test_reset_and_nan_counting(cls):
    det = cls()
    for x in [1.0, float("nan"), 2.0]:
        det.update(x)
    assert det.n_missing == 1
    det._reset()
    assert det.n_missing == 0
    assert not det.drift_detected


def test_invalid_arguments():
    with pytest.raises(ValueError):
        CUSUMDriftDetector(reference="other")
    with pytest.raises(ValueError):
        ADWIN(delta=1.5)
    with pytest.raises(ValueError):
        ADWIN(max_window=4, min_sub=2)
    with pytest.raises(ValueError):
        KSWIN(window_size=30, stat_size=30)


def test_doctests():
    import doctest

    import dense_armor.utility.drift.adwin as m1
    import dense_armor.utility.drift.detector as m2
    import dense_armor.utility.drift.kswin as m3
    import dense_armor.utility.drift.page_hinkley as m4

    for mod in (m1, m2, m3, m4):
        assert doctest.testmod(mod).failed == 0

