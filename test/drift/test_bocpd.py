"""Tests for the Bayesian online changepoint detector (Adams and MacKay 2007)."""

import numpy as np
import pytest
from scipy import stats

from dense_armor.utility.drift.bocpd import BayesianOnlineChangepoint


def _brute_force(xs, lam, mu0, kappa0, alpha0, beta0):
    """The same recursion, without truncation, on a stream shorter than
    max_run_length. Returns the run length distribution after each step."""
    r = np.zeros(1)
    r[0] = 1.0
    mu = np.full(1, mu0)
    kappa = np.full(1, kappa0)
    alpha = np.full(1, alpha0)
    beta = np.full(1, beta0)
    out = []
    h = 1.0 / lam
    for x in xs:
        nu = 2.0 * alpha
        scale = np.sqrt(beta * (kappa + 1.0) / (alpha * kappa))
        pi = stats.t.pdf(x, df=nu, loc=mu, scale=scale)
        pi = np.maximum(pi, 1e-300)
        growth = r * pi * (1.0 - h)
        cp = float(np.sum(r * pi * h))
        r_new = np.empty(len(growth) + 1)
        r_new[0] = cp
        r_new[1:] = growth
        kappa_new = kappa + 1.0
        mu_new = (kappa * mu + x) / kappa_new
        alpha_new = alpha + 0.5
        beta_new = beta + 0.5 * kappa * (x - mu) ** 2 / kappa_new
        mu = np.concatenate([[mu0], mu_new])
        kappa = np.concatenate([[kappa0], kappa_new])
        alpha = np.concatenate([[alpha0], alpha_new])
        beta = np.concatenate([[beta0], beta_new])
        total = r_new.sum()
        if total > 0:
            r_new /= total
        r = r_new
        out.append(r.copy())
    return out


def test_matches_brute_force_without_truncation():
    rng = np.random.default_rng(0)
    xs = list(rng.normal(0.0, 1.0, 30))
    det = BayesianOnlineChangepoint(
        lam=100.0, max_run_length=100, mass_window=10)
    got = []
    for x in xs:
        det.update(x)
        got.append(det.run_length_.copy())
    ref = _brute_force(xs, 100.0, 0.0, 0.01, 0.01, 0.01)
    for t, (a, b) in enumerate(zip(got, ref)):
        assert a.shape == b.shape, f"step {t}: shapes {a.shape} vs {b.shape}"
        np.testing.assert_allclose(a, b, atol=1e-9,
                                   err_msg=f"step {t}")


def test_truncation_keeps_max_run_length_plus_one():
    det = BayesianOnlineChangepoint(lam=100.0, max_run_length=5)
    rng = np.random.default_rng(0)
    for x in rng.normal(0.0, 1.0, 20):
        det.update(x)
        assert len(det.run_length_) <= 6
    assert len(det.run_length_) == 6


def test_detects_mean_shift():
    rng = np.random.default_rng(0)
    stream = list(rng.normal(0.0, 0.1, 200)) + list(rng.normal(1.0, 0.1, 200))
    det = BayesianOnlineChangepoint(lam=100.0)
    fired = []
    for i, x in enumerate(stream):
        det.update(x)
        if det.drift_detected:
            fired.append(i)
    after = [i for i in fired if i >= 20]
    assert len(after) >= 1
    assert after[0] - 200 < 5, f"delay {after[0] - 200}"
    assert det.n_alarms_ >= 1


def test_detects_variance_shift():
    rng = np.random.default_rng(1)
    stream = list(rng.normal(0.0, 0.1, 200)) + list(rng.normal(0.0, 1.0, 200))
    det = BayesianOnlineChangepoint(lam=100.0)
    fired = []
    for i, x in enumerate(stream):
        det.update(x)
        if det.drift_detected:
            fired.append(i)
    after = [i for i in fired if i >= 20]
    assert len(after) >= 1
    assert after[0] - 200 < 5, f"delay {after[0] - 200}"


def test_no_alarm_on_stationary_stream():
    rng = np.random.default_rng(2)
    stream = rng.normal(0.0, 1.0, 400)
    det = BayesianOnlineChangepoint(lam=1000.0)
    for x in stream:
        det.update(x)
    assert det.n_alarms_ <= 1


def test_drift_detected_is_true_for_one_step():
    rng = np.random.default_rng(0)
    stream = list(rng.normal(0.0, 0.1, 200)) + list(rng.normal(1.0, 0.1, 200))
    det = BayesianOnlineChangepoint(lam=100.0)
    flags = []
    for x in stream:
        det.update(x)
        flags.append(det.drift_detected)
    for i in range(1, len(flags)):
        if flags[i]:
            assert not flags[i - 1] or flags[i - 1] is False, (
                f"two consecutive alarms at {i-1}, {i}")


def test_nan_is_skipped():
    det = BayesianOnlineChangepoint(lam=100.0)
    det.update(0.0)
    r = det.run_length_.copy()
    det.update(float("nan"))
    np.testing.assert_array_equal(det.run_length_, r)


def test_reset_restores_initial_state():
    det = BayesianOnlineChangepoint(lam=100.0)
    rng = np.random.default_rng(0)
    for x in rng.normal(0.0, 1.0, 20):
        det.update(x)
    det._reset()
    assert det.n_seen_ == 0
    assert det.n_alarms_ == 0
    assert det.run_length_.shape == (1,)
    assert det.run_length_[0] == 1.0


def test_bad_arguments():
    with pytest.raises(ValueError):
        BayesianOnlineChangepoint(lam=0.0)
    with pytest.raises(ValueError):
        BayesianOnlineChangepoint(kappa0=0.0)
    with pytest.raises(ValueError):
        BayesianOnlineChangepoint(alpha0=-1.0)
    with pytest.raises(ValueError):
        BayesianOnlineChangepoint(beta0=0.0)
    with pytest.raises(ValueError):
        BayesianOnlineChangepoint(max_run_length=0)
    with pytest.raises(ValueError):
        BayesianOnlineChangepoint(mass_window=0)
    with pytest.raises(ValueError):
        BayesianOnlineChangepoint(detection_threshold=0.0)
    with pytest.raises(ValueError):
        BayesianOnlineChangepoint(detection_threshold=1.0)


def test_passes_check_estimator():
    from dense_armor.checks import check_estimator
    det = BayesianOnlineChangepoint(lam=100.0)
    out = check_estimator(det)
    assert out["not_applicable"] == []
    assert len(out["skipped"]) == 0


def test_budget_is_from_a_measured_p99():
    """budget_s is above the measured p99 on this host."""
    import time
    rng = np.random.default_rng(0)
    det = BayesianOnlineChangepoint(lam=100.0)
    ts = []
    for x in rng.normal(0.0, 1.0, 1000):
        t0 = time.perf_counter()
        det.update(x)
        ts.append(time.perf_counter() - t0)
    p99 = float(np.percentile(ts, 99))
    assert p99 < det.budget_s, (
        f"p99 {p99:.2e} s exceeds budget {det.budget_s:.2e} s")
