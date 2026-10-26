"""Tests for GARCH(1,1) conditional variance with fixed parameters."""

import math

import numpy as np
import pytest

from dense_armor.checks import check_estimator
from dense_armor.utility.timeseries import GARCH11

SEED = 20261026


def _garch_series(n, omega=1e-6, alpha=0.08, beta=0.90, seed=SEED):
    rng = np.random.default_rng(seed)
    sigma2 = omega / (1.0 - alpha - beta)
    xs = np.empty(n)
    for i in range(n):
        eps = float(rng.standard_normal())
        sigma2 = omega + alpha * sigma2 * eps * eps + beta * sigma2
        xs[i] = math.sqrt(sigma2) * eps
    return xs


def test_loglik_matches_scipy():
    scipy_stats = pytest.importorskip("scipy.stats")
    x = _garch_series(200)
    m = GARCH11(omega=1e-6, alpha=0.08, beta=0.90)
    for v in x:
        m.learn_one({"x": float(v)})
    last = float(x[-1])
    ref = float(scipy_stats.norm.logpdf(last, scale=math.sqrt(m.sigma2_)))
    got = m.transform_one({"x": last})["loglik"]
    assert abs(got - ref) < 1e-12, f"got {got!r}, ref {ref!r}"


def test_sigma2_positive_at_every_step():
    x = _garch_series(500)
    m = GARCH11(omega=1e-6, alpha=0.08, beta=0.90)
    for v in x:
        m.learn_one({"x": float(v)})
        assert m.sigma2_ > 0.0, f"non-positive sigma2_ after {m.n_} samples"


def test_constant_parameters_and_stationarity():
    x = _garch_series(500)
    m = GARCH11(omega=1e-6, alpha=0.08, beta=0.90)
    for v in x:
        m.learn_one({"x": float(v)})
    assert m.omega == 1e-6
    assert m.alpha == 0.08
    assert m.beta == 0.90
    assert m.alpha + m.beta < 1.0


def test_initial_sigma2_is_unconditional_variance():
    m = GARCH11(omega=1e-6, alpha=0.08, beta=0.90)
    expected = 1e-6 / (1.0 - 0.08 - 0.90)
    assert abs(m.sigma2_ - expected) < 1e-12


def test_empirical_variance_is_plausible():
    # The average conditional variance of one GARCH realization is a
    # noisy estimate of the theoretical unconditional variance. With
    # alpha + beta = 0.98 the process is highly persistent
    # (autocorrelation time ~50 samples), so the last 1000 samples
    # contain only ~20 independent draws. The test asserts a wide
    # plausibility range (factor 3) rather than a tight match: it
    # checks that the recursion does not blow up and does not vanish,
    # not that the empirical average equals the theoretical value.
    x = _garch_series(5000)
    m = GARCH11(omega=1e-6, alpha=0.08, beta=0.90)
    sigma2s = []
    for v in x:
        m.learn_one({"x": float(v)})
        sigma2s.append(m.sigma2_)
    avg_last = float(np.mean(sigma2s[-1000:]))
    theoretical = 1e-6 / (1.0 - 0.08 - 0.90)
    assert theoretical / 3.0 < avg_last < theoretical * 3.0, (
        f"avg sigma2={avg_last:.3e}, theoretical={theoretical:.3e}"
    )


def test_score_one_is_negative_loglik():
    x = _garch_series(50)
    m = GARCH11(omega=1e-6, alpha=0.08, beta=0.90)
    for v in x:
        m.learn_one({"x": float(v)})
    for v in (-0.01, 0.0, 0.02):
        score = m.score_one({"x": v})
        loglik = m.transform_one({"x": v})["loglik"]
        assert abs(score + loglik) < 1e-12


def test_merge_counts_and_average():
    a = GARCH11(omega=1e-6, alpha=0.08, beta=0.90)
    b = GARCH11(omega=1e-6, alpha=0.08, beta=0.90)
    for v in (0.01, -0.02):
        a.learn_one({"x": v})
    for v in (0.015, -0.005, 0.03):
        b.learn_one({"x": v})
    s_a = a.sigma2_
    s_b = b.sigma2_
    a.merge(b)
    assert a.n_ == 5
    expected = 0.4 * s_a + 0.6 * s_b
    assert abs(a.sigma2_ - expected) < 1e-12


def test_merge_empty_returns_self():
    a = GARCH11(omega=1e-6, alpha=0.08, beta=0.90)
    b = GARCH11(omega=1e-6, alpha=0.08, beta=0.90)
    a.merge(b)
    assert a.n_ == 0


def test_state_dict_roundtrip():
    m = GARCH11(omega=1e-6, alpha=0.08, beta=0.90)
    for v in (0.01, -0.02, 0.015):
        m.learn_one({"x": v})
    out1 = m.transform_one({"x": 0.005})
    sd = m.state_dict()
    m2 = GARCH11(omega=1e-6, alpha=0.08, beta=0.90)
    m2.load_state_dict(sd)
    out2 = m2.transform_one({"x": 0.005})
    assert out1["sigma2"] == out2["sigma2"]
    assert out1["loglik"] == out2["loglik"]
    assert m.initialized_ == m2.initialized_


def test_state_dict_is_json_serializable():
    import json

    m = GARCH11(omega=1e-6, alpha=0.08, beta=0.90)
    for v in (0.01, -0.02):
        m.learn_one({"x": v})
    s = json.dumps(m.state_dict())
    assert isinstance(s, str) and len(s) > 0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"omega": 0},
        {"omega": -1},
        {"alpha": -0.1},
        {"beta": -0.1},
        {"alpha": 0.6, "beta": 0.5},
        {"alpha": 0.5, "beta": 0.5},
    ],
)
def test_invalid_constructor_raises(kwargs):
    with pytest.raises(ValueError):
        GARCH11(**kwargs)


def test_nan_skipped():
    m = GARCH11(omega=1e-6, alpha=0.08, beta=0.90)
    s0 = m.sigma2_
    m.learn_one({"x": float("nan")})
    m.learn_one({"x": 0.01})
    assert m.n_ == 1
    assert m.n_missing_ == 1
    assert m.sigma2_ != s0


def test_fit_equals_learn_one():
    x = _garch_series(200)
    samples = [{"x": float(v)} for v in x]
    a = GARCH11(omega=1e-6, alpha=0.08, beta=0.90)
    a.fit(samples)
    b = GARCH11(omega=1e-6, alpha=0.08, beta=0.90)
    for s in samples:
        b.learn_one(s)
    assert a.n_ == b.n_
    assert a.sigma2_ == b.sigma2_
    assert a.epsilon_ == b.epsilon_


def test_check_estimator():
    check_estimator(GARCH11(omega=1e-6, alpha=0.08, beta=0.90), allow_not_applicable={})
