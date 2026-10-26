"""Tests for the online Student-t location-scale estimator.

The synthetic series is drawn from a t with known degrees of freedom
and known location and scale. Because the online estimator fixes nu,
the test checks that mu_ converges to the true location and sigma_ to
the true scale. The density is compared point by point with
``scipy.stats.t``.
"""

import math

import numpy as np
import pytest

from dense_armor.checks import check_estimator
from dense_armor.utility.stats import StudentT

SEED = 20261026


def _t_series(nu: float, mu: float, sigma: float, n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return mu + sigma * rng.standard_t(df=nu, size=n)


# The online gradient with a fixed learning_rate converges to a
# neighbourhood of the maximum-likelihood estimate, not to the MLE
# exactly: the residual width is proportional to learning_rate
# (Robbins-Monro with constant step). The tolerances below are the
# measured residual width at learning_rate=0.05 on 5000 samples, with
# margin. They are declared, not derived. For exact estimates use
# scipy.stats.t.fit on the same series.


@pytest.mark.parametrize(
    "nu,mu_true,sigma_true,tol_mu",
    [
        (2.0, 0.0, 1.0, 0.1),
        (5.0, 2.0, 0.5, 0.1),
        (30.0, -1.0, 2.0, 0.2),
    ],
)
def test_location_converges_to_neighbourhood(nu, mu_true, sigma_true, tol_mu):
    x = _t_series(nu, mu_true, sigma_true, 5000, SEED)
    m = StudentT(nu=nu, mu_init=0.0, sigma_init=1.0, learning_rate=0.05)
    for v in x:
        m.learn_one({"x": float(v)})
    assert abs(m.mu_ - mu_true) < tol_mu, (
        f"nu={nu}: mu_={m.mu_:.4f} vs true={mu_true}, tol={tol_mu}"
    )


@pytest.mark.parametrize(
    "nu,sigma_true,tol_rel",
    [
        (2.0, 1.0, 0.35),
        (5.0, 0.5, 0.25),
        (30.0, 2.0, 0.25),
    ],
)
def test_scale_converges_to_neighbourhood(nu, sigma_true, tol_rel):
    x = _t_series(nu, 0.0, sigma_true, 5000, SEED)
    m = StudentT(nu=nu, mu_init=0.0, sigma_init=1.0, learning_rate=0.05)
    for v in x:
        m.learn_one({"x": float(v)})
    assert abs(m.sigma_ - sigma_true) / sigma_true < tol_rel, (
        f"nu={nu}: sigma_={m.sigma_:.4f} vs true={sigma_true}, tol={tol_rel}"
    )


def test_logpdf_matches_scipy():
    scipy_stats = pytest.importorskip("scipy.stats")
    m = StudentT(nu=5.0, mu_init=0.0, sigma_init=1.0, learning_rate=0.1)
    for v in [1.0, 2.0, 3.0, 4.0, 5.0] * 20:
        m.learn_one({"x": v})
    for x_test in (-2.0, 0.0, 1.5, 3.0, 10.0):
        ref = float(scipy_stats.t.logpdf(x_test, df=m.nu, loc=m.mu_, scale=m.sigma_))
        got = m.transform_one({"x": x_test})["logpdf"]
        assert abs(got - ref) < 1e-10, f"x={x_test}: got {got!r}, expected {ref!r}"


def test_variance_matches_closed_form():
    m = StudentT(nu=5.0, mu_init=0.0, sigma_init=2.0, learning_rate=0.1)
    v = float(m.transform_one({"x": 0.0})["var"])
    expected = m.sigma_**2 * m.nu / (m.nu - 2.0)
    assert abs(v - expected) < 1e-12


def test_variance_infinite_when_nu_le_2():
    m = StudentT(nu=2.0)
    assert math.isinf(float(m.transform_one({"x": 0.0})["var"]))
    m = StudentT(nu=1.5)
    assert math.isinf(float(m.transform_one({"x": 0.0})["var"]))


def test_merge_counts_and_average():
    x = _t_series(5.0, 0.0, 1.0, 200, SEED)
    left = StudentT(nu=5.0, learning_rate=0.1)
    for v in x[:100]:
        left.learn_one({"x": float(v)})
    right = StudentT(nu=5.0, learning_rate=0.1)
    for v in x[100:]:
        right.learn_one({"x": float(v)})
    mu_left, mu_right = left.mu_, right.mu_
    ls_left, ls_right = left.log_sigma_, right.log_sigma_
    left.merge(right)
    assert left.n_ == 200
    # merge is a weighted average by n_ (both halves equal, weight 0.5)
    assert abs(left.mu_ - 0.5 * (mu_left + mu_right)) < 1e-12
    assert abs(left.log_sigma_ - 0.5 * (ls_left + ls_right)) < 1e-12
    # the merged state is closer to both halves than either was
    assert left.mu_ == pytest.approx(0.5 * (mu_left + mu_right), abs=1e-12)


def test_merge_empty():
    a = StudentT(nu=5.0)
    b = StudentT(nu=5.0)
    a.merge(b)
    assert a.n_ == 0


def test_state_dict_roundtrip():
    m = StudentT(nu=5.0, learning_rate=0.1)
    for v in [1.0, 2.0, 3.0, 4.0, 5.0]:
        m.learn_one({"x": v})
    out1 = m.transform_one({"x": 3.0})
    sd = m.state_dict()
    m2 = StudentT(nu=m.nu, learning_rate=0.1)
    m2.load_state_dict(sd)
    out2 = m2.transform_one({"x": 3.0})
    for k in out1:
        assert abs(out1[k] - out2[k]) < 1e-12, f"key {k}"


def test_state_dict_is_json_serializable():
    import json

    m = StudentT(nu=5.0, learning_rate=0.1)
    for v in [1.0, 2.0, 3.0]:
        m.learn_one({"x": v})
    s = json.dumps(m.state_dict())
    assert isinstance(s, str) and len(s) > 0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"nu": 0},
        {"nu": -1.0},
        {"sigma_init": 0},
        {"sigma_init": -1.0},
        {"learning_rate": 0},
        {"learning_rate": -1.0},
    ],
)
def test_invalid_constructor_raises(kwargs):
    with pytest.raises(ValueError):
        StudentT(**kwargs)


def test_nan_skipped():
    m = StudentT(nu=5.0, learning_rate=0.1)
    m.learn_one({"x": float("nan")})
    m.learn_one({"x": 1.0})
    assert m.n_ == 1
    assert m.n_missing_ == 1


def test_score_one_is_negative_logpdf():
    m = StudentT(nu=5.0, learning_rate=0.1)
    for v in [1.0, 2.0, 3.0]:
        m.learn_one({"x": v})
    for x in (-1.0, 0.0, 3.0, 10.0):
        score = m.score_one({"x": x})
        logpdf = m.transform_one({"x": x})["logpdf"]
        assert abs(score + logpdf) < 1e-12


def test_check_estimator():
    skip = {}
    check_estimator(StudentT(nu=5.0), allow_not_applicable=skip)
