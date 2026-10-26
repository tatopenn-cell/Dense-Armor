"""Tests for the online Gaussian mixture with diagonal covariance."""

import math

import numpy as np
import pytest

from dense_armor.checks import check_estimator
from dense_armor.utility.stats import OnlineGaussianMixture

SEED = 20261026


def _bimodal_1d(n_per_mode=200, sep=2.0, noise=0.1, seed=SEED, shuffle=True):
    rng = np.random.default_rng(seed)
    a = rng.normal(-sep, noise, n_per_mode)
    b = rng.normal(+sep, noise, n_per_mode)
    x = np.concatenate([a, b])
    if shuffle:
        rng.shuffle(x)
    return x


def test_1d_bimodal_recovers_two_means_with_init_means():
    # A realistic stream: the two modes are interleaved (shuffle=True).
    # A sequential stream (all of mode A, then all of mode B) collapses
    # the components on the first mode and then both follow the second;
    # that is the standard dead-component behaviour of online EM, and
    # it is documented by the test below.
    x = _bimodal_1d(shuffle=True)
    m = OnlineGaussianMixture(
        n_components=2, learning_rate=0.1, init_means=[[-2.0], [2.0]]
    )
    for v in x:
        m.learn_one({"x": float(v)})
    means = sorted(float(v) for v in m.means_.ravel())
    assert means[0] < -1.0, f"left mean not recovered: {means}"
    assert means[1] > 1.0, f"right mean not recovered: {means}"
    assert abs(sum(float(w) for w in m.weights_) - 1.0) < 1e-9


def test_1d_bimodal_sequential_stream_collapses_declared():
    # Declared limitation: on a stream where all samples of mode A come
    # before all samples of mode B, the two components collapse on mode
    # A during the first half and both follow mode B during the second
    # half. The result is a valid distribution (weights sum to one,
    # variances positive), but the two means are close to each other.
    # This is the standard dead-component behaviour of online EM with a
    # fixed step size, and it is declared in the class docstring.
    x = _bimodal_1d(shuffle=False)
    m = OnlineGaussianMixture(
        n_components=2, learning_rate=0.1, init_means=[[-2.0], [2.0]]
    )
    for v in x:
        m.learn_one({"x": float(v)})
    means = sorted(float(v) for v in m.means_.ravel())
    assert abs(means[0] - means[1]) < 1.0, (
        f"expected the two means to collapse on sequential data, got {means}"
    )
    assert abs(float(sum(m.weights_)) - 1.0) < 1e-9
    assert bool((m.variances_ > 0).all())


def test_1d_bimodal_default_init_is_valid_distribution():
    # Declared limitation, as for 2D: the default init (offsets along
    # the first axis) does not recover two well-separated 1D modes on
    # random draws, it needs init_means. The test only asserts that the
    # result is a valid distribution, not that it recovers the modes.
    x = _bimodal_1d()
    m = OnlineGaussianMixture(n_components=2, learning_rate=0.1)
    for v in x:
        m.learn_one({"x": float(v)})
    assert abs(float(sum(m.weights_)) - 1.0) < 1e-9
    assert bool((m.variances_ > 0).all())


def test_2d_three_components_with_init_means():
    rng = np.random.default_rng(SEED)
    centers = [(0.0, 0.0), (5.0, 5.0), (-5.0, 5.0)]
    m = OnlineGaussianMixture(
        n_components=3,
        learning_rate=0.05,
        init_scale=0.5,
        init_means=[list(c) for c in centers],
    )
    for _ in range(300):
        for cx, cy in centers:
            m.learn_one(
                {
                    "x": cx + float(rng.normal(0, 0.3)),
                    "y": cy + float(rng.normal(0, 0.3)),
                }
            )
    assert np.allclose(m.means_, np.asarray(centers), atol=0.5), m.means_
    assert all(float(w) > 0.2 for w in m.weights_), m.weights_
    assert bool((m.variances_ > 0).all())


def test_2d_default_init_keeps_valid_distribution():
    rng = np.random.default_rng(SEED)
    centers = [(0.0, 0.0), (5.0, 5.0), (-5.0, 5.0)]
    m = OnlineGaussianMixture(n_components=3, learning_rate=0.05, init_scale=2.0)
    for _ in range(300):
        for cx, cy in centers:
            m.learn_one(
                {
                    "x": cx + float(rng.normal(0, 0.3)),
                    "y": cy + float(rng.normal(0, 0.3)),
                }
            )
    assert abs(float(sum(m.weights_)) - 1.0) < 1e-9
    assert bool((m.variances_ > 0).all())


def test_logpdf_matches_scipy_for_one_component():
    scipy_stats = pytest.importorskip("scipy.stats")
    m = OnlineGaussianMixture(n_components=1, learning_rate=0.1, reg_covar=0.0)
    for v in [1.0, 2.0, 3.0, 4.0, 5.0] * 20:
        m.learn_one({"x": v})
    mu = float(m.means_[0, 0])
    sigma = math.sqrt(float(m.variances_[0, 0]))
    for x_test in (-2.0, 0.0, 1.5, 3.0, 10.0):
        ref = float(scipy_stats.norm.logpdf(x_test, loc=mu, scale=sigma))
        got = m.transform_one({"x": x_test})["logpdf"]
        assert abs(got - ref) < 1e-10, f"x={x_test}: got {got!r}, ref {ref!r}"


def test_variances_positive():
    m = OnlineGaussianMixture(n_components=3, learning_rate=0.1)
    rng = np.random.default_rng(SEED)
    for _ in range(100):
        m.learn_one({"x": float(rng.normal(0, 1))})
    assert bool((m.variances_ > 0).all())


def test_weights_sum_to_one():
    m = OnlineGaussianMixture(n_components=3, learning_rate=0.1)
    rng = np.random.default_rng(SEED)
    for _ in range(50):
        m.learn_one({"x": float(rng.normal(0, 1))})
    assert abs(float(sum(m.weights_)) - 1.0) < 1e-9


def test_score_one_is_negative_logpdf():
    m = OnlineGaussianMixture(n_components=2, learning_rate=0.1)
    for v in [0.0, 1.0, 2.0, 3.0]:
        m.learn_one({"x": v})
    for x in (-1.0, 0.0, 3.0):
        score = m.score_one({"x": x})
        logpdf = m.transform_one({"x": x})["logpdf"]
        assert abs(score + logpdf) < 1e-12


def test_merge_counts_and_average():
    a = OnlineGaussianMixture(n_components=2, learning_rate=0.1)
    b = OnlineGaussianMixture(n_components=2, learning_rate=0.1)
    for v in [1.0, 2.0, 3.0]:
        a.learn_one({"x": v})
    for v in [4.0, 5.0]:
        b.learn_one({"x": v})
    w_a = a.weights_.copy()
    w_b = b.weights_.copy()
    a.merge(b)
    assert a.n_ == 5
    expected = 0.6 * w_a + 0.4 * w_b
    assert np.allclose(a.weights_, expected, atol=1e-12)


def test_merge_empty_returns_self():
    a = OnlineGaussianMixture(n_components=2)
    b = OnlineGaussianMixture(n_components=2)
    a.merge(b)
    assert a.n_ == 0


def test_state_dict_roundtrip():
    m = OnlineGaussianMixture(n_components=2, learning_rate=0.1)
    for v in [1.0, 2.0, 3.0]:
        m.learn_one({"x": v})
    out1 = m.transform_one({"x": 2.0})
    sd = m.state_dict()
    m2 = OnlineGaussianMixture(n_components=2, learning_rate=0.1)
    m2.load_state_dict(sd)
    out2 = m2.transform_one({"x": 2.0})
    assert abs(out1["logpdf"] - out2["logpdf"]) < 1e-12


def test_state_dict_is_json_serializable():
    import json

    m = OnlineGaussianMixture(n_components=2, learning_rate=0.1)
    for v in [1.0, 2.0]:
        m.learn_one({"x": v})
    s = json.dumps(m.state_dict(), default=str)
    assert isinstance(s, str) and len(s) > 0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_components": 0},
        {"n_components": -1},
        {"learning_rate": 0},
        {"learning_rate": 1.5},
        {"learning_rate": -0.1},
        {"reg_covar": -1},
        {"init_scale": 0},
        {"init_scale": -1},
    ],
)
def test_invalid_constructor_raises(kwargs):
    with pytest.raises(ValueError):
        OnlineGaussianMixture(**kwargs)


def test_nan_skipped():
    m = OnlineGaussianMixture(n_components=2)
    m.learn_one({"x": float("nan")})
    m.learn_one({"x": 1.0})
    m.learn_one({"x": 2.0})
    assert m.n_ == 2
    assert m.n_missing_ == 1


def test_fit_equals_learn_one():
    samples = [{"x": float(v)} for v in _bimodal_1d(n_per_mode=50)]
    a = OnlineGaussianMixture(n_components=2, learning_rate=0.1)
    a.fit(samples)
    b = OnlineGaussianMixture(n_components=2, learning_rate=0.1)
    for s in samples:
        b.learn_one(s)
    assert a.n_ == b.n_
    assert np.allclose(a.means_, b.means_, atol=1e-15)
    assert np.allclose(a.weights_, b.weights_, atol=1e-15)


def test_transform_one_before_init_returns_nan():
    m = OnlineGaussianMixture(n_components=2)
    out = m.transform_one({"x": 1.0})
    assert math.isnan(out["logpdf"])
    assert out["responsibilities"] == []


def test_transform_one_nan_returns_nan():
    m = OnlineGaussianMixture(n_components=2)
    for v in [1.0, 2.0]:
        m.learn_one({"x": v})
    out = m.transform_one({"x": float("nan")})
    assert math.isnan(out["logpdf"])


def test_score_one_before_init_returns_nan():
    m = OnlineGaussianMixture(n_components=2)
    assert math.isnan(m.score_one({"x": 1.0}))


def test_score_one_nan_returns_nan():
    m = OnlineGaussianMixture(n_components=2)
    for v in [1.0, 2.0]:
        m.learn_one({"x": v})
    assert math.isnan(m.score_one({"x": float("nan")}))


def test_init_means_wrong_shape_raises():
    m = OnlineGaussianMixture(n_components=2, init_means=[[0.0], [1.0], [2.0]])
    with pytest.raises(ValueError, match="init_means"):
        m.learn_one({"x": 0.5})


def test_merge_uninitialized_with_initialized():
    a = OnlineGaussianMixture(n_components=2, learning_rate=0.1)
    b = OnlineGaussianMixture(n_components=2, learning_rate=0.1)
    for v in [1.0, 2.0, 3.0]:
        b.learn_one({"x": v})
    a.merge(b)
    assert a.initialized_
    assert a.n_ == 3


def test_merge_initialized_with_uninitialized():
    a = OnlineGaussianMixture(n_components=2, learning_rate=0.1)
    for v in [1.0, 2.0, 3.0]:
        a.learn_one({"x": v})
    b = OnlineGaussianMixture(n_components=2, learning_rate=0.1)
    a.merge(b)
    assert a.n_ == 3


def test_transform_one_feature_named():
    m = OnlineGaussianMixture(n_components=2, feature="x")
    for v in [1.0, 2.0, 3.0]:
        m.learn_one({"x": v, "y": 0.0})
    out = m.transform_one({"x": 2.0, "y": 100.0})
    assert out["logpdf"] == out["logpdf"]


def test_check_estimator():
    check_estimator(OnlineGaussianMixture(n_components=2), allow_not_applicable={})
