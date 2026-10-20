"""Tests for the online shrinkage covariance (Bodnar et al. 2014)."""

import numpy as np
import pytest

from dense_armor.utility.stats.shrinkage import OnlineShrinkageCovariance


def _data(n=500, p=4, seed=0):
    rng = np.random.default_rng(seed)
    L = np.linalg.cholesky(
        np.array([[1.0, 0.4, 0.2, 0.1],
                  [0.4, 1.0, 0.3, 0.2],
                  [0.2, 0.3, 1.0, 0.25],
                  [0.1, 0.2, 0.25, 1.0]])) if p == 4 else np.eye(p)
    return rng.normal(size=(n, p)) @ L.T


def _feed(p, data):
    o = OnlineShrinkageCovariance(p)
    for row in data:
        o.learn_one({f"v{i}": float(row[i]) for i in range(p)})
    return o


def test_sample_covariance_matches_numpy():
    data = _data()
    o = _feed(4, data)
    s_batch = np.cov(data, rowvar=False, ddof=1)
    np.testing.assert_allclose(o.sample_covariance(), s_batch, atol=1e-10)


def test_shrinkage_is_a_convex_combination():
    data = _data()
    o = _feed(4, data)
    s = o.sample_covariance()
    shrunk, alpha, beta = o.shrinkage()
    assert 0.0 <= alpha <= 1.0
    p = 4
    tr_s = float(np.trace(s))
    assert abs(beta - (1.0 - alpha) * tr_s / p) < 1e-12
    expected = alpha * s + beta * np.eye(p)
    np.testing.assert_allclose(shrunk, expected, atol=1e-12)


def test_off_diagonal_is_shrunk():
    data = _data()
    o = _feed(4, data)
    s = o.sample_covariance()
    shrunk, _, _ = o.shrinkage()
    assert abs(shrunk[0, 1]) < abs(s[0, 1])


def test_batch_formula_matches():
    data = _data(n=300, p=4, seed=1)
    o = _feed(4, data)
    s = o.sample_covariance()
    n = len(data)
    p = 4
    tr_s = float(np.trace(s))
    tr_s2 = float(np.sum(s * s))
    alpha_ref = 1.0 - (tr_s * tr_s / (n * p)) / (tr_s2 - tr_s * tr_s / p)
    alpha_ref = float(min(max(alpha_ref, 0.0), 1.0))
    _, alpha, beta = o.shrinkage()
    assert abs(alpha - alpha_ref) < 1e-12
    beta_ref = (1.0 - alpha_ref) * tr_s / p
    assert abs(beta - beta_ref) < 1e-12


def test_against_sklearn_ledoit_wolf():
    pytest.importorskip("sklearn")
    from sklearn.covariance import LedoitWolf
    data = _data(n=200, p=4, seed=2)
    o = _feed(4, data)
    shrunk, _, _ = o.shrinkage()
    lw = LedoitWolf().fit(data)
    np.testing.assert_allclose(shrunk, lw.covariance_, atol=0.3)


def test_shrunk_close_to_sample_when_n_large():
    """With p/n small the shrunk matrix is close to the sample one.

    The intensity alpha is below one on every finite sample (equation
    4.3, page 12: alpha < 1 as soon as c = p/n > 0), so the test checks
    that the two matrices are close, not that alpha reaches one.
    """
    rng = np.random.default_rng(0)
    p = 4
    data = rng.normal(size=(20000, p))
    o = _feed(p, data)
    s = o.sample_covariance()
    shrunk, alpha, _ = o.shrinkage()
    assert 0.0 < alpha < 1.0
    diff = float(np.max(np.abs(shrunk - s)))
    assert diff < 1e-2, f"max diff {diff}"


def test_bad_n_features():
    with pytest.raises(ValueError, match="n_features"):
        OnlineShrinkageCovariance(0)


def test_n_seen_counts():
    data = _data(n=10)
    o = _feed(4, data)
    assert o.n_seen_ == 10


def test_shrinkage_on_single_feature():
    data = _data(n=50, p=1)
    o = _feed(1, data)
    shrunk, alpha, _beta = o.shrinkage()
    assert shrunk.shape == (1, 1)
    assert 0.0 <= alpha <= 1.0


def test_degenerate_matrix_returns_sample():
    o = OnlineShrinkageCovariance(2)
    for _ in range(5):
        o.learn_one({"v0": 1.0, "v1": 2.0})
    shrunk, alpha, beta = o.shrinkage()
    np.testing.assert_allclose(shrunk, o.sample_covariance())
    assert alpha == 1.0
    assert beta == 0.0


def test_degenerate_denominator_returns_sample():
    o = OnlineShrinkageCovariance(1)
    for v in [1.0, 2.0, 3.0, 4.0]:
        o.learn_one({"v0": v})
    shrunk, alpha, beta = o.shrinkage()
    np.testing.assert_allclose(shrunk, o.sample_covariance())
    assert alpha == 1.0
    assert beta == 0.0
