"""Tests for the conjugate distributions in proba/.

update then revert returns to the same state, and the streaming
posterior equals the batch posterior on the same data.
"""

import doctest

import numpy as np
import pytest

from dense_armor.utility.proba import Beta, Gaussian, Multinomial
from dense_armor.utility.proba import beta as beta_mod
from dense_armor.utility.proba import gaussian as gaussian_mod
from dense_armor.utility.proba import multinomial as multinomial_mod


def test_doctests():
    for mod in (beta_mod, gaussian_mod, multinomial_mod):
        r = doctest.testmod(mod)
        assert r.failed == 0


# ------------------------------------------------------------------
# Beta
# ------------------------------------------------------------------


def test_beta_update_revert_identity():
    d = Beta(a=2.0, b=3.0)
    for x in [1, 1, 0, 1, 0, 0, 1]:
        d.update(x)
    before = (d.a, d.b)
    d.update(1).revert(1)
    d.update(0).revert(0)
    assert (d.a, d.b) == before


def test_beta_update_matches_batch_counts():
    data = [1, 1, 0, 1, 0, 0, 1, 1, 1, 0]
    d = Beta(a=1.0, b=1.0)
    for x in data:
        d.update(x)
    n1 = sum(data)
    n0 = len(data) - n1
    assert d.a == 1.0 + n1
    assert d.b == 1.0 + n0
    assert abs(d.mean() - (1.0 + n1) / (2.0 + len(data))) < 1e-12


def test_beta_weighted_update():
    d = Beta(a=1.0, b=1.0)
    d.update(1, w=3.0)
    d.update(0, w=2.0)
    assert d.a == 4.0
    assert d.b == 3.0
    d.revert(1, w=3.0)
    assert d.a == 1.0


def test_beta_variance_matches_bishop_2_16():
    d = Beta(a=2.0, b=3.0)
    a, b = d.a, d.b
    expected = a * b / ((a + b) ** 2 * (a + b + 1.0))
    assert abs(d.variance() - expected) < 1e-12


def test_beta_mode():
    d = Beta(a=2.0, b=3.0)
    assert abs(d.mode() - 1.0 / 3.0) < 1e-12
    with pytest.raises(ValueError, match="mode"):
        Beta(a=1.0, b=2.0).mode()


def test_beta_sample():
    d = Beta(a=2.0, b=2.0)
    rng = np.random.default_rng(0)
    xs = np.array([d.sample(rng) for _ in range(5000)])
    assert 0.0 < xs.min() and xs.max() < 1.0
    assert abs(xs.mean() - 0.5) < 0.05


def test_beta_bad_arguments():
    with pytest.raises(ValueError, match="a must be"):
        Beta(a=0.0)
    with pytest.raises(ValueError, match="b must be"):
        Beta(b=-1.0)
    with pytest.raises(ValueError, match="w must be"):
        Beta().update(1, w=0.0)
    with pytest.raises(ValueError, match="revert"):
        Beta(a=1.0, b=1.0).revert(1)


# ------------------------------------------------------------------
# Gaussian
# ------------------------------------------------------------------


def test_gaussian_update_revert_identity():
    g = Gaussian()
    for v in [1.0, 2.5, -0.5, 3.0, 0.1]:
        g.update(v)
    before = (g.mean(), g.variance(), g.n())
    g.update(7.0).revert(7.0)
    after = (g.mean(), g.variance(), g.n())
    assert np.allclose(before, after, rtol=1e-12, atol=1e-12)


def test_gaussian_streaming_matches_batch():
    rng = np.random.default_rng(0)
    data = rng.normal(2.0, 0.5, size=200)
    g = Gaussian()
    for v in data:
        g.update(float(v))
    assert abs(g.mean() - data.mean()) < 1e-12
    assert abs(g.variance() - data.var(ddof=1)) < 1e-12


def test_gaussian_pdf_at_mean():
    g = Gaussian()
    for v in [1.0, 2.0, 3.0]:
        g.update(v)
    var_mle = g._m2 / g.n()
    expected = 1.0 / np.sqrt(2.0 * np.pi * var_mle)
    assert abs(g.pdf(2.0) - expected) < 1e-12


def test_gaussian_cdf_monotone():
    g = Gaussian()
    for v in [0.0, 1.0, 2.0, 3.0, 4.0]:
        g.update(v)
    xs = np.linspace(-5, 15, 50)
    c = np.array([g.cdf(float(x)) for x in xs])
    assert np.all(np.diff(c) >= 0.0)
    assert abs(g.cdf(g.mean()) - 0.5) < 1e-12


def test_gaussian_bad_arguments():
    with pytest.raises(ValueError, match="var"):
        Gaussian(var=0.0)
    with pytest.raises(ValueError, match="n must"):
        Gaussian(n=-1.0)
    with pytest.raises(ValueError, match="w must"):
        Gaussian().update(1.0, w=-1.0)
    with pytest.raises(ValueError, match="revert"):
        Gaussian().revert(1.0)


# ------------------------------------------------------------------
# Multinomial (Dirichlet)
# ------------------------------------------------------------------


def test_multinomial_update_revert_identity():
    d = Multinomial(alpha=[2.0, 3.0, 1.0])
    for k in [0, 1, 1, 2, 0]:
        d.update(k)
    before = d.alpha_.copy()
    d.update(1).revert(1)
    d.update(2).revert(2)
    assert np.array_equal(d.alpha_, before)


def test_multinomial_posterior_equals_batch_counts():
    data = [0, 1, 1, 2, 0, 0, 1]
    d = Multinomial(alpha=1.0, n_categories=3)
    for k in data:
        d.update(k)
    counts = np.bincount(data, minlength=3)
    expected = (np.ones(3) + counts) / (3.0 + len(data))
    assert np.allclose(d.mean(), expected)


def test_multinomial_mode():
    d = Multinomial(alpha=[3.0, 5.0, 2.0])
    expected = (np.array([3.0, 5.0, 2.0]) - 1.0) / (10.0 - 3.0)
    assert np.allclose(d.mode(), expected)
    with pytest.raises(ValueError, match="mode"):
        Multinomial(alpha=[1.0, 1.0, 1.0]).mode()


def test_multinomial_sample():
    d = Multinomial(alpha=[10.0, 1.0, 1.0])
    rng = np.random.default_rng(0)
    xs = np.array([d.sample(rng) for _ in range(2000)])
    assert abs(xs.mean()) < 0.3


def test_multinomial_bad_arguments():
    with pytest.raises(ValueError, match="alpha"):
        Multinomial(alpha=0.0)
    with pytest.raises(ValueError, match="n_categories"):
        Multinomial(alpha=1.0, n_categories=0)
    with pytest.raises(ValueError, match="alpha"):
        Multinomial(alpha=[1.0, -1.0])
    with pytest.raises(ValueError, match="category"):
        Multinomial(alpha=1.0, n_categories=2).update(5)
    with pytest.raises(ValueError, match="category"):
        Multinomial(alpha=1.0, n_categories=2).update("x")
