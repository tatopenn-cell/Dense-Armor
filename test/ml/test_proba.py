"""Hand cases of the distributions of the machine learning package."""

import doctest

import numpy as np
import pytest

import dense_armor.ml.proba as PR
from dense_armor.ml.proba import Beta, Gaussian, Multinomial


def test_doctests():
    r = doctest.testmod(PR)
    assert r.failed == 0 and r.attempted >= 6


def test_beta_hand_case():
    d = Beta().fit([1, 1, 0, 1])
    assert (d.a_, d.b_) == (4.0, 2.0) and abs(d.mean() - 2 / 3) < 1e-12
    assert abs(d.variance() - 8.0 / (36.0 * 7.0)) < 1e-12 and d.mode() == 0.75
    assert abs(d.pdf(0.5) - 1.25) < 1e-12 and d.pdf(0.0) == 0.0 and d.pdf(2.0) == 0.0
    flat = Beta().fit([])
    assert flat.mode() == 0.5 and abs(flat.pdf(0.3) - 1.0) < 1e-12
    s = d.sample(500)
    assert s.shape == (500,) and abs(s.mean() - 2 / 3) < 0.05
    with pytest.raises(ValueError):
        Beta(a=0.0)


def test_gaussian_hand_case():
    d = Gaussian().fit([1.0, 2.0, 3.0, 4.0])
    assert d.mean_ == 2.5 and abs(d.var_ - 5.0 / 3.0) < 1e-12
    p = Gaussian(mean=0.0, var=1.0, n=2.0).fit([1.0, 2.0, 3.0, 4.0])
    assert (
        abs(p.mean_ - 5.0 / 3.0) < 1e-12
        and abs(p.var_ - 14.333333333333334 / 5.0) < 1e-12
    )
    assert abs(d.cdf(2.5) - 0.5) < 1e-12
    assert abs(d.pdf(2.5) - 1.0 / np.sqrt(2 * np.pi * 1.25)) < 1e-12
    assert d.m2_ == 5.0
    e = Gaussian(mean=3.0, var=2.0, n=1.0).fit([])
    assert (e.mean_, e.var_, e.n_) == (3.0, 2.0, 1.0)
    assert e.m2_ == 0.0 and e.pdf(3.0) == 0.0
    assert e.cdf(2.0) == 0.0 and e.cdf(3.0) == 1.0
    empty = Gaussian(mean=3.0, var=2.0).fit([])
    assert abs(empty.pdf(3.0) - 1.0 / np.sqrt(2 * np.pi * 2.0)) < 1e-12
    one = Gaussian().fit([5.0])
    assert one.mean_ == 5.0 and one.var_ == 1.0
    assert Gaussian().fit([1.0, 3.0]).sample(200).shape == (200,)
    with pytest.raises(ValueError):
        Gaussian(var=0.0)


def test_multinomial_hand_case():
    d = Multinomial(alpha=[1.0, 2.0, 3.0]).fit([1, 1])
    assert d.counts_.tolist() == [1.0, 4.0, 3.0] and d.mode() == 1
    assert abs(d.pmf(2) - 3.0 / 8.0) < 1e-12 and abs(d.mean().sum() - 1.0) < 1e-12
    assert set(d.sample(50).tolist()) <= {0, 1, 2}
    with pytest.raises(ValueError):
        Multinomial(alpha=0.0).fit([0])
