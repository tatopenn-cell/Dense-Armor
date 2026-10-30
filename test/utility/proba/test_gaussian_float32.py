"""Float32 path of Gaussian: default bit for bit, float32 agrees.

The reference of every float32 comparison is the float64 variance of the
same float32 values, never the variance of the float64 values they came
from, so the rounding of the input is not counted as error.
"""

import numpy as np
import pytest

from dense_armor.utility.proba import Gaussian

STORED = [0.0, 1.0, 2.5, 4.0, 7.0, 11.0]


def _run(cls, values):
    g = cls()
    for v in values:
        g.update(v)
    return g


def test_default_matches_hand_welford_bit_for_bit():
    mean = 0.0
    m2 = 0.0
    n = 0.0
    for v in STORED:
        n_new = n + 1.0
        delta = v - mean
        mean = mean + delta / n_new
        m2 = m2 + delta * (v - mean)
        n = n_new
    g = _run(Gaussian, STORED)
    assert g.mean() == mean
    assert g._m2 == m2
    assert g.n() == n


def test_default_old_signature_positional():
    g = Gaussian(1.0, 2.0, 3.0)
    assert g.mean() == 1.0
    assert g.n() == 3.0


def test_dtype_none_is_the_default_path():
    a = Gaussian(dtype=None)
    b = Gaussian()
    for v in STORED:
        a.update(v)
        b.update(v)
    assert a.mean() == b.mean()
    assert a.variance() == b.variance()


def test_dtype_float64_goes_through_default_path():
    a = Gaussian(dtype=np.float64)
    b = Gaussian()
    for v in STORED:
        a.update(v)
        b.update(v)
    assert a.mean() == b.mean()
    assert a._m2 == b._m2


def test_float32_agrees_with_float64_on_the_decision_stream():
    rng = np.random.default_rng(0)
    xs = (1e3 + 0.01 * rng.standard_normal(100_000)).astype(np.float32)
    g = Gaussian(dtype=np.float32)
    for v in xs:
        g.update(float(v))
    ref = float(xs.astype(np.float64).var(ddof=1))
    assert abs(g.variance() - ref) / ref <= 1e-6
    assert g._s1.dtype == np.float32
    assert g._s2.dtype == np.float32


@pytest.mark.parametrize("big", [1e10, 1e38])
def test_float32_constant_stream_is_exactly_zero(big):
    g = Gaussian(dtype=np.float32)
    for _ in range(5):
        g.update(float(big))
    assert g.variance() == 0.0


def test_float32_revert_raises():
    g = Gaussian(dtype=np.float32)
    g.update(1.0)
    with pytest.raises(NotImplementedError):
        g.revert(1.0)


def test_float32_weight_must_be_one():
    g = Gaussian(dtype=np.float32)
    with pytest.raises(ValueError):
        g.update(1.0, w=0.5)


def test_float32_variance_undefined_below_two():
    g = Gaussian(dtype=np.float32)
    g.update(1.0)
    with pytest.raises(ValueError):
        g.variance()


def test_float32_pdf_and_cdf_are_finite():
    g = Gaussian(dtype=np.float32)
    for v in STORED:
        g.update(v)
    assert np.isfinite(g.pdf(g.mean()))
    c = g.cdf(g.mean())
    assert 0.0 <= c <= 1.0


def test_float32_initial_state_with_n_positive():
    g = Gaussian(mean=5.0, var=2.0, n=3.0, dtype=np.float32)
    assert g.n() == 3.0
    assert abs(g.mean() - 5.0) < 1e-5
    g.update(5.0)
    assert abs(g.mean() - 5.0) < 1e-4
