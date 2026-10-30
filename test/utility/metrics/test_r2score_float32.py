"""Float32 path of R2Score: default bit for bit, float32 agrees.

R2Score is a Metric, not an Estimator: check_estimator does not apply.
The float32 comparison uses the float64 variance of the same float32
values as its reference.
"""

import numpy as np
import pytest

from dense_armor.roles import Estimator
from dense_armor.utility.metrics.regression import R2Score


def test_r2score_is_a_metric_not_an_estimator():
    assert not issubclass(R2Score, Estimator)


def test_default_matches_hand_welford_bit_for_bit():
    ys = [1.0, 2.0, 3.0, 4.0, 5.0]
    yhat = [1.1, 2.1, 2.9, 4.05, 5.1]
    ss_res = 0.0
    mean = 0.0
    m2 = 0.0
    total = 0.0
    for a, b in zip(ys, yhat):
        ss_res += (a - b) ** 2
        n_new = total + 1.0
        delta = a - mean
        mean += delta / n_new
        m2 += delta * (a - mean)
        total = n_new
    m = R2Score()
    for a, b in zip(ys, yhat):
        m.update(a, b)
    assert m.ss_res_ == ss_res
    assert m.m2_ == m2
    assert m.total_ == total


def test_default_old_signature_positional():
    m = R2Score("joint1")
    assert m.feature == "joint1"


def test_default_r2_perfect_predictions():
    m = R2Score()
    for a in [1.0, 2.0, 3.0, 4.0, 5.0]:
        m.update(a, a)
    assert m.get() == 1.0


def test_dtype_none_is_default_path():
    a = R2Score(dtype=None)
    b = R2Score()
    for x, y in [(1.0, 1.1), (2.0, 2.1), (3.0, 2.9)]:
        a.update(x, y)
        b.update(x, y)
    assert a.m2_ == b.m2_
    assert a.ss_res_ == b.ss_res_


def test_dtype_float64_goes_through_default_path():
    a = R2Score(dtype=np.float64)
    b = R2Score()
    for x, y in [(1.0, 1.1), (2.0, 2.1), (3.0, 2.9)]:
        a.update(x, y)
        b.update(x, y)
    assert a.m2_ == b.m2_


def test_float32_agrees_with_float64_on_the_decision_stream():
    rng = np.random.default_rng(0)
    y = (1e3 + 0.01 * rng.standard_normal(100_000)).astype(np.float32)
    yhat = (y + 0.001 * rng.standard_normal(100_000)).astype(np.float32)
    m = R2Score(dtype=np.float32)
    for a, b in zip(y, yhat):
        m.update(float(a), float(b))
    y64 = y.astype(np.float64)
    yh64 = yhat.astype(np.float64)
    m2_ref = float((y64 - y64.mean()).dot(y64 - y64.mean()))
    ssr_ref = float((y64 - yh64).dot(y64 - yh64))
    assert abs(m.m2_ - m2_ref) / m2_ref <= 1e-6
    assert abs(m.ss_res_ - ssr_ref) / ssr_ref <= 1e-6
    assert m._s2.dtype == np.float32


@pytest.mark.parametrize("big", [1e10, 1e38])
def test_float32_constant_target_get_is_zero(big):
    m = R2Score(dtype=np.float32)
    for _ in range(5):
        m.update(float(big), float(big) + 0.1)
    assert m.get() == 0.0


def test_float32_weight_must_be_one():
    m = R2Score(dtype=np.float32)
    with pytest.raises(ValueError):
        m.update(1.0, 1.0, w=0.5)


def test_float32_merge_raises_not_implemented():
    """The reviewer case: merging float32 partial states would carry
    the accumulated error of the two shifted sums (about 5.5e-5 on a
    split N(1e3, 0.01) stream), so the method raises instead."""
    a = R2Score(dtype=np.float32)
    b = R2Score(dtype=np.float32)
    for x, y in [(1.0, 1.1), (2.0, 2.1), (3.0, 2.9)]:
        a.update(x, y)
    for x, y in [(4.0, 4.05), (5.0, 4.9)]:
        b.update(x, y)
    with pytest.raises(NotImplementedError):
        a.merge(b)


def test_float32_merge_with_default_other_raises():
    a = R2Score(dtype=np.float32)
    b = R2Score()
    a.update(1.0, 1.1)
    b.update(2.0, 2.1)
    with pytest.raises(NotImplementedError):
        a.merge(b)


def test_default_merge_does_not_enable_float32():
    a = R2Score()
    b = R2Score()
    for x, y in [(1.0, 1.1), (2.0, 2.1), (3.0, 2.9)]:
        a.update(x, y)
    for x, y in [(4.0, 4.05), (5.0, 4.9)]:
        b.update(x, y)
    m = a.merge(b)
    assert m._f32 is False
    assert m.dtype is None
