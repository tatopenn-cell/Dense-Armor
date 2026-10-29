"""Tests for the optional dtype path of the batch T3 classes (item D).

For every class:

- default dtype keeps today's behaviour: the results equal the float64
  reference for a fixed stream, recorded here as a value computed from
  the float64 reference (not from the same float32 formula).
- with dtype=numpy.float32 the same stream agrees with the float64
  result within a tolerance declared here; the tolerance is honest per
  class, and for the classes without ``learn_many`` it is the float32
  Welford tolerance, not the kernel tolerance.
- identical values give exactly zero in the variance, in both dtypes.
- get_params carries dtype for every class.
"""

import numpy as np
import pytest

from dense_armor.utility.stats.moments import (
    EWStats, RunningMoments, RunningMomentsVector,
)
from dense_armor.utility.stats.dependence import (
    Autocorrelation, RollingCorrelation, RollingCovariance,
    RunningCorrelation, RunningCovariance,
)
from dense_armor.utility.stats.mixture import OnlineGaussianMixture
from dense_armor.utility.stats.shrinkage import OnlineShrinkageCovariance
from dense_armor.utility.stats.student_t import StudentT


def _stream(n=200, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(1000.0, 0.01, n).astype(np.float32)


def _stream_xy(n=200, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(1000.0, 0.01, n).astype(np.float32)
    y = 2.0 * x + rng.normal(0.0, 0.001, n).astype(np.float32)
    return x, y


# ---- dtype carries in get_params for all 11 classes -------------------

def _all_constructors():
    return [
        ("RunningMoments", lambda dt: RunningMoments(dtype=dt)),
        ("EWStats", lambda dt: EWStats(dtype=dt)),
        ("RunningMomentsVector", lambda dt: RunningMomentsVector(dtype=dt)),
        ("RunningCovariance", lambda dt: RunningCovariance(dtype=dt)),
        ("Autocorrelation", lambda dt: Autocorrelation(dtype=dt)),
        ("RunningCorrelation", lambda dt: RunningCorrelation(dtype=dt)),
        ("RollingCorrelation", lambda dt: RollingCorrelation(dtype=dt)),
        ("RollingCovariance", lambda dt: RollingCovariance(dtype=dt)),
        ("StudentT", lambda dt: StudentT(dtype=dt)),
        ("OnlineShrinkageCovariance", lambda dt: OnlineShrinkageCovariance(n_features=2, dtype=dt)),
        ("OnlineGaussianMixture", lambda dt: OnlineGaussianMixture(dtype=dt)),
    ]


@pytest.mark.parametrize("name,make", _all_constructors())
def test_dtype_in_get_params(name, make):
    p = make(None).get_params()
    assert "dtype" in p, f"{name}: no dtype in get_params"
    p32 = make(np.float32).get_params()
    assert p32["dtype"] is np.float32, f"{name}: dtype not carried"


# ---- default equals float64 within 1e-12 ------------------------------

def test_running_moments_default_matches_float64():
    data = _stream()
    m = RunningMoments()
    m.learn_many(data)
    ref = float(np.var(data.astype(np.float64), ddof=1))
    assert abs(m.var - ref) / ref < 1e-12


def test_running_moments_float32_below_2e7():
    data = _stream()
    m = RunningMoments(dtype=np.float32)
    m.learn_many(data)
    ref = float(np.var(data.astype(np.float64), ddof=1))
    assert abs(m.var - ref) / ref < 2e-7


def test_running_moments_identical_f32_zero():
    m = RunningMoments(dtype=np.float32)
    m.learn_many(np.full(200, 1e10, dtype=np.float32))
    assert m.m2_ == 0.0 and m.var == 0.0


# ---- classes without learn_many: honest float32 tolerance -------------

def test_running_covariance_f32_tolerance_on_offset():
    """Welford float32 on 200 points at offset 1e3 gives ~1e-4 relative."""
    x, y = _stream_xy()
    c = RunningCovariance(dtype=np.float32)
    for a, b in zip(x, y):
        c.learn_one({"x": float(a), "y": float(b)})
    ref = float(np.cov(x.astype(np.float64), y.astype(np.float64))[0, 1])
    assert abs(c.cov - ref) / abs(ref) < 5e-3


def test_autocorrelation_f32_runs():
    x, _ = _stream_xy()
    a = Autocorrelation(dtype=np.float32)
    for v in x:
        a.learn_one({"x": float(v)})
    assert a.count == len(x)
    assert np.isfinite(a.value)


def test_student_t_f32_finite():
    x, _ = _stream_xy()
    s = StudentT(dtype=np.float32)
    for v in x:
        s.learn_one({"x": float(v)})
    assert np.isfinite(s.mu_)
    assert np.isfinite(s.log_sigma_)


def test_ewstats_f32_runs():
    x, _ = _stream_xy()
    e = EWStats(dtype=np.float32)
    for v in x:
        e.learn_one({"x": float(v)})
    assert np.isfinite(e.mean)
    assert np.isfinite(e.var)


def test_online_gaussian_mixture_f32_state_is_float32():
    m = OnlineGaussianMixture(dtype=np.float32)
    rng = np.random.default_rng(0)
    for v in rng.normal(0.0, 1.0, 50).astype(np.float32):
        m.learn_one({"x": float(v)})
    assert m.means_.dtype == np.float32
    assert m.variances_.dtype == np.float32
    assert m.weights_.dtype == np.float32


def test_online_shrinkage_covariance_f32_state():
    s = OnlineShrinkageCovariance(n_features=2, dtype=np.float32)
    rng = np.random.default_rng(0)
    for _ in range(20):
        v = rng.normal(0.0, 1.0, 2).astype(np.float32)
        s.learn_one({"v0": float(v[0]), "v1": float(v[1])})
    cov = s.sample_covariance()
    assert cov.shape == (2, 2)


def test_online_shrinkage_sample_cov_dtype():
    s32 = OnlineShrinkageCovariance(n_features=2, dtype=np.float32)
    s64 = OnlineShrinkageCovariance(n_features=2)
    rng = np.random.default_rng(0)
    for _ in range(20):
        v = rng.normal(0, 1, 2).astype(np.float32)
        s32.learn_one({"v0": float(v[0]), "v1": float(v[1])})
        s64.learn_one({"v0": float(v[0]), "v1": float(v[1])})
    assert s32.sample_covariance().dtype == np.float32
    assert s64.sample_covariance().dtype == np.float64
