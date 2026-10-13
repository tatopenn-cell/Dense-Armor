"""Tests for dense_armor.utility.timeseries."""

import doctest
import importlib

import numpy as np
import pytest

from dense_armor.checks import check_estimator
from dense_armor.checks.robot import check_step_is_jittable
from dense_armor.utility.timeseries import (
    CycleResidual,
    InnovationAdaptiveKalman,
    KalmanFilter,
    OnlineAR,
    OnlineVAR,
    VBAdaptiveKalman,
)


@pytest.fixture(scope="module")
def arm_series():
    """The tau_1 stream of a SyntheticArm payload fault, cached per module.

    Building the SyntheticArm compiles the JAX dynamics once (~45 s).
    Both cycle tests need the same stream, so the fixture builds it a
    single time and hands the cached array to each test.
    """
    from pathlib import Path

    from dense_armor.utility.datasets import SyntheticArm

    ds = SyntheticArm(
        Path("test/fixtures/urdf/panda.urdf"),
        period_s=1.0,
        rate_hz=50.0,
        n_cycles=16,
        fault_at_s=12.0,
        fault="payload",
        payload_mass=5.0,
        seed=0,
    )
    return np.asarray([float(sig["tau_1"]) for sig, _ in ds.stream()])


MODULES = [
    "dense_armor.utility.timeseries.ar",
    "dense_armor.utility.timeseries.kalman",
    "dense_armor.utility.timeseries.cycle",
]


@pytest.mark.parametrize("modname", MODULES)
def test_doctests(modname):
    mod = importlib.import_module(modname)
    r = doctest.testmod(mod)
    assert r.failed == 0, f"{r.failed} doctest failures in {modname}"


def _kalman_kwargs():
    return {
        "F": np.eye(2),
        "H": np.eye(2),
        "Q": 1e-3 * np.eye(2),
        "R": 1e-2 * np.eye(2),
    }


@pytest.mark.parametrize(
    "est",
    [
        OnlineAR(order=3),
        OnlineVAR(order=1, feature_keys=["a", "b"]),
        KalmanFilter(**_kalman_kwargs()),
        VBAdaptiveKalman(**_kalman_kwargs()),
        InnovationAdaptiveKalman(**_kalman_kwargs()),
        CycleResidual(period=20, n_cycles=3),
    ],
)
def test_check_estimator(est):
    check_estimator(est)


def test_online_ar_matches_batch_ls():
    """``lam=1`` reproduces batch least squares on the same series."""
    rng = np.random.default_rng(0)
    series = rng.normal(size=200).tolist()
    order = 3
    ar = OnlineAR(order=order, lam=1.0, delta=1e10)
    for v in series:
        ar.learn_one({"x": float(v)})
    rows = []
    targets = []
    for t in range(order, len(series)):
        rows.append([series[t - order + j] for j in range(order)])
        targets.append(series[t])
    A = np.asarray(rows)
    y = np.asarray(targets)
    w_batch, *_ = np.linalg.lstsq(A, y, rcond=None)
    assert np.allclose(ar._w, w_batch, atol=1e-3)


def test_online_ar_skips_nan():
    ar = OnlineAR(order=2)
    ar.learn_one({"x": 1.0})
    ar.learn_one({"x": float("nan")})
    ar.learn_one({"x": 2.0})
    ar.learn_one({"x": None})
    assert ar.n_missing == 2


def test_online_ar_step_jittable():
    ar = OnlineAR(order=2)
    check_step_is_jittable(ar)


def test_online_var_runs():
    var = OnlineVAR(order=1, feature_keys=["a", "b"])
    for i in range(50):
        var.learn_one({"a": float(i), "b": float(2 * i)})
    out = var.predict_one({"a": 49.0, "b": 98.0})
    assert set(out.keys()) == {"a", "b"}
    assert all(np.isfinite(v) for v in out.values())


def test_online_var_skips_nan():
    var = OnlineVAR(order=1, feature_keys=["a", "b"])
    var.learn_one({"a": 1.0, "b": 2.0})
    var.learn_one({"a": float("nan"), "b": 2.0})
    var.learn_one({"a": 1.0, "b": None})
    assert var.n_missing == 2


def test_kalman_constant_velocity():
    F = np.array([[1.0, 0.1], [0.0, 1.0]])
    H = np.array([[1.0, 0.0]])
    Q = 1e-4 * np.eye(2)
    R = np.array([[1e-2]])
    kf = KalmanFilter(F=F, H=H, Q=Q, R=R)
    rng = np.random.default_rng(0)
    for i in range(100):
        pos = 1.0 + 0.1 * i + 0.05 * rng.normal()
        kf.learn_one({"pos": float(pos)})
    est = kf.predict_one({})
    assert est.shape == (2,)
    assert abs(est[0] - (1.0 + 0.1 * 99)) < 0.5


def _run_kalman_on_jump(cls, seed=0):
    F = np.array([[1.0, 0.1], [0.0, 1.0]])
    H = np.array([[1.0, 0.0]])
    Q = 1e-4 * np.eye(2)
    R0 = np.array([[0.1**2]])
    kf = cls(F=F, H=H, Q=Q, R=R0)
    rng = np.random.default_rng(seed)
    n = 400
    half = n // 2
    errs = []
    for i in range(n):
        std = 0.1 if i < half else 1.0
        pos = 1.0 + 0.1 * i + std * rng.normal()
        kf.learn_one({"pos": float(pos)})
        if i >= half:
            est = kf.predict_one({})
            errs.append(abs(est[0] - (1.0 + 0.1 * i)))
    return kf, float(np.mean(errs))


def test_kalman_adaptive_follows_R_jump():
    _fixed, err_fixed = _run_kalman_on_jump(KalmanFilter)
    vb, err_vb = _run_kalman_on_jump(VBAdaptiveKalman)
    inno, err_inno = _run_kalman_on_jump(InnovationAdaptiveKalman)
    r_vb = float(vb.R_estimate[0, 0])
    r_inno = float(inno.R[0, 0])
    assert r_vb > 0.5, f"VB-AKF R estimate {r_vb} not tracking the jump"
    assert r_inno > 0.5, f"inno R estimate {r_inno} not tracking the jump"
    assert err_vb < err_fixed, (err_vb, err_fixed)
    assert err_inno < err_fixed, (err_inno, err_fixed)


@pytest.mark.slow
def test_cycle_period_detected(arm_series):
    taus = arm_series
    cr = CycleResidual(period=None, max_period=250, n_cycles=4)
    for v in taus[:800]:
        cr.transform_one({"x": float(v)})
    assert cr.period is not None
    assert 190 <= cr.period <= 210, f"detected {cr.period}"


@pytest.mark.slow
def test_cycle_residual_rises_after_fault(arm_series):
    from dense_armor.utility.drift.detector import CUSUMDriftDetector

    taus = arm_series
    cr = CycleResidual(period=200, n_cycles=2)
    residuals = []
    for v in taus:
        out = cr.transform_one({"x": float(v)})
        residuals.append(out["residual"])
    residuals = np.asarray(residuals)
    fault_idx = 12 * 50
    before = residuals[400:590]
    after = residuals[700:800]
    mean_before = float(np.mean(np.abs(before)))
    mean_after = float(np.mean(np.abs(after)))
    assert mean_after > 2.0 * mean_before, (mean_before, mean_after)
    det = CUSUMDriftDetector(radius=30, ref_mult=3, k=0.5, h=20.0, reference="adaptive")
    first_after = None
    fa_before = 0
    for i, r in enumerate(residuals):
        det.update(float(abs(r)))
        if det.drift_detected:
            if i < fault_idx:
                fa_before += 1
            elif first_after is None:
                first_after = i
    assert first_after is not None
    assert first_after >= fault_idx


def test_online_var_group_lasso_shrinks_only_off_diagonal():
    """Equation (25) of Zaman 2019: `1{n != n'}` keeps the diagonal free."""
    import numpy as np

    from dense_armor.utility.timeseries import OnlineVAR

    # two independent AR(1) channels with strong auto-regression
    rng = np.random.default_rng(0)
    series = np.zeros((400, 2))
    for t in range(1, 400):
        series[t, 0] = 0.9 * series[t - 1, 0] + 0.05 * rng.normal()
        series[t, 1] = 0.9 * series[t - 1, 1] + 0.05 * rng.normal()

    var0 = OnlineVAR(order=1, lam=0.99, group_lambda=0.0, feature_keys=["a", "b"])
    var1 = OnlineVAR(order=1, lam=0.99, group_lambda=0.05, feature_keys=["a", "b"])

    # at step 1 the two must be identical on the diagonal, bit for bit
    var0.learn_one({"a": float(series[0, 0]), "b": float(series[0, 1])})
    var1.learn_one({"a": float(series[0, 0]), "b": float(series[0, 1])})
    assert var0._a[0, 0] == var1._a[0, 0]
    assert var0._a[1, 1] == var1._a[1, 1]

    # run the rest of the stream
    for t in range(1, 400):
        var0.learn_one({"a": float(series[t, 0]), "b": float(series[t, 1])})
        var1.learn_one({"a": float(series[t, 0]), "b": float(series[t, 1])})

    diag0 = np.array([var0._a[0, 0], var0._a[1, 1]])
    diag1 = np.array([var1._a[0, 0], var1._a[1, 1]])
    off0 = np.array([var0._a[0, 1], var0._a[1, 0]])
    off1 = np.array([var1._a[0, 1], var1._a[1, 0]])

    diag_diff = float(np.max(np.abs(diag0 - diag1)))
    off_diff = float(np.max(np.abs(off0 - off1)))

    # the shrinkage does not touch the diagonal: any difference there is
    # feedback from the changed off-diagonal blocks, at least an order of
    # magnitude smaller than the direct shrinkage
    assert diag_diff < 1e-2, f"diagonal moved by {diag_diff}"
    # the off-diagonal blocks are shrunk visibly
    assert off_diff > 1e-2, f"off-diagonal did not move ({off_diff})"


def test_kalman_n_missing_property():
    kf = KalmanFilter(F=np.eye(2), H=np.eye(2), Q=1e-3 * np.eye(2), R=1e-2 * np.eye(2))
    assert kf.n_missing == 0
    kf.learn_one(None, 0.0)
    kf.learn_one({"a": "abc", "b": 1.0}, 0.0)
    assert kf.n_missing == 2


def test_kalman_meas_empty_dict():
    kf = KalmanFilter(F=np.eye(2), H=np.eye(2), Q=1e-3 * np.eye(2), R=1e-2 * np.eye(2))
    kf.learn_one({}, 0.0)
    assert kf.n_missing == 0


def test_kalman_transform_before_learning():
    kf = KalmanFilter(F=np.eye(2), H=np.eye(2), Q=1e-3 * np.eye(2), R=1e-2 * np.eye(2))
    out = kf.transform_one({})
    assert "innovation" in out and "S" in out
    assert out["innovation"].shape == (2,)
    assert out["S"].shape == (2, 2)


def test_vbkalman_rejects_bad_n_iter():
    import pytest

    with pytest.raises(ValueError):
        VBAdaptiveKalman(
            F=np.eye(2), H=np.eye(2), Q=1e-3 * np.eye(2), R=1e-2 * np.eye(2), n_iter=0
        )


def test_innovation_adaptive_rejects_bad_window():
    import pytest

    with pytest.raises(ValueError):
        InnovationAdaptiveKalman(
            F=np.eye(2), H=np.eye(2), Q=1e-3 * np.eye(2), R=1e-2 * np.eye(2), window=1
        )


def test_vbkalman_step_jittable_atol():
    """step exists and returns a scalar for a scalar input."""
    import jax.numpy as jnp

    est = VBAdaptiveKalman(
        F=np.eye(2), H=np.eye(2), Q=1e-3 * np.eye(2), R=1e-2 * np.eye(2)
    )
    state = est.init_state()
    _, out = est.step(state, jnp.array(1.5))
    assert np.isfinite(float(out))


def test_innovation_adaptive_step_runs():
    """step advances the ring buffer and returns a scalar."""
    import jax.numpy as jnp

    est = InnovationAdaptiveKalman(
        F=np.eye(2), H=np.eye(2), Q=1e-3 * np.eye(2), R=1e-2 * np.eye(2), window=5
    )
    state = est.init_state()
    for _ in range(3):
        state, out = est.step(state, jnp.array(1.5))
    assert np.isfinite(float(out))
    _, _, _buf_inn, _buf_res, fill, head = state
    assert int(fill) == 3
    assert int(head) == 3


def test_kalman_predict_one_with_none():
    kf = KalmanFilter(F=np.eye(2), H=np.eye(2), Q=1e-3 * np.eye(2), R=1e-2 * np.eye(2))
    est = kf.predict_one(None)
    assert est.shape == (2,)


def test_kalman_transform_after_learning():
    kf = KalmanFilter(F=np.eye(2), H=np.eye(2), Q=1e-3 * np.eye(2), R=1e-2 * np.eye(2))
    kf.learn_one({"a": 1.0, "b": 2.0}, 0.0)
    out = kf.transform_one({})
    assert out["innovation"].shape == (2,)
    assert out["S"].shape == (2, 2)
    assert np.all(np.isfinite(out["innovation"]))
