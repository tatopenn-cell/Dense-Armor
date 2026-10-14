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
    assert np.allclose(ar.coef_, w_batch, atol=1e-3)


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
    """Position error after a jump of the measurement variance from 0.01 to 1.0.

    Seed 0, 400 samples, jump at sample 200: mean absolute position error
    0.2265 (fixed), 0.1504 (VB), 0.1675 (innovation); estimated R 0.811
    and 0.843 for a true value of 1.0.
    """
    _fixed, err_fixed = _run_kalman_on_jump(KalmanFilter)
    vb, err_vb = _run_kalman_on_jump(VBAdaptiveKalman)
    inno, err_inno = _run_kalman_on_jump(InnovationAdaptiveKalman)
    r_vb = float(vb.R_estimate[0, 0])
    r_inno = float(inno.R_estimate[0, 0])
    assert abs(err_fixed - 0.2265) < 0.01, err_fixed
    assert abs(err_vb - 0.1504) < 0.01, err_vb
    assert abs(err_inno - 0.1675) < 0.01, err_inno
    assert abs(r_vb - 0.811) < 0.02, r_vb
    assert abs(r_inno - 0.843) < 0.02, r_inno
    assert err_vb < err_fixed and err_inno < err_fixed


def _literal_vb_akf(rho, zs, F, H, Q, R0, n_iter=5):
    """Algorithm 1 of Mbalawata et al. (2013), equation (22), written as printed."""
    d = H.shape[0]
    nu = d + 2.0
    sigma = R0.copy()
    m = np.zeros(F.shape[0])
    P = np.eye(F.shape[0])
    for z in zs:
        mm = F @ m
        PP = F @ P @ F.T + Q
        nu_new = rho * (nu - d - 1) + d + 2
        s_j = sigma.copy()
        for _ in range(n_iter):
            S = H @ PP @ H.T + s_j
            K = PP @ H.T @ np.linalg.inv(S)
            mj = mm + K @ (z - H @ mm)
            Pj = PP - K @ S @ K.T
            e = z - H @ mj
            s_j = (
                (nu - d - 1) / (nu_new - d - 1) * sigma
                + (H @ Pj @ H.T) / (nu_new - d - 1)
                + np.outer(e, e) / (nu_new - d - 1)
            )
        sigma, m, P, nu = s_j, mj, Pj, nu_new
    return sigma, m


def test_vb_akf_equals_algorithm_1_at_rho_one():
    F = np.array([[1.0, 0.1], [0.0, 1.0]])
    H = np.array([[1.0, 0.0]])
    Q = 1e-4 * np.eye(2)
    R0 = np.array([[0.01]])
    rng = np.random.default_rng(0)
    zs = [
        np.array([1.0 + 0.1 * i + (0.1 if i < 200 else 1.0) * rng.normal()])
        for i in range(400)
    ]
    sigma, m = _literal_vb_akf(1.0, zs, F, H, Q, R0)
    vb = VBAdaptiveKalman(F=F, H=H, Q=Q, R=R0, rho=1.0)
    for z in zs:
        vb.learn_one({"p": float(z[0])})
    assert abs(float(vb.R_estimate[0, 0]) - float(sigma[0, 0])) < 1e-9
    assert float(np.max(np.abs(vb._x - m))) < 1e-9


def test_adaptive_kalman_scan_equals_learn_one():
    import jax
    import jax.numpy as jnp

    F = np.array([[1.0, 0.1], [0.0, 1.0]])
    H = np.array([[1.0, 0.0]])
    Q = 1e-4 * np.eye(2)
    R0 = np.array([[0.01]])
    rng = np.random.default_rng(0)
    zs = [1.0 + 0.1 * i + (0.1 if i < 100 else 1.0) * rng.normal() for i in range(200)]
    for cls in (VBAdaptiveKalman, InnovationAdaptiveKalman):
        a = cls(F=F, H=H, Q=Q, R=R0)
        for z in zs:
            a.learn_one({"p": float(z)})
        b = cls(F=F, H=H, Q=Q, R=R0)
        step = jax.jit(b.step)
        state = b.init_state()
        for z in zs:
            state, _ = step(state, jnp.asarray([z]))
        np.testing.assert_allclose(a._x, np.asarray(state[0]), rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(a.R_estimate, np.asarray(state[6]), rtol=1e-4, atol=1e-6)
    np.testing.assert_allclose(a.Q_estimate, np.asarray(state[7]), rtol=1e-4, atol=1e-8)


def test_innovation_adaptive_keeps_constructor_parameters():
    F = np.eye(2)
    H = np.eye(2)
    Q = 1e-3 * np.eye(2)
    R = 1e-2 * np.eye(2)
    est = InnovationAdaptiveKalman(F=F, H=H, Q=Q, R=R)
    rng = np.random.default_rng(0)
    for _ in range(100):
        est.learn_one({"a": float(rng.normal()), "b": float(rng.normal())})
    assert est.get_params()["R"] == R.tolist()
    assert est.clone().R_estimate.tolist() == R.tolist()
    assert not np.allclose(est.R_estimate, R)


def test_vb_step_is_jittable():
    check_step_is_jittable(VBAdaptiveKalman(**_kalman_kwargs()))


def test_online_ar_transform_before_learn_is_the_residual():
    ar = OnlineAR(order=1)
    ar.learn_one({"x": 1.0})
    ar.learn_one({"x": 2.0})
    forecast = ar.predict_one({})
    assert abs(ar.transform_one({"x": 3.0})["residual"] - (3.0 - forecast)) < 1e-12
    ar.learn_one({"x": 3.0})
    assert ar.predict_one({}) != forecast


def test_online_ar_coefficients_run_from_oldest_to_newest():
    rng = np.random.default_rng(0)
    x = np.zeros(400)
    for t in range(2, 400):
        x[t] = 0.6 * x[t - 1] - 0.3 * x[t - 2] + 0.05 * rng.normal()
    ar = OnlineAR(order=2)
    for v in x:
        ar.learn_one({"x": float(v)})
    assert ar.coef_[1] > 0.4 and ar.coef_[0] < -0.1


def test_online_var_matches_batch_least_squares():
    rng = np.random.default_rng(2)
    A = np.array([[0.5, 0.2], [-0.1, 0.4]])
    Y = np.zeros((2000, 2))
    for t in range(1, 2000):
        Y[t] = A @ Y[t - 1] + 0.1 * rng.normal(size=2)
    var = OnlineVAR(order=1, lam=1.0, group_lambda=0.0, feature_keys=["a", "b"])
    for t in range(2000):
        var.learn_one({"a": float(Y[t, 0]), "b": float(Y[t, 1])})
    batch = np.linalg.lstsq(Y[:-1], Y[1:], rcond=None)[0].T
    assert float(np.max(np.abs(var._a - batch))) < 1e-2


def test_online_var_sigma2_changes_the_estimates():
    rng = np.random.default_rng(1)
    series = 0.1 * rng.normal(size=(30, 2))
    v0 = OnlineVAR(order=1, sigma2=0.0, feature_keys=["a", "b"])
    v1 = OnlineVAR(order=1, sigma2=10.0, feature_keys=["a", "b"])
    for row in series:
        v0.learn_one({"a": float(row[0]), "b": float(row[1])})
        v1.learn_one({"a": float(row[0]), "b": float(row[1])})
    assert float(np.max(np.abs(v0._a - v1._a))) > 1e-6
    with pytest.raises(ValueError):
        OnlineVAR(order=1, sigma2=-1.0)


def test_online_var_group_lasso_zeroes_unrelated_channels():
    rng = np.random.default_rng(0)
    series = np.zeros((600, 2))
    for t in range(1, 600):
        series[t] = 0.9 * series[t - 1] + 0.05 * rng.normal(size=2)
    var = OnlineVAR(order=1, lam=0.99, group_lambda=0.5, feature_keys=["a", "b"])
    for t in range(600):
        var.learn_one({"a": float(series[t, 0]), "b": float(series[t, 1])})
    assert var._a[0, 1] == 0.0 and var._a[1, 0] == 0.0
    assert var._a[0, 0] > 0.8 and var._a[1, 1] > 0.7


@pytest.mark.slow
def test_cycle_period_detected(arm_series):
    cr = CycleResidual(period=None, max_period=250, n_cycles=2)
    for v in arm_series:
        cr.transform_one({"x": float(v)})
    assert cr.period == 200


@pytest.mark.slow
def test_cycle_residual_rises_after_fault(arm_series):
    """tau_1 of the arm, 800 samples, 5 kg payload fault at sample 600.

    The period is found (200), the template is ready at sample 499. Before
    the fault the residual has mean 0.015 and standard deviation 0.060;
    from sample 650 it has mean -6.35 and standard deviation 15.5.
    """
    from dense_armor.utility.drift.detector import CUSUMDriftDetector

    cr = CycleResidual(period=None, max_period=250, n_cycles=2)
    residuals = np.array(
        [cr.transform_one({"x": float(v)})["residual"] for v in arm_series]
    )
    fault_idx = 12 * 50
    valid = np.where(~np.isnan(residuals))[0]
    before = residuals[valid[valid < fault_idx]]
    after = residuals[valid[valid >= fault_idx + 50]]
    assert valid[0] == 500
    assert (
        abs(float(before.mean()) - 0.015) < 0.02
        and abs(float(before.std()) - 0.060) < 0.02
    )
    assert (
        abs(float(after.mean()) + 6.35) < 0.5 and abs(float(after.std()) - 15.5) < 1.0
    )
    det = CUSUMDriftDetector(radius=30, ref_mult=3, k=0.5, h=20.0, reference="adaptive")
    fa_before, first_after = 0, None
    for i in valid:
        det.update(float(abs(residuals[i])))
        if det.drift_detected:
            if i < fault_idx:
                fa_before += 1
            elif first_after is None:
                first_after = int(i)
    assert fa_before == 0
    assert first_after is not None and first_after >= fault_idx


def test_cycle_residual_is_nan_until_the_template_exists():
    cr = CycleResidual(period=20, n_cycles=3)
    outs = [
        cr.transform_one({"x": float(np.sin(2 * np.pi * i / 20))}) for i in range(80)
    ]
    assert all(np.isnan(o["residual"]) and o["ready"] == 0.0 for o in outs[:60])
    assert all(o["ready"] == 1.0 for o in outs[61:])
    assert max(abs(o["residual"]) for o in outs[60:]) < 1e-9


def test_cycle_template_follows_a_change_of_amplitude():
    sig = np.sin(2 * np.pi * np.arange(2000) / 200)
    sig[900:] *= 1.5
    cr = CycleResidual(period=200, n_cycles=2)
    res = np.array([cr.transform_one({"x": float(v)})["residual"] for v in sig])
    assert float(np.nanmean(np.abs(res[900:1000]))) > 0.3
    assert float(np.nanmean(np.abs(res[1500:]))) < 0.05


def test_cycle_phase_search_removes_a_shift():
    base = np.sin(2 * np.pi * np.arange(2000) / 200)
    base[1000:] = np.sin(2 * np.pi * (np.arange(1000, 2000) + 7) / 200)
    plain = CycleResidual(period=200, n_cycles=2, search=0)
    aligned = CycleResidual(period=200, n_cycles=2, search=10)
    r0 = np.array([plain.transform_one({"x": float(v)})["residual"] for v in base])
    r1 = np.array([aligned.transform_one({"x": float(v)})["residual"] for v in base])
    assert float(np.nanmean(np.abs(r0[1200:1400]))) > 0.05
    assert float(np.nanmean(np.abs(r1[1200:1400]))) < 0.01


def test_cycle_memory_is_one_array_of_cycles():
    cr = CycleResidual(period=50, n_cycles=3, max_period=1000)
    for i in range(5000):
        cr.transform_one({"x": float(np.sin(2 * np.pi * i / 50))})
    assert cr._ring.shape == (3, 50)
    assert cr._buf == []


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
    _, _, _buf_inn, _buf_res, fill, head, _R, _Q = state
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


@pytest.mark.parametrize("kwargs", [{"order": 0}, {"lam": 0.0}, {"lam": 1.5}])
def test_online_ar_rejects_bad_arguments(kwargs):
    with pytest.raises(ValueError):
        OnlineAR(**kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [{"order": 0}, {"lam": 0.0}, {"group_lambda": -1.0}, {"alpha": 0.0}],
)
def test_online_var_rejects_bad_arguments(kwargs):
    with pytest.raises(ValueError):
        OnlineVAR(**kwargs)


def test_online_ar_before_any_sample_and_with_bad_input():
    ar = OnlineAR(order=2, feature="x")
    assert ar.coef_.tolist() == [0.0, 0.0]
    assert ar.predict_one({"x": 1.0}) == 0.0
    ar.learn_one("not a dict")
    ar.learn_one({"y": 1.0})
    assert ar.n_missing == 2
    assert np.isnan(ar.transform_one({"x": float("nan")})["residual"])
    ar.learn_one({"x": 1.0})
    assert ar.predict_one({"x": 1.0}) == 0.0


def test_online_var_with_sorted_keys_and_bad_input():
    var = OnlineVAR(order=1)
    assert var.predict_one("not a dict") == {}
    assert var.predict_one({"b": 0.0, "a": 0.0}) == {"a": 0.0, "b": 0.0}
    var.learn_one({"b": 2.0, "a": 1.0})
    var.learn_one("not a dict")
    assert var.n_missing == 1
    assert var.transform_one({"a": None, "b": 1.0}) == {}
    out = var.transform_one({"a": 1.0, "b": 2.0})
    assert set(out) == {"a", "b"} and all(np.isfinite(v) for v in out.values())


@pytest.mark.parametrize(
    "kwargs",
    [{"period": 0}, {"max_period": 1}, {"n_cycles": 0}, {"search": -1}],
)
def test_cycle_residual_rejects_bad_arguments(kwargs):
    with pytest.raises(ValueError):
        CycleResidual(**kwargs)


def test_cycle_residual_counts_missing_samples():
    cr = CycleResidual(period=5, n_cycles=2, feature="x")
    assert cr.n_missing == 0 and not cr.ready
    out = cr.transform_one({"x": float("nan")})
    assert np.isnan(out["residual"]) and out["ready"] == 0.0
    cr.transform_one("not a dict")
    cr.transform_one({"y": 1.0})
    assert cr.n_missing == 3
    for i in range(10):
        cr.transform_one({"x": float(i % 5)})
    assert cr.ready


def test_cycle_residual_without_a_period_keeps_a_bounded_buffer():
    cr = CycleResidual(period=None, max_period=10, n_cycles=2)
    for v in np.arange(300.0):
        cr.transform_one({"x": float(v)})
    assert cr.period is None and len(cr._buf) <= 20
    flat = CycleResidual(period=None, max_period=10, n_cycles=2)
    for _ in range(60):
        flat.transform_one({"x": 1.0})
    assert flat.period is None
