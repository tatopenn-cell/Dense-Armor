"""Regression pins for N10.

Pin the observable behaviour of RunningMoments, RollingMAD, OnlineAR and
KalmanFilter on fixed inputs, before N10 touches any file. Every value in
this file was measured on master at commit 213dbb1 (1.2.0). If a future
change moves one of these numbers, the test fails and the change is either
reverted or declared.
"""

import numpy as np
import pytest

from dense_armor.utility.stats import RollingMAD, RunningMoments
from dense_armor.utility.timeseries import KalmanFilter, OnlineAR


def test_running_moments_pinned():
    rm = RunningMoments()
    for v in (1.0, 2.0, 3.0, 4.0, 5.0):
        rm.learn_one({"x": v})
    assert rm.count == 5
    assert rm.mean == pytest.approx(3.0, abs=1e-12)
    assert rm.var == pytest.approx(2.5, abs=1e-12)
    assert rm.std == pytest.approx(1.5811388300841898, abs=1e-12)
    assert rm.skewness == pytest.approx(0.0, abs=1e-12)
    assert rm.kurtosis == pytest.approx(-1.3, abs=1e-12)
    assert rm.min == pytest.approx(1.0)
    assert rm.max == pytest.approx(5.0)


def test_rolling_mad_pinned():
    rmad = RollingMAD(window_s=5.0)
    for i, v in enumerate((1.0, 2.0, 3.0, 4.0, 5.0, 100.0, 2.0, 3.0)):
        rmad.learn_one({"x": v}, t=float(i + 1))
    assert rmad.count == 8
    assert rmad.value == pytest.approx(1.4826, abs=1e-9)
    assert rmad.window_s == 5.0


def test_online_ar_pinned():
    ar = OnlineAR(order=2)
    for v in (0.1, 0.2, 0.3, 0.25, 0.15, 0.18, 0.22, 0.19, 0.21, 0.20):
        ar.learn_one({"x": v})
    assert np.allclose(
        np.asarray(ar.coef_),
        np.array([-0.22525470995545976, 1.169057542806564]),
        atol=1e-10,
    )
    assert ar.predict_one({"x": 0.2}) == pytest.approx(
        0.18650801947066625, abs=1e-10
    )


def test_kalman_filter_pinned():
    F = np.array([[1.0]])
    H = np.array([[1.0]])
    Q = np.array([[0.01]])
    R = np.array([[0.1]])
    kf1 = KalmanFilter(F=F, H=H, Q=Q, R=R)
    kf2 = KalmanFilter(F=F, H=H, Q=Q, R=R)
    stream = (1.0, 1.05, 0.98, 1.02, 1.01, 0.99, 1.0, 1.03, 0.97, 1.01)
    for v in stream:
        kf1.learn_one({"x": v})
        kf2.learn_one({"x": v})
    snapshot_1 = {k: np.array(v, copy=True) for k, v in kf1.state_dict().items()
                  if isinstance(v, (np.ndarray, float, int))}
    snapshot_2 = {k: np.array(v, copy=True) for k, v in kf2.state_dict().items()
                  if isinstance(v, (np.ndarray, float, int))}
    for key in snapshot_1:
        assert np.allclose(snapshot_1[key], snapshot_2[key], atol=0.0), (
            f"KalmanFilter is not deterministic on {key}"
        )
