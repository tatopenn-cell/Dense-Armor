"""Tests for BatchedLandmarkShield."""

import numpy as np

from dense_armor.checks import check_estimator
from dense_armor.core.online.timeseries.kalman import KalmanFilter
from dense_armor.core.signals.vision.landmarks import (
    BatchedLandmarkShield,
    _hampel_batch,
    _hampel_value,
)

_TOL_1E_9 = 1e-9
_TOL_1E_6 = 1e-6
_OUTLIER_TOL = 1.0
_CLEANED_BOTH = 2


def _frame(pts, conf, i, fps=30.0):
    return {"points": pts, "confidence": conf, "t": float(i) / fps}


def test_batched_landmark_shield_clean_stream():
    shield = BatchedLandmarkShield(window=7, n_sigmas=3.0, max_gap=5)
    n = 10
    pts = np.tile(np.arange(n, dtype=np.float32).reshape(n, 1), (1, 2)) * 10.0
    conf = np.ones(n, dtype=np.float32)
    for i in range(20):
        out = shield.transform_one(_frame(pts, conf, i))
        assert np.abs(out["points"] - pts).max() < _TOL_1E_6
        assert out["cleaned"].sum() == 0


def test_batched_landmark_shield_one_frame_outlier():
    shield = BatchedLandmarkShield(window=7, n_sigmas=3.0, max_gap=5)
    base = np.tile(np.arange(5, dtype=np.float32).reshape(5, 1), (1, 2)) * 10.0
    conf = np.ones(5, dtype=np.float32)
    for i in range(10):
        shield.transform_one(_frame(base, conf, i))
    bad = base.copy()
    bad[2] = [999.0, 999.0]
    out = shield.transform_one(_frame(bad, conf, 10))
    assert np.abs(out["points"][2] - base[2]).max() < _OUTLIER_TOL
    assert out["cleaned"][2] == _CLEANED_BOTH


def test_batched_landmark_shield_moving_occluder_static_face():
    shield = BatchedLandmarkShield(window=7, n_sigmas=3.0, max_gap=5)
    n = 8
    base = np.zeros((n, 2), dtype=np.float32)
    base[:, 0] = np.arange(n) * 10.0
    conf = np.ones(n, dtype=np.float32)
    for i in range(10):
        shield.transform_one(_frame(base, conf, i))
    max_err = 0.0
    for k in range(20):
        c = conf.copy()
        s = k % (n - 1)
        c[s] = 0.0
        c[s + 1] = 0.0
        out = shield.transform_one(_frame(base, c, 10 + k))
        for p in (s, s + 1):
            max_err = max(max_err, float(np.linalg.norm(out["points"][p] - base[p])))
            assert out["cleaned"][p] == _CLEANED_BOTH
    assert max_err < 1.0


def test_batched_landmark_shield_constant_velocity_during_gap():
    shield = BatchedLandmarkShield(window=7, n_sigmas=3.0, max_gap=10)
    conf = np.ones(1, dtype=np.float32)
    for i in range(60):
        shield.transform_one(
            _frame(np.array([[i * 3.0, 0.0]], dtype=np.float32), conf, i)
        )
    off = np.zeros(1, dtype=np.float32)
    for i in range(10):
        out = shield.transform_one(
            _frame(np.array([[0.0, 0.0]], dtype=np.float32), off, 60 + i)
        )
        assert abs(out["points"][0, 0] - (177.0 + (i + 1) * 3.0)) < 1.0


def test_batched_landmark_shield_variable_fps():
    shield = BatchedLandmarkShield(window=7, n_sigmas=3.0, max_gap=10)
    conf = np.ones(1, dtype=np.float32)
    for i in range(60):
        pt = np.array([[i * 3.0, 0.0]], dtype=np.float32)
        shield.transform_one(_frame(pt, conf, i, fps=15.0))
    off = np.zeros(1, dtype=np.float32)
    for i in range(5):
        pt = np.array([[0.0, 0.0]], dtype=np.float32)
        out = shield.transform_one(_frame(pt, off, 60 + i, fps=15.0))
        assert abs(out["points"][0, 0] - (177.0 + (i + 1) * 3.0)) < 1.0


def test_batched_landmark_shield_gap_decays_to_neutral():
    neutral = np.array([[0.0, 0.0], [10.0, 0.0], [20.0, 0.0]], dtype=np.float32)
    shield = BatchedLandmarkShield(window=7, n_sigmas=3.0, max_gap=3, neutral=neutral)
    conf = np.ones(3, dtype=np.float32)
    moved = neutral + 5.0
    for i in range(10):
        shield.transform_one(_frame(neutral, conf, i))
    for i in range(5):
        shield.transform_one(_frame(moved, conf, 10 + i))
    off = np.zeros(3, dtype=np.float32)
    prev = [None] * 3
    for i in range(15):
        out = shield.transform_one(_frame(moved, off, 15 + i))
        if i > 3:
            for p in range(3):
                d = float(np.linalg.norm(out["points"][p] - neutral[p]))
                assert d < float(np.linalg.norm(out["points"][p] - moved[p]))
                if prev[p] is not None and prev[p] > 1e-3:
                    assert d < prev[p] - _TOL_1E_9
                prev[p] = d


def test_batched_landmark_shield_neutral_first_frame():
    shield = BatchedLandmarkShield()
    base = np.array([[5.0, 7.0], [10.0, 3.0], [20.0, 9.0]], dtype=np.float32)
    shield.transform_one(_frame(base, np.ones(3, dtype=np.float32), 0))
    assert np.allclose(shield.neutral, base, atol=_TOL_1E_6)


def test_batched_landmark_shield_neutral_partial_occlusion():
    shield = BatchedLandmarkShield()
    base = np.array([[5.0, 7.0], [10.0, 3.0], [20.0, 9.0]], dtype=np.float32)
    conf = np.ones(3, dtype=np.float32)
    conf[0] = 0.0
    out = shield.transform_one(_frame(base, conf, 0))
    assert np.isnan(shield.neutral[0]).all()
    assert np.allclose(shield.neutral[1:], base[1:], atol=_TOL_1E_6)
    assert np.allclose(out["points"][0], base[0])
    shield.transform_one(_frame(base, np.ones(3, dtype=np.float32), 1))
    assert np.allclose(shield.neutral[0], base[0], atol=_TOL_1E_6)


def test_batched_landmark_shield_point_occluded_before_first_measurement():
    shield = BatchedLandmarkShield()
    pts = np.array([[50.0, 60.0], [10.0, 3.0]], dtype=np.float32)
    out = shield.transform_one(_frame(pts, np.array([0.0, 1.0], dtype=np.float32), 0))
    assert np.allclose(out["points"][0], [50.0, 60.0])


def test_batched_landmark_shield_clone_with_neutral_then_long_gap():
    neutral = np.zeros((3, 2))
    shield = BatchedLandmarkShield(neutral=neutral, max_gap=2).clone()
    pts = np.array([[5.0, 7.0], [10.0, 3.0], [20.0, 9.0]], dtype=np.float32)
    for i in range(5):
        shield.transform_one(_frame(pts, np.ones(3, dtype=np.float32), i))
    for i in range(6):
        out = shield.transform_one(_frame(pts, np.zeros(3, dtype=np.float32), 5 + i))
    d_neutral = np.linalg.norm(out["points"] - neutral, axis=1)
    d_last = np.linalg.norm(out["points"] - pts, axis=1)
    assert np.all(d_neutral < d_last)


def test_batched_landmark_shield_hampel_history_reset_after_gap():
    shield = BatchedLandmarkShield(window=7, n_sigmas=3.0, max_gap=20)
    at = np.array([[10.0, 10.0]], dtype=np.float32)
    far = np.array([[40.0, 10.0]], dtype=np.float32)
    for i in range(10):
        shield.transform_one(_frame(at, np.ones(1, dtype=np.float32), i))
    for i in range(10):
        shield.transform_one(_frame(at, np.zeros(1, dtype=np.float32), 10 + i))
    out = shield.transform_one(_frame(far, np.ones(1, dtype=np.float32), 20))
    assert out["cleaned"][0] == 0


def test_check_estimator_batched_landmark_shield():
    shield = BatchedLandmarkShield()
    result = check_estimator(shield)
    print(
        f"check_estimator result: failed={result['failed']}, not_applicable={result['not_applicable']}, skipped={result['skipped']}"
    )
    assert result["failed"] == [], result["failed"]
    assert result["not_applicable"] == [], result["not_applicable"]
    assert shield._unit_test_skips() == set()


def test_batched_landmark_shield_matches_dense_armor_kalman_filter():
    dt = 1.0 / 30.0
    f = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1.0]])
    h = np.array([[1.0, 0, 0, 0], [0, 1.0, 0, 0]])
    q = np.diag([1e-6, 1e-6, 1e-4, 1e-4])
    r = np.diag([1e-3, 1e-3])
    rng = np.random.default_rng(2)
    track = np.cumsum(rng.normal(0.5, 0.05, size=(60, 2)), axis=0)
    shield = BatchedLandmarkShield(n_sigmas=1e9)
    kf = KalmanFilter(
        F=f,
        H=h,
        Q=q,
        R=r,
        x0=np.r_[track[0], 0.0, 0.0],
        P0=np.diag([1e-6, 1e-6, 1e-2, 1e-2]),
    )
    kf.learn_one({"x": track[0, 0], "y": track[0, 1]})
    kf._x = np.r_[track[0], 0.0, 0.0]
    kf._P = np.diag([1e-6, 1e-6, 1e-2, 1e-2])
    for i, pt in enumerate(track):
        out = shield.transform_one(_frame(pt[None, :], np.ones(1), i))["points"][0]
        if i:
            kf.learn_one({"x": pt[0], "y": pt[1]})
            assert np.abs(out - kf._x[:2]).max() < 1e-3


def test_batched_landmark_shield_hampel_matches_landmark_cleaner():
    """BatchedLandmarkShield's Hampel rule matches LandmarkCleaner's per-point rule."""
    rng = np.random.default_rng(1)
    hist = rng.normal(0.0, 1.0, size=(40, 7))
    hist[:5, :5] = np.nan
    hist[5:10] = 2.0
    vals = rng.normal(0.0, 4.0, size=40)
    filt, flag = _hampel_batch(hist, vals, 3.0)
    for k in range(40):
        window = [float(v) for v in hist[k] if not np.isnan(v)]
        want, was = _hampel_value(window, float(vals[k]), 3.0)
        assert flag[k] == was
        assert abs(filt[k] - want) < _TOL_1E_6


def test_batched_landmark_shield_p99_performance():
    """Measure p99 latency for 68 points."""
    import time

    shield = BatchedLandmarkShield(window=7, n_sigmas=3.0, max_gap=5)
    base = np.random.randn(68, 2).astype(np.float32)
    conf = np.ones(68, dtype=np.float32)
    latencies = []
    for i in range(100):
        frame = {
            "points": base + 0.01 * np.random.randn(68, 2).astype(np.float32),
            "confidence": conf,
            "t": float(i) / 30.0,
        }
        t0 = time.perf_counter()
        shield.transform_one(frame)
        latencies.append(time.perf_counter() - t0)

    p99 = np.percentile(latencies, 99)
    print(f"BatchedLandmarkShield p99: {p99:.6f} s")
    assert p99 < shield.budget_s
