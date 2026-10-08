import numpy as np
import pytest

from dense_armor.checks import check_estimator
from dense_armor.utility.preprocessing.scale import (
    EWScaler,
    MinMaxScaler,
    RobustScaler,
    StandardScaler,
)


def test_standard_matches_batch():
    rng = np.random.default_rng(0)
    data = rng.normal(size=200)
    sc = StandardScaler()
    for v in data[:-1]:
        sc.learn_one({"x": float(v)})
    got = sc.transform_one({"x": float(data[-1])})["x"]
    mu = data[:-1].mean()
    sd = data[:-1].std(ddof=1)
    want = (data[-1] - mu) / sd
    assert got == pytest.approx(want, rel=1e-9, abs=1e-9)


def test_standard_per_joint_vector_matches_numpy():
    rng = np.random.default_rng(0)
    data = rng.normal(size=(100, 3))
    sc = StandardScaler()
    for row in data[:-1]:
        sc.learn_one({"q": row.tolist()})
    got = sc.transform_one({"q": data[-1].tolist()})["q"]
    mu = data[:-1].mean(axis=0)
    sd = data[:-1].std(axis=0, ddof=1)
    want = (data[-1] - mu) / sd
    assert got == pytest.approx(want.tolist(), rel=1e-9, abs=1e-9)


def test_standard_unseen_passes():
    sc = StandardScaler()
    assert sc.transform_one({"y": 3.0}) == {"y": 3.0}


def test_ew_scaler_hand_case():
    sc = EWScaler(alpha=0.5)
    for v in [1.0, 2.0, 3.0]:
        sc.learn_one({"x": v})
    got = sc.transform_one({"x": 3.0})["x"]
    var = 0.5 * (0.25 + 0.5 * 2.25)
    assert got == pytest.approx(0.75 / var ** 0.5, rel=1e-12)


def test_robust_scaler_matches_numpy():
    data = [0.1, 0.5, 0.3, 0.8, 0.2, 0.9, 0.4, 0.6]
    sc = RobustScaler(window=8)
    for v in data:
        sc.learn_one({"x": v})
    arr = np.asarray(data)
    med = float(np.median(arr))
    mad_raw = float(np.median(np.abs(arr - med)))
    mad = 1.4826 * mad_raw
    got = sc.transform_one({"x": 0.6})["x"]
    want = (0.6 - med) / mad
    assert got == pytest.approx(want, rel=1e-12)


def test_robust_scaler_flat_window_passes():
    sc = RobustScaler(window=5)
    for v in [1.0, 1.0, 1.0, 1.0, 1.0]:
        sc.learn_one({"x": v})
    assert sc.transform_one({"x": 1.0})["x"] == 1.0


def test_minmax_scaler_range():
    sc = MinMaxScaler()
    for v in [1.0, 2.0, 3.0, 4.0, 5.0]:
        sc.learn_one({"x": v})
    assert sc.transform_one({"x": 1.0})["x"] == 0.0
    assert sc.transform_one({"x": 5.0})["x"] == 1.0


def test_check_estimator_standard():
    check_estimator(StandardScaler())


def test_check_estimator_ew():
    check_estimator(EWScaler())


def test_check_estimator_minmax():
    check_estimator(MinMaxScaler())


def test_check_estimator_robust():
    check_estimator(RobustScaler())
