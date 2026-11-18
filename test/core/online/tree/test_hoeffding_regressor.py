"""HoeffdingTreeRegressor: variance reduction, E-BST and quantile splitters, categories, memory."""

import pickle

import numpy as np
import pytest

from dense_armor.checks import check_estimator
from dense_armor.core.online.tree.hoeffding_regressor import HoeffdingTreeRegressor


def _step_stream(n, seed=0):
    rng = np.random.default_rng(seed)
    for _ in range(n):
        x = float(rng.random())
        yield (
            {"x": x, "z": float(rng.random())},
            (5.0 if x > 0.5 else 1.0) + float(rng.normal(0, 0.1)),
        )


@pytest.mark.parametrize("splitter", ["ebst", "quantile"])
def test_the_regressor_finds_the_step(splitter):
    t = HoeffdingTreeRegressor(grace_period=100, splitter=splitter)
    for x, y in _step_stream(4000):
        t.learn_one(x, y)
    assert t.root_.split_feature_ == "x"
    assert abs(t.root_.split_threshold_ - 0.5) < 0.1
    assert t.predict_one({"x": 0.9, "z": 0.2}) == pytest.approx(5.0, abs=0.2)
    assert t.predict_one({"x": 0.1, "z": 0.2}) == pytest.approx(1.0, abs=0.2)


def test_return_std_and_explain():
    t = HoeffdingTreeRegressor(grace_period=100)
    assert t.predict_one({"x": 0.1}) == 0.0
    assert t.predict_one({"x": 0.1}, return_std=True) == (0.0, 0.0)
    assert t.explain_one({"x": 0.1}) == []
    for x, y in _step_stream(3000):
        t.learn_one(x, y)
    mean, std = t.predict_one({"x": 0.9, "z": 0.2}, return_std=True)
    assert std < 0.3
    path = t.explain_one({"x": 0.9, "z": 0.2})
    assert path[0]["feature"] == "x" and path[0]["side"] == "right"


def test_categorical_feature_with_variance_reduction():
    rng = np.random.default_rng(0)
    values = {"a": 1.0, "b": 2.0, "c": 7.0, "d": 3.0}
    t = HoeffdingTreeRegressor(grace_period=60, nominal_attributes=["k"])
    for _ in range(1500):
        k = "abcd"[int(rng.integers(4))]
        t.learn_one({"k": k}, values[k] + float(rng.normal(0, 0.1)))
    assert t.root_.split_category_ == "c"
    assert t.predict_one({"k": "c"}) == pytest.approx(7.0, abs=0.3)
    assert t.explain_one({"k": "c"}) == [
        {"feature": "k", "category": "c", "side": "left"}
    ]


def test_a_missing_value_or_target_is_counted():
    t = HoeffdingTreeRegressor()
    t.learn_one({"x": None}, 1.0)
    t.learn_one({"x": float("nan")}, 1.0)
    t.learn_one({"x": "a"}, 1.0)
    t.learn_one({"x": 1.0}, float("nan"))
    t.learn_one({"x": 1.0}, None)
    t.learn_one({}, 1.0)
    assert t.n_missing == 6 and t.n_seen_ == 0


def test_deactivation_keeps_the_depth_and_the_accuracy():
    rng = np.random.default_rng(0)
    free = HoeffdingTreeRegressor(
        grace_period=40, memory_period=200, max_active_leaves=None
    )
    capped = HoeffdingTreeRegressor(
        grace_period=40, memory_period=200, max_active_leaves=3
    )
    for _ in range(15000):
        x = float(rng.random())
        y = float(int(x * 8)) + float(rng.normal(0, 0.05))
        free.learn_one({"x": x}, y)
        capped.learn_one({"x": x}, y)
    assert capped.n_active_leaves_ <= 3
    assert len(list(capped._leaves())) >= 8
    xs = np.random.default_rng(1).random(500)
    err = np.mean([abs(capped.predict_one({"x": float(v)}) - int(v * 8)) for v in xs])
    assert err < 0.5


def test_limits_and_bad_arguments():
    t = HoeffdingTreeRegressor(grace_period=40, max_nodes=3)
    for x, y in _step_stream(3000):
        t.learn_one(x, y)
    assert t.n_nodes_ <= 3
    with pytest.raises(ValueError):
        HoeffdingTreeRegressor(splitter="gaussian")
    with pytest.raises(ValueError):
        HoeffdingTreeRegressor(grace_period=0)
    with pytest.raises(ValueError):
        HoeffdingTreeRegressor(delta=1.0)
    with pytest.raises(ValueError):
        HoeffdingTreeRegressor(nominal_attributes="k")


def test_pickle_roundtrip():
    t = HoeffdingTreeRegressor(grace_period=100, splitter="quantile")
    for x, y in _step_stream(2000):
        t.learn_one(x, y)
    t2 = pickle.loads(pickle.dumps(t))
    assert t2.predict_one({"x": 0.9, "z": 0.1}) == t.predict_one({"x": 0.9, "z": 0.1})


@pytest.mark.parametrize("splitter", ["ebst", "quantile"])
def test_check_estimator_regressor(splitter):
    check_estimator(HoeffdingTreeRegressor(grace_period=20, splitter=splitter))
