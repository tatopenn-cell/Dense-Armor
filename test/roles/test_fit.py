import numpy as np
import pytest

from dense_armor.utility.linear.glm import LinearRegression
from dense_armor.utility.optim.sgd import SGD
from dense_armor.utility.stats.moments import RunningMoments
from dense_armor.utility.tree.hoeffding import HoeffdingTreeClassifier


def _data(n=200):
    rng = np.random.default_rng(0)
    X = [{"a": float(v), "b": float(w)} for v, w in rng.normal(size=(n, 2))]
    y = [2.0 * x["a"] - x["b"] for x in X]
    return X, y


def test_fit_equals_manual_learn_one_loop():
    X, y = _data()
    m1 = LinearRegression(optimizer=SGD()).fit(X, y)
    m2 = LinearRegression(optimizer=SGD())
    for x, v in zip(X, y):
        m2.learn_one(x, v)
    assert m1.predict_one(X[0]) == m2.predict_one(X[0])


def test_fit_returns_self_and_runs_unsupervised():
    r = RunningMoments()
    assert r.fit([{"a": 1.0}, {"a": 3.0}]) is r


def test_fit_epochs_and_shuffle_are_reproducible():
    X, _ = _data()
    c = [int(x["a"] > 0) for x in X]
    p = []
    for _ in range(2):
        m = HoeffdingTreeClassifier(grace_period=20).fit(
            X, c, epochs=2, shuffle=True, seed=1)
        p.append(m.predict_one({"a": 2.0, "b": 0.0}))
    assert p[0] == p[1]


def test_fit_with_timestamps_one_pass():
    X, y = _data(20)
    m = LinearRegression(optimizer=SGD()).fit(X, y, t=list(range(20)))
    assert m.predict_one(X[0]) is not None


@pytest.mark.parametrize("kw", [
    {"epochs": 0},
    {"y": [1.0]},
    {"t": [0.0]},
    {"t": list(range(200)), "epochs": 2},
])
def test_fit_rejects_bad_arguments(kw):
    X, y = _data()
    kw.setdefault("y", y)
    with pytest.raises(ValueError):
        LinearRegression(optimizer=SGD()).fit(X, **kw)


def test_fit_needs_learn_one():
    from dense_armor.roles.root import Root
    with pytest.raises(TypeError):
        Root().fit([{"a": 1.0}])
