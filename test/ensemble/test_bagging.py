"""Online bagging: the Poisson(1) resample and the hard vote."""

import numpy as np

from dense_armor.utility.ensemble import OnlineBaggingClassifier
from dense_armor.utility.ensemble._common import as_dict, finite
from dense_armor.utility.tree.hoeffding import HoeffdingTreeClassifier


def _tree():
    return HoeffdingTreeClassifier(grace_period=20)


class _NoPredict:
    def learn_one(self, x, y, t=None):
        return self

    def predict_one(self, x, t=None):
        raise AttributeError("no predict")


def test_poisson_draw_mean_and_variance():
    rng = np.random.default_rng(0)
    k = rng.poisson(1.0, 100000)
    assert abs(k.mean() - 1.0) < 0.02
    assert abs(k.var() - 1.0) < 0.03


def test_learns_a_simple_split():
    rng = np.random.default_rng(0)
    clf = OnlineBaggingClassifier(_tree, n_models=3, seed=0)
    for _ in range(400):
        x = float(rng.normal(0.0 if rng.random() < 0.5 else 4.0, 0.5))
        y = int(x > 2.0)
        _ = clf.learn_one({"x": x}, y)
    assert clf.predict_one({"x": 0.1}) == 0
    assert clf.predict_one({"x": 4.0}) == 1


def test_nan_sample_is_skipped_and_counted():
    clf = OnlineBaggingClassifier(_tree, n_models=2, seed=0)
    _ = clf.learn_one({"x": 0.0}, 0)
    _ = clf.learn_one({"x": float("nan")}, 0)
    _ = clf.learn_one({"x": float("inf")}, 1)
    _ = clf.learn_one({}, 0)
    _ = clf.learn_one({"x": 1.0}, None)
    assert clf.n_missing == 4
    assert clf.n_seen_ == 1


def test_clone_preserves_constructor():
    clf = OnlineBaggingClassifier(_tree, n_models=5, seed=42)
    c = clf.clone()
    assert type(c) is OnlineBaggingClassifier
    assert c.n_models == 5
    assert c.seed == 42
    assert c.get_params() == clf.get_params()


def test_no_nan_in_proba_after_a_nan_sample():
    clf = OnlineBaggingClassifier(_tree, n_models=2, seed=0)
    for _ in range(20):
        _ = clf.learn_one({"x": 0.5}, 0)
    _ = clf.learn_one({"x": float("nan")}, 0)
    for _ in range(20):
        _ = clf.learn_one({"x": 0.5}, 0)
    for v in clf.predict_proba_one({"x": 0.5}).values():
        assert np.isfinite(v)


def test_predict_before_learning():
    clf = OnlineBaggingClassifier(_tree, n_models=2, seed=0)
    assert clf.predict_one({"x": 0.0}) is None
    assert clf.predict_proba_one({"x": 0.0}) == {}


def test_member_without_predict_returns_none():
    clf = OnlineBaggingClassifier(lambda: _NoPredict(), n_models=2, seed=0)
    _ = clf.learn_one({"x": 0.0}, 0)
    assert clf.predict_one({"x": 0.0}) is None
    assert clf.predict_proba_one({"x": 0.0}) == {}


def test_rejects_zero_models():
    try:
        OnlineBaggingClassifier(_tree, n_models=0)
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


def test_skip_set_is_empty():
    assert (
        OnlineBaggingClassifier(_tree, n_models=2, seed=0)._unit_test_skips() == set()
    )


def test_as_dict_and_finite_helpers():
    class _ToDict:
        def to_dict(self):
            return {"a": 1.0}

    class _Bad:
        def __iter__(self):
            raise TypeError("no")

    assert as_dict(_ToDict()) == {"a": 1.0}
    assert as_dict(_Bad()) == {}
    assert finite({"a": 1.0})
    assert not finite({"a": None})
    assert not finite({"a": "hello"})
    assert not finite({"a": float("nan")})


def test_new_member_deepcopy_when_no_clone():
    from dense_armor.utility.ensemble._common import new_member

    class _Bare:
        def __init__(self):
            self.calls = 0

        def learn_one(self, x, y, t=None):
            self.calls += 1
            return self

        def predict_one(self, x, t=None):
            return 0

    base = _Bare()
    m = new_member(base)
    assert m is not base
    m.learn_one({}, 0)
    assert base.calls == 0
