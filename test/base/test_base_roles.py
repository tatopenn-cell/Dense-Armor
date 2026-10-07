"""Direct tests of dense_armor.base roles, Protected and Base utilities."""
import pickle

import numpy as np
import pytest

from dense_armor.base import (
    AnomalyDetector, AnomalyFilter, Base, Classifier, DriftDetector, Protected,
    Regressor, Transformer, TransformerSupervised,
)


class MeanReg(Regressor):
    _mutable_attributes = frozenset({"shrink"})

    def __init__(self, shrink: float = 0.0):
        self.shrink = shrink
        self.n_ = 0
        self.mean_ = 0.0
        self.m2_ = 0.0

    def learn_one(self, x, y, t=None):
        self.n_ += 1
        d = float(y) - self.mean_
        self.mean_ += d / self.n_
        self.m2_ += d * (float(y) - self.mean_)
        return self

    def predict_one(self, x, t=None, return_std=False):
        if return_std:
            return self.mean_, (self.m2_ / self.n_) ** 0.5 if self.n_ else 0.0
        return self.mean_


class Scale(Transformer):
    def __init__(self, k: float = 2.0):
        self.k = k

    def transform_one(self, x, t=None):
        return {key: self.k * v for key, v in x.items()}


class Abs(AnomalyDetector):
    def __init__(self, limit: float = 3.0):
        self.limit = limit

    def learn_one(self, x, t=None):
        return self

    def score_one(self, x, t=None):
        return abs(float(next(iter(x.values()))))


class Gate(AnomalyFilter):
    def classify(self, score):
        return score > 3.0


class Majority(Classifier):
    def __init__(self):
        self.counts_ = {}

    def learn_one(self, x, y, t=None):
        self.counts_[y] = self.counts_.get(y, 0) + 1
        return self

    def predict_proba_one(self, x, t=None):
        n = sum(self.counts_.values())
        return {k: v / n for k, v in self.counts_.items()} if n else {}


class Jump(DriftDetector):
    def __init__(self, h: float = 5.0):
        super().__init__()
        self.h = h

    def update(self, x, t=None):
        self._drift_detected = abs(x) > self.h
        return self


def test_regressor_batch_and_std():
    r = MeanReg()
    r.learn_many(np.array([[0.0], [0.0], [0.0]]), np.array([1.0, 2.0, 3.0]))
    assert r.predict_one({"a": 0.0}) == pytest.approx(2.0)
    mean, std = r.predict_one({"a": 0.0}, return_std=True)
    assert std == pytest.approx(np.std([1.0, 2.0, 3.0]))
    assert len(r.predict_many(np.zeros((4, 1)))) == 4


def test_transformer_one_and_many():
    s = Scale(3.0)
    assert s.learn_one({"a": 1.0}).transform_one({"a": 1.0}) == {"a": 3.0}
    out = s.learn_many([{"a": 1.0}, {"a": 2.0}]).transform_many([{"a": 1.0}, {"a": 2.0}])
    assert len(out) == 2


def test_anomaly_filter_protects_detector():
    f = Gate(Abs(), protect=True)
    assert f.score_one({"v": 5.0}) == 5.0
    assert f.classify(f.score_one({"v": 5.0})) is True
    assert f.learn_one({"v": 5.0}) is f
    assert len(Abs().score_many([{"v": 1.0}, {"v": -4.0}])) == 2


def test_classifier_defaults_and_many():
    c = Majority()
    assert c.predict_one({"a": 1.0}) is None
    c.learn_many(np.zeros((3, 1)), np.array([1, 1, 0]))
    assert c.predict_one({"a": 1.0}) == 1
    assert len(c.predict_many(np.zeros((2, 1)))) == 2
    assert len(c.predict_proba_many(np.zeros((2, 1)))) == 2


def test_drift_detector_update_many():
    d = Jump()
    d.update_many(np.array([0.0, 9.0]))
    assert bool(d.drift_detected) is True


def test_protected_skips_flagged_and_falls_back():
    p = Protected(MeanReg(), Gate(Abs()), fallback=-1.0)
    assert p.predict_one({"v": 10.0}) == -1.0
    p.learn_one({"v": 1.0}, 4.0)
    p.learn_one({"v": 10.0}, 1000.0)
    assert p.predict_one({"v": 1.0}) == pytest.approx(4.0)
    assert p.predict_one({"v": 10.0}) == pytest.approx(4.0)
    assert p.score_one({"v": 10.0}) == float("inf")


def test_base_params_clone_mutate_pickle_state():
    r = MeanReg(shrink=0.1)
    r.learn_one({"a": 0.0}, 5.0)
    assert r.get_params()["shrink"] == 0.1
    assert r.set_params(shrink=0.2).shrink == 0.2
    c = r.clone()
    assert c.n_ == 0 and c.shrink == 0.2
    assert r.clone(include_attributes=True).n_ == 1
    r.mutate({"shrink": 0.3})
    with pytest.raises(ValueError):
        r.mutate({"nope": 1})
    r2 = pickle.loads(pickle.dumps(r))
    assert r2.predict_one({}) == r.predict_one({})
    m = MeanReg().load_state_dict(r.state_dict())
    assert m.predict_one({}) == pytest.approx(5.0)
    assert "MeanReg" in repr(r) and isinstance(r.describe(), dict)
    assert isinstance(r._memory_usage, str) and r._raw_memory_usage > 0


def test_time_base_and_describe():
    b = Base()
    assert isinstance(b.describe(), dict)


def test_time_step_dt_rate_jitter_and_windows():
    b = MeanReg()
    for i in range(20):
        b._time_step(i * 0.01)
    assert b.dt == pytest.approx(0.01)
    assert b.rate == pytest.approx(100.0)
    assert b.jitter == pytest.approx(0.0, abs=1e-12)
    assert b.window_samples(0.5) == 50
    with pytest.raises(ValueError):
        b._time_step(0.0)
    assert MeanReg().dt is None


def test_repr_floats_nested_and_stochastic():
    class P(Base):
        def __init__(self, a=0.0, b=100000.0, c=-2e-7, inner=None, seed=None):
            self.a, self.b, self.c, self.inner, self.seed = a, b, c, inner, seed
    r = repr(P(inner=MeanReg()))
    assert "a=0.0" in r and "b=100000.0" in r and "c=-2e-07" in r and "MeanReg" in r
    assert str(P()) == "P"
    assert P()._is_stochastic() is True
    assert P(seed=1)._is_stochastic() is False


def test_union_and_product_of_transformers():
    u = Scale(2.0) + Scale(3.0)
    assert u.learn_one({"a": 1.0}).transform_one({"a": 1.0}) == {"a": 3.0}
    p = Scale(2.0) * Scale(3.0)
    assert isinstance(p.transform_one({"a": 1.0}), dict)


def test_call_logger_logs_public_calls(caplog):
    import logging
    with caplog.at_level(logging.DEBUG, logger="dense_armor.base"):
        with Base._call_logger(klass=MeanReg):
            MeanReg().learn_one({"a": 0.0}, 1.0)
    assert any("learn_one" in m for m in caplog.messages)
