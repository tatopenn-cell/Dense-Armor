"""Edge paths of the role foundation: repr helpers, params, pipelines, safety, check_estimator."""
import pickle

import pytest

from dense_armor.checks import check_estimator
from dense_armor.roles import Classifier, Regressor, SafeEstimator, Transformer
from dense_armor.roles import root as root_mod


class _Reg(Regressor):
    def __init__(self, a=1.0, items=None, *args, **kwargs):
        self.a = a
        self.items = items
        self.args = args
        self.kwargs = kwargs

    def learn_one(self, x, y, t=None):
        return self

    def predict_one(self, x, t=None):
        return 0.5


class _Double(Transformer):
    def __init__(self):
        pass

    def transform_one(self, x, t=None):
        return {k: 2 * v for k, v in x.items()}


class _Clf(Classifier):
    def __init__(self):
        pass

    def learn_one(self, x, y, t=None):
        return self

    def predict_proba_one(self, x, t=None):
        return {0: 0.25, 1: 0.75}


def test_repr_helpers():
    assert root_mod._format_float(float("nan")) == "nan"
    assert root_mod._format_float(float("inf")) == "inf"
    assert root_mod._format_float(float("-inf")) == "-inf"
    for v in ({1, 2}, (1,), (1, 2), {"b": 1, "a": 2}, int, len):
        assert isinstance(root_mod._short_repr(v, 0), str)


def test_params_set_clone_mutate_errors():
    r = _Reg(a=2.0, items={1, 2})
    assert r.get_params(deep=False)["a"] == 2.0
    repr(r)
    with pytest.raises(ValueError):
        r.set_params(nope=1)
    with pytest.raises(ValueError):
        r.clone(new_params={"nope": 1})
    with pytest.raises(ValueError):
        r.mutate({"nope": 1})


def test_time_helpers_and_describe():
    r = _Reg()
    assert r.jitter is None
    assert r.window_samples(3.4) == 3
    d = r.describe()
    assert isinstance(d, dict)
    assert isinstance(r._memory_usage, str)


def test_pipeline_paths():
    p = _Double() | _Reg()
    assert isinstance(p, Regressor)
    p.learn_one({"x": 1.0}, 2.0, t=0.1)
    p.learn_one({"x": 1.0}, 2.0)
    assert p.predict_one({"x": 1.0}) == 0.5
    assert "_Double" in repr(p)
    q = _Double() | _Clf()
    assert q.predict_proba_one({"x": 1.0}) == {0: 0.25, 1: 0.75}
    assert q.predict_proba_one({"x": 1.0}, t=0.2) == {0: 0.25, 1: 0.75}
    t2 = _Double() | _Double()
    assert t2.transform_one({"x": 1.0}) == {"x": 4.0}
    assert _Reg().__or__(object()) is NotImplemented
    assert p.__or__(object()) is NotImplemented
    assert (p | _Reg()).steps[-1][0] == "_Reg"


class _OldSig:
    def __init__(self):
        self.seen = []

    def learn_one(self, x):
        self.seen.append(x)

    def predict_one(self, x, t=None):
        return "bad"


def test_safe_estimator_fallbacks(tmp_path):
    m = _OldSig()
    s = SafeEstimator(m, guard=lambda x: x.get("bad", False))
    s.learn_one({"x": 1.0}, 2.0)
    assert m.seen == [{"x": 1.0}]
    assert s.predict_one({"x": 1.0}) == "bad"
    assert s.predict_one({"bad": True}) == "bad"
    assert s.score_one({"x": 1.0}) != s.score_one({"x": 1.0})
    a = SafeEstimator(_Reg())
    p = a.save(tmp_path / "cp.pkl")
    b = SafeEstimator(_Clf())
    with pytest.raises(TypeError):
        b.restore(p)


class _Broken(Regressor):
    def __init__(self, a=1.0):
        self.a = 2.0

    def learn_one(self, x, y, t=None):
        return self

    def predict_one(self, x, t=None):
        return 0.0


def test_check_estimator_reports_failures():
    with pytest.raises(AssertionError, match="failed check_estimator"):
        check_estimator(_Broken())


def test_pickle_roundtrip_keeps_params():
    r = _Reg(a=3.0)
    assert pickle.loads(pickle.dumps(r)).a == 3.0
