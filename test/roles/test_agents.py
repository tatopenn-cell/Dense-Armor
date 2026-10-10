"""Tests for dense_armor/roles/agents.py."""
import doctest
import json

import pytest

import dense_armor.roles.agents as agents_mod
from dense_armor.roles import Regressor
from dense_armor.roles.agents import call_json, schema
from dense_armor.roles.physics import UnitSpec
from dense_armor.utility.stats.moments import RunningMoments


class _TypedReg(Regressor):
    units = UnitSpec(inputs={"q": "rad"}, outputs={"y": "rad"})
    budget_s = 1e-4
    memory_class = "O(1)"

    def __init__(self) -> None:
        self._last = 0.0
        self.last_x = None

    def learn_one(self, x, y, t=None):
        self._last = float(y)
        self.last_x = x
        return self

    def predict_one(self, x, t=None, return_std=False):
        return self._last


def test_schema_is_json_serialisable():
    d = schema(RunningMoments())
    json.dumps(d)


def test_schema_carries_budget_and_memory_class():
    d = schema(_TypedReg())
    assert d["budget_s"] == 1e-4
    assert d["memory_class"] == "O(1)"


def test_schema_carries_units_when_declared():
    d = schema(_TypedReg())
    assert d["units"]["inputs"] == {"q": "rad"}
    assert d["units"]["outputs"] == {"y": "rad"}


def test_schema_has_empty_units_when_undeclared():
    d = schema(RunningMoments())
    assert d["units"] == {}


def test_schema_has_methods_list():
    d = schema(RunningMoments())
    assert "learn_one" in d["methods"]
    assert "transform_one" in d["methods"]


def test_call_json_learn_one_with_dict():
    est = RunningMoments()
    out = call_json(est, {"method": "learn_one", "x": {"a": 1.0}, "y": 0.0})
    parsed = json.loads(out)
    assert parsed["ok"] is True
    assert est.count == 1


def test_call_json_predict_one_on_typed_regressor():
    est = _TypedReg()
    est.learn_one({"q": 1.0}, 2.0)
    out = call_json(est, {"method": "predict_one", "x": {"q": 1.0}})
    parsed = json.loads(out)
    assert parsed["ok"] is True
    assert parsed["result"] == 2.0


def test_call_json_with_signal_block():
    est = _TypedReg()
    payload = {
        "method": "learn_one",
        "x": {
            "__signal__": {
                "values": [0.1, 0.2],
                "names": ["q0", "q1"],
                "units": ["rad", "rad"],
                "t": 0.0,
            }
        },
        "y": 0.5,
    }
    out = call_json(est, payload)
    assert json.loads(out)["ok"] is True
    assert est._last == 0.5
    x = est.last_x
    assert x is not None
    assert x["q0"] == pytest.approx(0.1, abs=1e-6)
    assert x["q1"] == pytest.approx(0.2, abs=1e-6)


def test_call_json_unknown_method():
    est = RunningMoments()
    out = call_json(est, {"method": "fly", "x": {}})
    parsed = json.loads(out)
    assert parsed["ok"] is False
    assert "fly" in parsed["error"]


def test_call_json_missing_method():
    out = call_json(RunningMoments(), {"x": {}})
    parsed = json.loads(out)
    assert parsed["ok"] is False
    assert "method" in parsed["error"].lower()


def test_call_json_invalid_json_string():
    out = call_json(RunningMoments(), "{not json")
    parsed = json.loads(out)
    assert parsed["ok"] is False
    assert "invalid JSON" in parsed["error"]


def test_call_json_with_timestamp():
    est = RunningMoments()
    for i in range(10):
        call_json(est, {
            "method": "learn_one",
            "x": {"a": float(i)},
            "y": 0.0,
            "t": i * 0.01,
        })
    assert est.count == 10
    assert abs(est.dt - 0.01) < 1e-9


def test_call_json_result_is_jsonable_for_transform():
    est = RunningMoments()
    call_json(est, {"method": "learn_one", "x": {"a": 1.0}, "y": 0.0})
    out = call_json(est, {"method": "transform_one", "x": {"a": 1.0}})
    parsed = json.loads(out)
    assert parsed["ok"] is True
    assert isinstance(parsed["result"], dict)
    assert "mean" in parsed["result"]


def test_doctests():
    assert doctest.testmod(agents_mod).failed == 0

