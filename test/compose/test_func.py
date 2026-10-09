"""Tests for FuncTransformer."""

from dense_armor.roles import Signal
from dense_armor.utility.compose import FuncTransformer, Select


def test_scale_selected_channel():
    f = FuncTransformer(lambda v: v * 2.0, keys=("a",))
    out = f.transform_one({"a": 1.5, "b": 3.0})
    assert out == {"a": 3.0, "b": 3.0}


def test_whole_dict():
    f = FuncTransformer(lambda d: {"sum": sum(d.values())}, whole=True)
    out = f.transform_one({"a": 1.0, "b": 2.0})
    assert out == {"sum": 3.0}


def test_signal_keeps_t_and_units():
    s = Signal(
        values=[1.0, 2.0],
        names=["a", "b"],
        units=["rad", "rad"],
        t=0.5,
    )
    out = FuncTransformer(lambda v: v + 1.0).transform_one(s)
    assert isinstance(out, Signal)
    assert out.t == 0.5
    assert out.units == ["rad", "rad"]
    assert out["a"] == 2.0
    assert out["b"] == 3.0


def test_none_result_skips_channel():
    f = FuncTransformer(lambda v: None if v < 0 else v, keys=("a",))
    out = f.transform_one({"a": -1.0, "b": 2.0})
    assert out == {"a": -1.0, "b": 2.0}


def test_pipe_with_select():
    f = FuncTransformer(lambda v: v * 10.0, keys=("a",))
    pipe = Select(keys=("a",)) | f
    out = pipe.transform_one({"a": 1.0, "b": 2.0})
    assert out == {"a": 10.0}


def test_check_estimator():
    from dense_armor.checks import check_estimator

    check_estimator(FuncTransformer(lambda v: v))
