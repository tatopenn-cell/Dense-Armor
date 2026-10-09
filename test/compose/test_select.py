"""Tests for Select and Discard."""

import pytest

from dense_armor.roles import Signal
from dense_armor.utility.compose import Discard, Select


def _sig():
    return Signal(
        values=[1.0, 2.0, 3.0],
        names=["a", "b", "c"],
        units=["rad", "rad", "N*m"],
        t=0.5,
    )


def test_select_keeps_order_and_metadata():
    out = Select(keys=("c", "a")).transform_one(_sig())
    assert isinstance(out, Signal)
    assert out.names == ["c", "a"]
    assert out.units == ["N*m", "rad"]
    assert out.t == 0.5
    assert out["a"] == 1.0
    assert out["c"] == 3.0


def test_select_ignores_missing_keys():
    out = Select(keys=("a", "z")).transform_one(_sig())
    assert out.names == ["a"]


def test_discard_keeps_remaining():
    out = Discard(keys=("b",)).transform_one(_sig())
    assert out.names == ["a", "c"]
    assert out["c"] == 3.0


def test_select_on_plain_dict():
    out = Select(keys=("x",)).transform_one({"x": 1, "y": 2})
    assert out == {"x": 1}


def test_discard_on_plain_dict():
    out = Discard(keys=("y",)).transform_one({"x": 1, "y": 2})
    assert out == {"x": 1}


def test_select_preserves_missing_mask():
    s = Signal(
        values=[1.0, float("nan")],
        names=["a", "b"],
        units=["", ""],
        t=0.0,
    )
    out = Select(keys=("b",)).transform_one(s)
    assert out.n_missing == 1


@pytest.mark.parametrize("cls", [Select, Discard])
def test_check_estimator(cls):
    from dense_armor.checks import check_estimator

    check_estimator(cls(keys=("a",)))
