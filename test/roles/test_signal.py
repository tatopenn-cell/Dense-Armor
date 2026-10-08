"""Tests for dense_armor/roles/signal.py."""
import math

import jax.numpy as jnp
import pytest

from dense_armor.roles.signal import Signal


def test_signal_vector_as_dict():
    s = Signal(values=[0.1, 0.2], names=["q0", "q1"], units=["rad", "rad"], t=0.5)
    assert s["q0"] == pytest.approx(0.1)
    assert s.array.shape == (2,)
    assert s.t == 0.5
    assert s.n_missing == 0
    assert s.to_dict() == {"q0": pytest.approx(0.1), "q1": pytest.approx(0.2)}


def test_signal_nan_is_missing():
    s = Signal(values=[1.0, float("nan")], names=["a", "b"], units=["", ""])
    assert s.n_missing == 1
    assert "n_missing=1" in repr(s)


def test_signal_repr_without_missing():
    s = Signal(values=[1.0], names=["a"], units=["m"])
    assert repr(s) == "Signal(names=['a'], units=['m'], t=None)"


def test_signal_rejects_wrong_names():
    with pytest.raises(ValueError, match="names"):
        Signal(values=[1.0, 2.0], names=["a"], units=["", ""])


def test_signal_rejects_wrong_units():
    with pytest.raises(ValueError, match="units"):
        Signal(values=[1.0, 2.0], names=["a", "b"], units=[""])


def test_signal_rejects_wrong_mask_shape():
    with pytest.raises(ValueError, match="missing"):
        Signal(values=[1.0, 2.0], names=["a", "b"], units=["", ""], missing=[True])


def test_signal_batch_values_are_tuples():
    s = Signal(values=jnp.array([[1.0, 2.0], [3.0, 4.0]]), names=["a", "b"], units=["", ""])
    assert s["a"] == (1.0, 3.0)
    assert s["b"] == (2.0, 4.0)


def test_signal_from_dict_marks_none_and_nan():
    s = Signal.from_dict({"b": None, "a": 1.0, "c": float("nan")}, units={"a": "m"}, t=1.0)
    assert s.names == ["a", "b", "c"]
    assert s.units == ["m", "", ""]
    assert s.n_missing == 2
    assert math.isnan(s["b"])


def test_signal_from_dict_explicit_order():
    s = Signal.from_dict({"a": 1.0, "b": 2.0}, names=["b", "a"])
    assert s.names == ["b", "a"]
    assert float(s.array[0]) == 2.0


def test_signal_stacked_same_time():
    a = Signal(values=[1.0], names=["x"], units=["m"], t=0.0)
    b = Signal(values=[2.0], names=["x"], units=["m"], t=0.0)
    s = a.stacked([b])
    assert s.array.shape == (2, 1)
    assert s.t == 0.0
    assert s["x"] == (1.0, 2.0)


def test_signal_stacked_different_time():
    a = Signal(values=[1.0], names=["x"], units=["m"], t=0.0)
    b = Signal(values=[2.0], names=["x"], units=["m"], t=0.1)
    assert a.stacked([b]).t is None


def test_signal_stacked_rejects_mismatch():
    a = Signal(values=[1.0], names=["x"], units=["m"])
    b = Signal(values=[2.0], names=["x"], units=["s"])
    with pytest.raises(ValueError, match="share"):
        a.stacked([b])
