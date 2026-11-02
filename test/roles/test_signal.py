"""Tests for dense_armor/roles/signal.py."""

import math

import jax.numpy as jnp
import pytest

from dense_armor.roles.signal import Signal, SignalBatch


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


def test_signal_refuses_a_matrix_and_points_to_signal_batch():
    with pytest.raises(ValueError, match="SignalBatch"):
        Signal(
            values=jnp.array([[1.0, 2.0], [3.0, 4.0]]), names=["a", "b"], units=["", ""]
        )


def test_signal_from_dict_marks_none_and_nan():
    s = Signal.from_dict(
        {"b": None, "a": 1.0, "c": float("nan")}, units={"a": "m"}, t=1.0
    )
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
    assert isinstance(s, SignalBatch)
    assert not isinstance(s, dict)
    assert s.array.shape == (2, 1)
    assert s.t == 0.0
    assert s.times == (0.0, 0.0)
    assert len(s) == 2
    assert s.row(1)["x"] == 2.0


def test_signal_stacked_different_time():
    a = Signal(values=[1.0], names=["x"], units=["m"], t=0.0)
    b = Signal(values=[2.0], names=["x"], units=["m"], t=0.1)
    assert a.stacked([b]).t is None


def test_signal_batch_is_not_a_dict_and_validates_its_shape():
    batch = SignalBatch(jnp.ones((3, 2)), ["a", "b"], ["m", "m"], times=[0.0, 0.1, 0.2])
    assert not isinstance(batch, dict)
    assert len(batch) == 3
    assert batch.array.shape == (3, 2)
    assert batch.times == (0.0, 0.1, 0.2)
    assert batch.t is None
    with pytest.raises(ValueError, match="expected"):
        SignalBatch(jnp.ones(2), ["a", "b"], ["m", "m"])
    with pytest.raises(ValueError, match="names"):
        SignalBatch(jnp.ones((3, 2)), ["a"], ["m", "m"])
    with pytest.raises(ValueError, match="units"):
        SignalBatch(jnp.ones((3, 2)), ["a", "b"], ["m"])
    with pytest.raises(ValueError, match="times"):
        SignalBatch(jnp.ones((3, 2)), ["a", "b"], ["m", "m"], times=[0.0])
    with pytest.raises(ValueError, match="missing"):
        SignalBatch(jnp.ones((3, 2)), ["a", "b"], ["m", "m"], missing=[[True]])


def test_signal_batch_row_gives_the_sample_with_its_mask_and_time():
    v = jnp.array([[1.0, float("nan")], [3.0, 4.0]])
    batch = SignalBatch(v, ["a", "b"], ["m", "s"], times=[0.5, 0.6])
    r = batch.row(0)
    assert isinstance(r, Signal)
    assert (r.t, r.n_missing, r.units) == (0.5, 1, ["m", "s"])
    assert batch.row(1)["b"] == 4.0
    assert "n=2" in repr(batch)


def test_signal_stacked_rejects_mismatch():
    a = Signal(values=[1.0], names=["x"], units=["m"])
    b = Signal(values=[2.0], names=["x"], units=["s"])
    with pytest.raises(ValueError, match="share"):
        a.stacked([b])


def test_signal_batch_empty_is_valid_and_t_is_none():
    # A batch with zero rows is a valid empty stream. It must not raise:
    # ``ts`` is empty, so the shared timestamp is None and the length is 0.
    batch = SignalBatch(jnp.zeros((0, 3)), ["a", "b", "c"], ["", "", ""])
    assert len(batch) == 0
    assert batch.t is None
    assert batch.times == ()
    assert "SignalBatch(n=0" in repr(batch)
    assert batch.array.shape == (0, 3)
    assert batch.mask.shape == (0, 3)
    assert batch.names == ["a", "b", "c"]
    assert batch.units == ["", "", ""]
    with pytest.raises(IndexError):
        batch.row(0)


def test_signal_batch_empty_is_not_a_dict():
    # SignalBatch is deliberately not a dict: an estimator that takes one
    # sample cannot take a batch. The empty case must not smuggle a dict
    # interface in either.
    batch = SignalBatch(jnp.zeros((0, 3)), ["a", "b", "c"], ["", "", ""])
    assert not isinstance(batch, dict)
    with pytest.raises(TypeError):
        dict(batch)
