"""Tests for dense_armor/utility/evaluate/prequential.py."""
import doctest

import jax.numpy as jnp
import pytest

import dense_armor.utility.evaluate.prequential as preq_mod
from dense_armor.roles import Signal
from dense_armor.utility.evaluate import progressive_val_score
from dense_armor.utility.metrics import Accuracy, MeanSquaredError


class _LastValue:
    """Predict the last y seen."""

    def __init__(self) -> None:
        self._last = 0

    def learn_one(self, x, y, t=None):
        self._last = y
        return self

    def predict_one(self, x, t=None):
        return self._last


class _Constant:
    """Predict a fixed value."""

    def __init__(self, c=0):
        self.c = c

    def learn_one(self, x, y, t=None):
        return self

    def predict_one(self, x, t=None):
        return self.c


def _constant_stream(n: int = 20, value: int = 1):
    return [({}, value)] * n


def _alternating_stream(n: int = 20):
    return [({}, i % 2) for i in range(n)]


def _timestamped_stream(n: int = 20):
    return [
        (Signal(values=jnp.array([float(i)]), names=["a"],
                units=[""], t=float(i)), i % 2)
        for i in range(n)
    ]


def test_constant_stream_accuracy_high():
    m = progressive_val_score(
        _constant_stream(20, 1), _LastValue(), Accuracy()
    )
    assert m.n == 20
    assert m.get() == pytest.approx(19.0 / 20.0)


def test_constant_predictor_on_alternating():
    m = progressive_val_score(
        _alternating_stream(20), _Constant(0), Accuracy()
    )
    assert m.get() == pytest.approx(0.5)


def test_immediate_vs_delayed_labels_same_count():
    a = progressive_val_score(
        _constant_stream(20), _LastValue(), Accuracy()
    )
    b = progressive_val_score(
        _constant_stream(20), _LastValue(), Accuracy(), delay=3
    )
    assert a.n == b.n


def test_delay_in_seconds_uses_timestamps():
    m = progressive_val_score(
        _timestamped_stream(20), _LastValue(), Accuracy(), delay=2.0
    )
    assert m.n > 0


def test_returns_trace_with_every():
    m, trace = progressive_val_score(
        _constant_stream(20), _LastValue(), Accuracy(), every=5
    )
    assert len(trace) == 4
    assert m.n == 20


def test_every_must_be_positive():
    with pytest.raises(ValueError):
        progressive_val_score(
            _constant_stream(5), _LastValue(), Accuracy(), every=0
        )


def test_regression_metric_with_constant_stream():
    stream = [({}, 2.0)] * 10
    m = progressive_val_score(stream, _LastValue(), MeanSquaredError())
    assert m.n == 10
    assert m.get() > 0.0


def test_doctests():
    assert doctest.testmod(preq_mod).failed == 0
