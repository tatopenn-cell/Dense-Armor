"""Tests for dense_armor/utility/metrics/rolling.py."""
import doctest

import jax.numpy as jnp
import pytest

import dense_armor.utility.metrics.rolling as rolling_mod
from dense_armor.roles import Classifier, Regressor, Signal
from dense_armor.utility.metrics import (
    Accuracy,
    MeanAbsoluteError,
    MeanSquaredError,
)
from dense_armor.utility.metrics.rolling import Rolling


def test_rolling_accuracy_window():
    m = Rolling(Accuracy(), window=5)
    for a, b in [(1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (0, 1)]:
        m.update(a, b)
    assert m.get() == pytest.approx(0.8)


def test_rolling_matches_manual_window():
    stream = [(1, 1), (0, 1), (1, 0), (1, 1), (0, 0), (1, 1)]
    m = Rolling(Accuracy(), window=3)
    for a, b in stream:
        m.update(a, b)
    manual = Accuracy()
    for a, b in stream[-3:]:
        manual.update(a, b)
    assert m.get() == pytest.approx(manual.get())


def test_rolling_mse_window():
    stream = [(1.0, 1.0), (2.0, 2.0), (3.0, 3.0), (4.0, 0.0)]
    m = Rolling(MeanSquaredError(), window=2)
    for a, b in stream:
        m.update(a, b)
    manual = MeanSquaredError()
    for a, b in stream[-2:]:
        manual.update(a, b)
    assert m.get() == pytest.approx(manual.get())


def test_rolling_revert():
    m = Rolling(Accuracy(), window=5)
    for a, b in [(1, 1), (1, 1), (0, 1)]:
        m.update(a, b)
    before = m.get()
    m.update(0, 1)
    m.revert(0, 1)
    assert m.get() == pytest.approx(before)


def test_rolling_window_s_uses_timestamps():
    m = Rolling(Accuracy(), window_s=0.05)
    for i in range(20):
        s = Signal(
            values=jnp.array([0.0]),
            names=["x"],
            units=[""],
            t=i * 0.01,
        )
        m.update(1, 1, t=s.t)
    assert m.count == 5


def test_rolling_requires_window_arg():
    with pytest.raises(ValueError):
        Rolling(Accuracy())
    with pytest.raises(ValueError):
        Rolling(Accuracy(), window=5, window_s=1.0)


def test_rolling_rejects_bad_window():
    with pytest.raises(ValueError):
        Rolling(Accuracy(), window=0)
    with pytest.raises(ValueError):
        Rolling(Accuracy(), window_s=0.0)


def test_rolling_works_with_role():
    m = Rolling(MeanAbsoluteError(), window=3)
    assert m.works_with(Regressor())
    assert not m.works_with(Classifier())


def test_rolling_merge():
    left = Rolling(Accuracy(), window=5)
    right = Rolling(Accuracy(), window=5)
    for a, b in [(1, 1), (1, 1)]:
        left.update(a, b)
    for a, b in [(0, 1), (1, 1)]:
        right.update(a, b)
    merged = left.merge(right)
    assert merged.n == 4
    assert merged.get() == pytest.approx(0.75)


def test_rolling_merge_rejects_wrong_type():
    with pytest.raises(TypeError):
        Rolling(Accuracy(), window=5).merge("x")


def test_doctests():
    assert doctest.testmod(rolling_mod).failed == 0
