"""Tests for GraphSignal."""

import numpy as np
import pytest

from dense_armor.core.graph import Graph, GraphSignal
from dense_armor.roles import Signal


def _path(n):
    return Graph(n, [(i, i + 1, 1.0) for i in range(n - 1)])


def test_roundtrip():
    g = _path(5)
    s = Signal(
        [1.0, 2.0, 3.0, 4.0, 5.0], names=["n0", "n1", "n2", "n3", "n4"], units=["V"] * 5
    )
    gs = GraphSignal.from_signal(s, g)
    back = gs.to_signal()
    assert back == s
    assert back.names == s.names
    assert back.units == s.units


def test_values_order():
    g = _path(5)
    s = Signal(
        [1.0, 2.0, 3.0, 4.0, 5.0], names=["a", "b", "c", "d", "e"], units=["V"] * 5
    )
    gs = GraphSignal.from_signal(s, g)
    assert np.allclose(gs.values, [1.0, 2.0, 3.0, 4.0, 5.0])


def test_wrong_number_of_channels():
    g = _path(5)
    s = Signal([1.0, 2.0, 3.0], names=["a", "b", "c"], units=["V"] * 3)
    with pytest.raises(ValueError):
        GraphSignal.from_signal(s, g)
