"""Tests for CausalGraph."""

import numpy as np
import pandas as pd
import pytest

from dense_armor.core.graph import CausalGraph


def _data(n=2000, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal(n)
    Z = X + rng.standard_normal(n) * 0.5
    Y = 2.0 * Z + rng.standard_normal(n) * 0.5
    return pd.DataFrame({"X": X, "Z": Z, "Y": Y})


def test_cycle_refused():
    with pytest.raises(ValueError):
        CausalGraph(nodes=["X", "Y"], edges=[("X", "Y"), ("Y", "X")])


def test_unknown_node_refused():
    with pytest.raises(ValueError):
        CausalGraph(nodes=["X"], edges=[("X", "Z")])


def test_parents():
    cg = CausalGraph(nodes=["X", "Z", "Y"], edges=[("X", "Z"), ("Z", "Y")])
    assert cg.parents("Z") == ["X"]
    assert cg.parents("X") == []
    assert cg.parents("Y") == ["Z"]
    assert cg.is_valid()


def test_estimate_effect():
    pytest.importorskip("dowhy")
    cg = CausalGraph(nodes=["X", "Z", "Y"], edges=[("X", "Z"), ("Z", "Y")])
    df = _data()
    eff = cg.estimate_effect("Z", 1.0, df, outcome="Y")
    assert abs(eff - 2.0) < 0.05


def test_confounder():
    """Un vero confondente: C -> Z, C -> Y, Z -> Y."""
    cg = CausalGraph(nodes=["C", "Z", "Y"], edges=[("C", "Z"), ("C", "Y"), ("Z", "Y")])
    df = _data_with_confounder()
    eff = cg.estimate_effect("Z", 1.0, df, outcome="Y")
    assert abs(eff - 2.0) < 0.1


def _data_with_confounder(n=3000, seed=0):
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(seed)
    C = rng.standard_normal(n)
    Z = C + rng.standard_normal(n) * 0.5
    Y = 2.0 * Z + 3.0 * C + rng.standard_normal(n) * 0.5
    return pd.DataFrame({"C": C, "Z": Z, "Y": Y})
