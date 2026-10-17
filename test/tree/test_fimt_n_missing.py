"""Tests for the n_missing property on FIMTDDRegressor.

The property is what item D of task Z3 asks: ``n_missing`` reads the same
counter as ``n_missing_``. The two are equal on a fresh estimator and
after a missing sample. The rest of the class (learn on a finite sample
and the tree) is broken on master (``root_`` not initialised) and is
reported in the delivery, not fixed here.
"""

import pytest

from dense_armor.utility.tree.fimt import FIMTDDRegressor


def test_n_missing_is_zero_on_a_fresh_estimator():
    m = FIMTDDRegressor()
    assert m.n_missing == 0
    assert m.n_missing_ == 0


def test_missing_sample_increments_both():
    m = FIMTDDRegressor()
    m.learn_one({"a": float("nan")}, 1.0)
    assert m.n_missing_ == 1
    assert m.n_missing == 1
    m.learn_one({"a": float("inf")}, 1.0)
    assert m.n_missing_ == 2
    assert m.n_missing == 2


def test_property_is_read_only():
    m = FIMTDDRegressor()
    with pytest.raises(AttributeError):
        m.n_missing = 5
