"""``StandardScaler.learn_many``: a block of samples through ``RunningMoments.learn_many``.

The block equals the loop of ``learn_one`` within a tolerance that the first
test states on both sides: equal inside it, different outside it.
"""

import numpy as np
import pytest

from dense_armor.utility.preprocessing.scale import StandardScaler

TOL = 1e-12


def _table(n=4000, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal([1e3, 5.0, -2.0], [1.0, 2.0, 0.5], size=(n, 3))


def test_block_equals_loop():
    X = _table()
    a, b = StandardScaler(), StandardScaler()
    for r in X:
        a.learn_one({"f0": r[0], "f1": r[1], "f2": r[2]})
    b.learn_many({"f0": X[:, 0], "f1": X[:, 1], "f2": X[:, 2]})
    for k in ("f0", "f1", "f2"):
        sa, sb = a.stats_[k][0], b.stats_[k][0]
        assert sa.count == sb.count
        assert abs(sa.mean - sb.mean) / abs(sa.mean) <= TOL
        assert abs(sa.var - sb.var) / sa.var <= TOL


def test_block_detects_a_different_input():
    X = _table()
    a, b = StandardScaler(), StandardScaler()
    a.learn_many({"f0": X[:, 0]})
    b.learn_many({"f0": X[:, 0] * 1.01})
    va, vb = a.stats_["f0"][0].var, b.stats_["f0"][0].var
    assert abs(va - vb) / va > 1e-3


def test_block_counts_nan_as_missing():
    col = np.array([1.0, 2.0, np.nan, 3.0, np.inf, 4.0])
    r = StandardScaler().learn_many({"x": col}).stats_["x"][0]
    assert r.count == 4
    assert r.n_missing == 2
    assert r.mean == pytest.approx(2.5)


def test_block_with_a_vector_feature():
    q = np.array([[0.0, 1.0], [1.0, 2.0], [2.0, 3.0]])
    s = StandardScaler().learn_many({"q": q})
    assert [round(v, 4) for v in s.transform_one({"q": [2.0, 3.0]})["q"]] == [1.0, 1.0]


def test_list_of_dicts_uses_the_loop():
    s = StandardScaler().learn_many([{"x": 1.0}, {"x": 2.0}, {"x": 3.0}])
    assert s.stats_["x"][0].count == 3


@pytest.mark.parametrize("block", [{}, {"x": []}])
def test_an_empty_block_changes_nothing(block):
    s = StandardScaler()
    assert s.learn_many(block) is s
    assert s.stats_ is None


def test_a_missing_or_reshaped_column_is_skipped():
    s = StandardScaler().learn_many({"x": [1.0, 2.0, 3.0], "y": [4.0, 5.0, 6.0]})
    s.learn_many({"x": [4.0, 5.0]})
    s.learn_many({"x": [6.0], "y": [[1.0, 2.0]]})
    assert s.stats_["x"][0].count == 6
    assert s.stats_["y"][0].count == 3
