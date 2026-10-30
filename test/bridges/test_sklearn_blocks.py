"""Block path of the scikit-learn bridge.

The bridge hands the columns of the array to a model that declares
``learn_many_columns = True`` (the ``StandardScaler`` of Dense-Armor), and
keeps the loop of ``learn_one`` for every other model.
"""

import numpy as np
import pytest

pytest.importorskip("sklearn")

from dense_armor.bridges.sklearn import SklearnTransformer  # noqa: E402
from dense_armor.utility.preprocessing.scale import (  # noqa: E402
    EWScaler,
    StandardScaler,
)

TOL = 1e-12


class _NoLoop(StandardScaler):
    def learn_one(self, x, y=None, t=None):
        raise AssertionError("the loop of learn_one was used")


class _LoopOnly(StandardScaler):
    learn_many_columns = False


def _table(n=4000, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal([1e3, 5.0, -2.0], [1.0, 2.0, 0.5], size=(n, 3))


def test_bridge_uses_the_block_path():
    SklearnTransformer(_NoLoop).fit(_table())


def test_bridge_block_equals_loop():
    X = _table()
    blk = SklearnTransformer(StandardScaler).fit(X).transform(X)
    loop = SklearnTransformer(_LoopOnly).fit(X).transform(X)
    assert np.max(np.abs(blk - loop)) <= 1e-9


def test_bridge_partial_fit_in_blocks_equals_one_fit():
    X = _table()
    one = SklearnTransformer(StandardScaler).fit(X)
    many = SklearnTransformer(StandardScaler)
    for i in range(0, 4000, 500):
        many.partial_fit(X[i : i + 500])
    assert np.max(np.abs(one.transform(X) - many.transform(X))) <= 1e-9


def test_bridge_keeps_the_loop_for_a_model_without_a_block_path():
    assert not getattr(EWScaler(), "learn_many_columns", False)
    X = _table(500)
    out = SklearnTransformer(EWScaler).fit(X).transform(X)
    assert out.shape == X.shape
    assert np.all(np.isfinite(out))


def test_overflow_input_through_the_bridge_keeps_the_mean_finite():
    v = np.finfo(np.float64).max / 1e5 / 1e2
    sc = SklearnTransformer(StandardScaler)
    for _ in range(20):
        sc.partial_fit(np.full((50000, 1), v))
    r = sc.model_.stats_[0][0]
    assert r.count == 1_000_000
    assert np.isfinite(r.mean)
    assert r.mean == pytest.approx(v, rel=TOL)
    assert r.var == 0.0


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_large_offset_input_through_the_bridge_matches_numpy():
    x = np.random.default_rng(0).normal(1e300, 1e290, size=(20000, 1))
    r = SklearnTransformer(StandardScaler).fit(x).model_.stats_[0][0]
    assert np.isfinite(r.mean)
    assert r.mean == pytest.approx(float(x.mean()), rel=TOL)


def test_full_reproducer_through_the_bridge():
    v = np.finfo(np.float64).max / 1e5 / 1e2
    sc = SklearnTransformer(StandardScaler)
    batch = np.full((500000, 1), v)
    for _ in range(100):
        sc.partial_fit(batch)
    r = sc.model_.stats_[0][0]
    assert r.count == 50_000_000
    assert r.mean == pytest.approx(v, rel=TOL)
