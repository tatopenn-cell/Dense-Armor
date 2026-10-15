"""Tests for OnlineExtraTreesRegressor."""

import doctest
import math

import numpy as np
import pytest

from dense_armor.checks import check_estimator
from dense_armor.roles import DriftDetector
from dense_armor.utility.forest import OnlineExtraTreesRegressor
from dense_armor.utility.forest import oxt as mod
from dense_armor.utility.forest.oxt import (
    _as_dict,
    _LeafModel,
    _Moments,
    _RandomSplitter,
    _RandomTree,
    _split_choice,
)


class _Fire(DriftDetector):
    def __init__(self, at: int = 5) -> None:
        super().__init__()
        self.at = at
        self.k_ = 0

    def update(self, x, t=None):
        self.k_ += 1
        self._drift_detected = self.k_ == self.at
        return self


class _Const:
    def __init__(self, c: float) -> None:
        self.c = c

    def predict(self, v):
        return self.c


def _stream(n: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    for _ in range(n):
        a, b = float(rng.uniform(-1, 1)), float(rng.uniform(-1, 1))
        yield {"a": a, "b": b}, 4.0 * (a > 0.0) + 0.05 * float(rng.normal())


def _cfg(**kw):
    base = {
        "max_features": None,
        "buffer_size": 5,
        "grace_period": 50,
        "delta": 0.05,
        "tau": 0.05,
        "fading": 0.95,
        "learning_rate": 0.01,
        "max_nodes": 1000,
        "max_depth": None,
    }
    base.update(kw)
    return base


def test_doctests():
    res = doctest.testmod(mod)
    assert res.failed == 0 and res.attempted > 0


def test_check_estimator_passes_with_no_check_skipped():
    est = OnlineExtraTreesRegressor(n_models=3, seed=0)
    assert est._unit_test_skips() == set()
    check_estimator(est)


def test_as_dict_and_moments():
    assert _as_dict(None) == {} and _as_dict({"a": 1}) == {"a": 1}
    m = _Moments()
    assert m.std == 0.0
    for v in (1.0, 3.0):
        m.update(v)
    assert m.mean == 2.0 and m.std == pytest.approx(1.0)


def test_random_splitter_fills_its_buffer_then_fixes_the_split_point():
    rng = np.random.default_rng(0)
    sp = _RandomSplitter()
    for x, y in ((1.0, 0.0), (3.0, 10.0)):
        sp.update(x, y, 2, rng)
        assert not sp.ready
    sp.update(2.0, 5.0, 2, rng)
    assert sp.ready and 1.0 <= sp.phi <= 3.0
    assert sp.le.n + sp.gt.n == 3
    sp.update(2.5, 1.0, 2, rng)
    assert sp.le.n + sp.gt.n == 4


def test_the_merit_is_the_variance_reduction_of_a_hand_case():
    sp = _RandomSplitter()
    sp.phi = 5.0
    sp.buf = None
    for x, y in ((1.0, 0.0), (2.0, 0.0), (8.0, 10.0), (9.0, 10.0)):
        sp._add(x, y)
    assert sp.merit() == pytest.approx(25.0)
    assert _RandomSplitter().merit() == 0.0


def test_split_choice_follows_the_hoeffding_ratio_test():
    assert _split_choice([], 50, 0.05, 0.05) is None
    assert _split_choice([0.0], 50, 0.05, 0.05) is None
    assert _split_choice([3.0], 50, 0.05, 0.05) == 0
    assert _split_choice([0.0, 10.0], 50, 0.05, 0.05) == 1
    assert _split_choice([10.0, 9.9], 50, 0.05, 0.05) is None
    assert _split_choice([10.0, 9.9], 600, 0.05, 0.05) == 0
    assert (
        math.sqrt(math.log(20.0) / 1200.0) < 0.05 < math.sqrt(math.log(20.0) / 1100.0)
    )


def test_leaf_model_starts_at_the_mean_and_learns_a_plane():
    rng = np.random.default_rng(0)
    lm = _LeafModel(2)
    assert lm.predict(np.array([1.0, 1.0])) == 0.0
    for _ in range(1500):
        a, b = rng.uniform(-1, 1, 2)
        lm.learn(np.array([a, b]), 3 * a - 2 * b, 0.01, 0.95)
    pts = rng.uniform(-1, 1, (100, 2))
    err = np.mean([abs(lm.predict(p) - (3 * p[0] - 2 * p[1])) for p in pts])
    assert err < 0.3
    flat = _LeafModel(1)
    for i in range(40):
        flat.learn(np.array([float(i % 2)]), 5.0, 0.01, 0.95)
    assert flat.predict(np.array([0.0])) == pytest.approx(5.0)


def test_a_tree_splits_on_a_step_and_respects_its_limits():
    rng = np.random.default_rng(1)
    t = _RandomTree(1, _cfg(), rng)
    for _ in range(600):
        a = float(rng.uniform(-1, 1))
        t.learn(np.array([a]), 4.0 * (a > 0.0))
    assert t.n_nodes >= 3
    assert abs(t.predict(np.array([0.9])) - 4.0) < 1.0
    assert abs(t.predict(np.array([-0.9]))) < 1.0
    one = _RandomTree(1, _cfg(max_nodes=1), np.random.default_rng(1))
    deep = _RandomTree(1, _cfg(max_depth=1), np.random.default_rng(1))
    for _ in range(600):
        a = float(rng.uniform(-1, 1))
        one.learn(np.array([a]), 4.0 * (a > 0.0))
        deep.learn(np.array([a]), 4.0 * (a > 0.0))
    assert one.n_nodes == 1 and deep.n_nodes <= 3


def test_default_subset_size_is_the_square_root_of_the_features():
    t = _RandomTree(9, _cfg(), np.random.default_rng(0))
    assert len(t.root.splitters) == 3
    assert (
        len(
            _RandomTree(
                9, _cfg(max_features=2), np.random.default_rng(0)
            ).root.splitters
        )
        == 2
    )
    assert (
        len(
            _RandomTree(
                9, _cfg(max_features=20), np.random.default_rng(0)
            ).root.splitters
        )
        == 9
    )


def test_learns_a_step_and_predicts_it():
    m = OnlineExtraTreesRegressor(n_models=5, seed=0)
    for x, y in _stream(1500):
        m.learn_one(x, y)
    assert m.predict_one({"a": 0.5, "b": 0.0}) == pytest.approx(4.0, abs=0.5)
    assert m.predict_one({"a": -0.5, "b": 0.0}) == pytest.approx(0.0, abs=0.5)


def test_prediction_is_the_mean_of_the_trees_and_return_std_is_their_spread():
    m = OnlineExtraTreesRegressor(n_models=2, seed=0)
    m.learn_one({"a": 1.0}, 1.0)
    m.members_[0].tree = _Const(1.0)
    m.members_[1].tree = _Const(3.0)
    assert m.predict_one({"a": 1.0}) == 2.0
    assert m.predict_one({"a": 1.0}, return_std=True) == (2.0, 1.0)


def test_predictions_before_learning_and_on_unusable_input():
    m = OnlineExtraTreesRegressor(seed=0)
    assert m.predict_one({"a": 1.0}) == 0.0
    assert m.predict_one({"a": 1.0}, return_std=True) == (0.0, 0.0)
    m.learn_one({"a": 1.0}, 1.0)
    assert m.predict_one({"a": float("nan")}) == 0.0
    assert m.predict_one(None) == 0.0


def test_subbagging_chance():
    counts = {}
    for p in (1.0, 0.5):
        m = OnlineExtraTreesRegressor(
            n_models=1,
            p=p,
            drift_detector=_Fire(10**9),
            warning_detector=_Fire(10**9),
            seed=0,
        )
        for x, y in _stream(800):
            m.learn_one(x, y)
        counts[p] = m.members_[0].tree.n_learned
    assert counts[1.0] == 800
    assert 300 <= counts[0.5] <= 500


def test_warning_starts_a_background_tree_that_trains_and_does_not_vote_then_drift_promotes_it():
    m = OnlineExtraTreesRegressor(
        n_models=1, p=1.0, drift_detector=_Fire(8), warning_detector=_Fire(5), seed=0
    )
    s = _stream(8)
    for _ in range(4):
        m.learn_one(*next(s))
    assert m.members_[0].background is None
    m.learn_one(*next(s))
    bg = m.members_[0].background
    assert bg is not None and m.n_warnings_ == 1
    m.learn_one(*next(s))
    assert bg.n_learned >= 1
    v = np.array([0.3, 0.1])
    assert m.predict_one({"a": 0.3, "b": 0.1}) == m.members_[0].tree.predict(v)
    m.learn_one(*next(s))
    m.learn_one(*next(s))
    assert (
        m.members_[0].tree is bg
        and m.members_[0].background is None
        and m.n_drifts_ == 1
    )


def test_a_drift_without_a_warning_starts_an_empty_tree():
    m = OnlineExtraTreesRegressor(
        n_models=1, p=1.0, drift_detector=_Fire(3), warning_detector=_Fire(99), seed=0
    )
    s = _stream(3)
    for _ in range(2):
        m.learn_one(*next(s))
    old = m.members_[0].tree
    m.learn_one(*next(s))
    assert m.members_[0].tree is not old and m.members_[0].tree.n_learned == 1


def test_detector_templates_are_cloned_for_every_tree():
    tpl = _Fire(10**9)
    m = OnlineExtraTreesRegressor(
        n_models=3, drift_detector=tpl, warning_detector=tpl, seed=0
    )
    m.learn_one({"a": 1.0}, 1.0)
    dets = [x.drift for x in m.members_] + [x.warn for x in m.members_]
    assert len({id(d) for d in dets}) == 6 and all(d is not tpl for d in dets)
    d0 = OnlineExtraTreesRegressor(n_models=1, seed=0)
    d0.learn_one({"a": 1.0}, 1.0)
    assert d0.members_[0].drift.delta == 0.001 and d0.members_[0].warn.delta == 0.01


def test_nan_missing_and_none_are_skipped_and_counted():
    m = OnlineExtraTreesRegressor(seed=0, feature_keys=["a", "b"])
    for x, y in _stream(30):
        m.learn_one(x, y)
    n0, seen = m.n_missing, m.n_seen_
    for x, y in (
        ({"a": float("nan"), "b": 0.0}, 1.0),
        ({"a": 1.0}, 1.0),
        ({"a": "word", "b": 0.0}, 1.0),
        ({}, 1.0),
        (None, 1.0),
        ({"a": 0.1, "b": 0.0}, None),
        ({"a": 0.1, "b": 0.0}, float("inf")),
        ({"a": 0.1, "b": 0.0}, "word"),
    ):
        m.learn_one(x, y)
    assert m.n_missing == n0 + 8 and m.n_seen_ == seen


def test_constructor_arguments_not_overwritten_and_clone_is_fresh():
    m = OnlineExtraTreesRegressor(n_models=4, p=0.7, max_depth=3, seed=3)
    before = m.get_params()
    for x, y in _stream(300):
        m.learn_one(x, y)
    assert m.get_params() == before
    c = m.clone()
    assert c.get_params() == before and c.members_ == [] and c.n_seen_ == 0


def test_same_seed_same_forest():
    outs = []
    for _ in range(2):
        m = OnlineExtraTreesRegressor(n_models=3, seed=7)
        for x, y in _stream(400):
            m.learn_one(x, y)
        outs.append(m.predict_one({"a": 0.3, "b": -0.2}))
    assert outs[0] == outs[1]


def test_bad_parameters_raise():
    for kw in (
        {"n_models": 0},
        {"buffer_size": 0},
        {"grace_period": 0},
        {"max_nodes": 0},
        {"max_features": 0},
        {"p": 0.0},
        {"p": 1.5},
        {"delta": 0.0},
        {"fading": 1.0},
        {"tau": -0.1},
        {"learning_rate": 0.0},
        {"max_depth": 0},
    ):
        with pytest.raises(ValueError):
            OnlineExtraTreesRegressor(**kw)
