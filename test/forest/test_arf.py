"""Tests for AdaptiveRandomForestClassifier and SubspaceHoeffdingTree."""

import doctest
import math

import numpy as np
import pytest

from dense_armor.checks import check_estimator
from dense_armor.roles import DriftDetector
from dense_armor.utility.datasets import DriftStream
from dense_armor.utility.drift.adwin import ADWIN
from dense_armor.utility.forest import arf as arf_mod
from dense_armor.utility.forest.arf import (
    AdaptiveRandomForestClassifier,
    SubspaceHoeffdingTree,
    _Member,
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


class _Stub:
    def __init__(self, proba: dict) -> None:
        self.proba = proba

    def predict_proba_one(self, x, t=None):
        return dict(self.proba)


def _stream(n: int, flip_at: int | None = None, seed: int = 0):
    rng = np.random.default_rng(seed)
    for i in range(n):
        x = float(rng.normal(0.0 if rng.random() < 0.5 else 5.0, 0.4))
        y = 0 if x < 2.5 else 1
        if flip_at is not None and i >= flip_at:
            y = 1 - y
        yield {"x": x}, y


def test_doctests():
    res = doctest.testmod(arf_mod)
    assert res.failed == 0 and res.attempted > 0


def test_check_estimator_passes():
    check_estimator(AdaptiveRandomForestClassifier(n_models=3, seed=0))


def test_predict_before_learning_is_empty():
    arf = AdaptiveRandomForestClassifier(n_models=2, seed=0)
    assert arf.predict_one({"x": 1.0}) is None
    assert arf.predict_proba_one({"x": 1.0}) == {}


def test_bad_parameters_raise():
    for kw in (
        {"n_models": 0},
        {"max_features": 0},
        {"lambda_": 0.0},
        {"grace_period": 0},
    ):
        with pytest.raises(ValueError):
            AdaptiveRandomForestClassifier(**kw)
    with pytest.raises(ValueError):
        SubspaceHoeffdingTree(max_features=0)


def test_constructor_arguments_not_overwritten_by_learning():
    det = ADWIN(delta=0.01)
    arf = AdaptiveRandomForestClassifier(
        n_models=3, max_features=2, lambda_=4.0, drift_detector=det, seed=5
    )
    before = arf.get_params()
    for x, y in _stream(300):
        arf.learn_one(x, y)
    assert arf.get_params() == before
    assert arf.drift_detector is det
    c = arf.clone()
    assert c.get_params() == before
    assert c.members_ == [] and c.n_seen_ == 0


def test_nan_and_missing_are_skipped_and_counted():
    arf = AdaptiveRandomForestClassifier(n_models=3, seed=0)
    for x, y in _stream(200):
        arf.learn_one(x, y)
    n0 = arf.n_missing
    arf.learn_one({"x": float("nan")}, 0)
    arf.learn_one({"x": float("inf")}, 0)
    arf.learn_one({"x": None}, 0)
    arf.learn_one({"x": "a"}, 0)
    arf.learn_one({}, 0)
    arf.learn_one({"x": 1.0}, None)
    assert arf.n_missing == n0 + 6
    seen = arf.n_seen_
    for x, y in _stream(100, seed=1):
        arf.learn_one(x, y)
    assert arf.n_seen_ == seen + 100
    p = arf.predict_proba_one({"x": 5.0})
    assert all(math.isfinite(v) for v in p.values())
    assert arf.predict_one({"x": 5.0}) == 1


def test_poisson_weight_mean_is_lambda():
    arf = AdaptiveRandomForestClassifier(n_models=1, lambda_=6.0, seed=0)
    n = 400
    for x, y in _stream(n):
        arf.learn_one(x, y)
    assert arf.n_drifts_ == 0
    seen = arf.members_[0].tree.n_seen_
    assert abs(seen - 6.0 * n) < 5.0 * math.sqrt(6.0 * n)


def test_default_subspace_size_is_sqrt_m_plus_one():
    t = SubspaceHoeffdingTree()
    assert t._subset_size(9) == 4
    assert t._subset_size(1) == 1
    assert SubspaceHoeffdingTree(max_features=2)._subset_size(9) == 2
    assert SubspaceHoeffdingTree(max_features=20)._subset_size(9) == 9


def test_every_split_uses_a_feature_of_the_subset_of_its_node():
    rng = np.random.default_rng(0)
    tree = SubspaceHoeffdingTree(grace_period=30, max_features=1, seed=0)
    for _ in range(1500):
        v = rng.normal(size=4)
        y = int(v[0] + v[1] > 0)
        tree.learn_one({f"f{i}": float(v[i]) for i in range(4)}, y)
    nodes = [tree.root_]
    internal = 0
    while nodes:
        node = nodes.pop()
        if node.left_ is None:
            continue
        internal += 1
        assert node.subset_ is not None and len(node.subset_) == 1
        assert node.split_feature_ in node.subset_
        nodes += [node.left_, node.right_]
    assert internal >= 3


def test_subspace_tree_restores_node_statistics_after_a_split_attempt():
    rng = np.random.default_rng(1)
    tree = SubspaceHoeffdingTree(grace_period=20, max_features=1, seed=1)
    for _ in range(100):
        v = rng.normal(size=3)
        tree.learn_one(
            {"a": float(v[0]), "b": float(v[1]), "c": float(v[2])}, int(v[0] > 0)
        )
    leaves = [tree.root_]
    while leaves:
        node = leaves.pop()
        if node.left_ is None:
            if node.n_ > 0:
                assert set(node.feature_stats_) == {"a", "b", "c"}
        else:
            leaves += [node.left_, node.right_]


def test_detectors_are_cloned_per_tree():
    template = ADWIN(delta=0.01)
    arf = AdaptiveRandomForestClassifier(
        n_models=3, drift_detector=template, warning_detector=template, seed=0
    )
    for x, y in _stream(5):
        arf.learn_one(x, y)
    dets = [m.drift for m in arf.members_] + [m.warn for m in arf.members_]
    assert len({id(d) for d in dets}) == 6
    assert all(d is not template for d in dets)
    assert all(d.delta == 0.01 for d in dets)


def test_default_detectors_are_the_fast_setting_of_the_paper():
    arf = AdaptiveRandomForestClassifier(n_models=1, seed=0)
    arf.learn_one({"x": 1.0}, 0)
    assert arf.members_[0].drift.delta == 0.001
    assert arf.members_[0].warn.delta == 0.01


def test_vote_is_weighted_by_recent_accuracy():
    arf = AdaptiveRandomForestClassifier(n_models=2, seed=0)
    arf.learn_one({"x": 1.0}, 0)
    good = _Member(_Stub({0: 1.0, 1: 0.0}), None, None)
    good.correct, good.seen = 9, 10
    bad = _Member(_Stub({0: 0.0, 1: 1.0}), None, None)
    bad.correct, bad.seen = 1, 10
    arf.members_ = [good, bad]
    arf.classes_ = {0, 1}
    p = arf.predict_proba_one({"x": 1.0})
    assert p[0] == pytest.approx(0.9)
    assert p[1] == pytest.approx(0.1)
    assert arf.predict_one({"x": 1.0}) == 0


def test_equal_vote_when_no_tree_has_been_scored():
    arf = AdaptiveRandomForestClassifier(n_models=2, seed=0)
    arf.learn_one({"x": 1.0}, 0)
    a = _Member(_Stub({0: 1.0, 1: 0.0}), None, None)
    b = _Member(_Stub({0: 0.0, 1: 1.0}), None, None)
    arf.members_ = [a, b]
    arf.classes_ = {0, 1}
    p = arf.predict_proba_one({"x": 1.0})
    assert p[0] == pytest.approx(0.5)


def test_background_tree_does_not_vote():
    arf = AdaptiveRandomForestClassifier(n_models=3, seed=0)
    for x, y in _stream(400):
        arf.learn_one(x, y)
    before = arf.predict_proba_one({"x": 0.1})
    for m in arf.members_:
        bg = SubspaceHoeffdingTree(grace_period=20, seed=0)
        for x, y in _stream(300, seed=9):
            bg.learn_one(x, 1 - y)
        m.background = bg
    assert arf.predict_proba_one({"x": 0.1}) == before


def test_warning_starts_a_background_tree_and_drift_promotes_it():
    arf = AdaptiveRandomForestClassifier(
        n_models=1, drift_detector=_Fire(at=8), warning_detector=_Fire(at=5), seed=0
    )
    stream = _stream(8)
    for _ in range(4):
        arf.learn_one(*next(stream))
    assert arf.members_[0].background is None and arf.n_warnings_ == 0
    arf.learn_one(*next(stream))
    bg = arf.members_[0].background
    assert isinstance(bg, SubspaceHoeffdingTree) and arf.n_warnings_ == 1
    for _ in range(2):
        arf.learn_one(*next(stream))
    assert arf.members_[0].background is bg and bg.n_seen_ > 0
    old = arf.members_[0].tree
    arf.learn_one(*next(stream))
    assert arf.members_[0].tree is bg and arf.members_[0].tree is not old
    assert arf.members_[0].background is None
    assert arf.n_drifts_ == 1


def test_drift_without_warning_replaces_the_tree_by_a_new_empty_one():
    arf = AdaptiveRandomForestClassifier(
        n_models=1, drift_detector=_Fire(at=3), warning_detector=_Fire(at=99), seed=0
    )
    stream = _stream(3)
    for _ in range(2):
        arf.learn_one(*next(stream))
    old = arf.members_[0].tree
    arf.learn_one(*next(stream))
    new = arf.members_[0].tree
    assert new is not old and new.n_seen_ == 0 and arf.n_drifts_ == 1


def test_recovers_after_the_labels_flip():
    arf = AdaptiveRandomForestClassifier(n_models=3, seed=0)
    for x, y in _stream(600):
        arf.learn_one(x, y)
    ok = []
    for x, y in _stream(900, flip_at=0, seed=2):
        ok.append(arf.predict_one(x) == y)
        arf.learn_one(x, y)
    assert arf.n_warnings_ >= 1
    assert np.mean(ok[-300:]) > 0.95


def test_replaced_member_restarts_its_score_and_detectors():
    arf = AdaptiveRandomForestClassifier(n_models=1, seed=0)
    for x, y in _stream(300):
        arf.learn_one(x, y)
    m = arf.members_[0]
    old_drift, old_warn = m.drift, m.warn
    m.background = SubspaceHoeffdingTree(seed=0)
    arf._promote(m)
    assert m.seen == 0 and m.correct == 0 and m.weight == 0.0
    assert m.background is None
    assert m.drift is not old_drift and m.warn is not old_warn


def test_same_seed_gives_the_same_forest():
    outs = []
    for _ in range(2):
        arf = AdaptiveRandomForestClassifier(n_models=3, seed=7)
        for x, y in _stream(300):
            arf.learn_one(x, y)
        outs.append(arf.predict_proba_one({"x": 2.4}))
    assert outs[0] == outs[1]


def test_follows_a_sudden_change_on_drift_stream():
    ds = DriftStream(n_samples=1200, n_features=4, n_drifts=1, kind="sudden", seed=0)
    X, y = ds.data()
    names = [f"x{i}" for i in range(4)]
    arf = AdaptiveRandomForestClassifier(n_models=5, seed=0)
    ok = []
    for xi, yi in zip(X, y):
        d = dict(zip(names, map(float, xi)))
        ok.append(arf.predict_one(d) == yi)
        arf.learn_one(d, int(yi))
    ok = np.array(ok)
    assert ok[400:600].mean() > 0.9
    assert ok[900:].mean() > 0.9
