"""Unit tests for dense_armor/utility/metric_learning.py."""
import doctest
import importlib
import sys

import numpy as np
import pytest

pytest.importorskip("river")

import dense_armor.learn.metric_learning as ml  # noqa: E402
from dense_armor.learn.metric_learning import (  # noqa: E402
    LEGO, MetricKNNClassifier, OASIS, POLA,
)


def test_oasis_first_distance_is_euclidean():
    o = OASIS(C=0.1)
    d = o.distance({"a": 1.0, "b": 0.0}, {"a": 0.0, "b": 1.0})
    assert d == pytest.approx(np.sqrt(2.0))


def test_oasis_margin_met_no_update():
    o = OASIS(C=0.1)
    x = {"a": 1.0, "b": 0.0}
    x_pos = {"a": 1.0, "b": 0.0}
    x_neg = {"a": 0.0, "b": 1.0}
    o.learn_triplet(x, x_pos, x_neg)
    W_before = o.W.copy()
    o.learn_triplet(x, x_pos, x_neg)
    assert np.array_equal(o.W, W_before)


def test_oasis_loss_goes_to_zero_when_C_is_large():
    o = OASIS(C=1e6)
    x = {"a": 1.0, "b": 0.0}
    x_pos = {"a": 0.9, "b": 0.1}
    x_neg = {"a": 0.1, "b": 0.9}
    for _ in range(3):
        o.learn_triplet(x, x_pos, x_neg)
    s_pos = o._to_array(x) @ o.W @ o._to_array(x_pos)
    s_neg = o._to_array(x) @ o.W @ o._to_array(x_neg)
    assert s_pos - s_neg >= 1.0 - 1e-6


def test_oasis_distance_is_reflexive_and_symmetric():
    o = OASIS(C=0.1)
    o.learn_triplet({"a": 1.0, "b": 0.0}, {"a": 0.9, "b": 0.1}, {"a": 0.1, "b": 0.9})
    assert o.distance({"a": 1.0, "b": 0.0}, {"a": 1.0, "b": 0.0}) == pytest.approx(0.0, abs=1e-6)
    d1 = o.distance({"a": 1.0, "b": 0.0}, {"a": 0.0, "b": 1.0})
    d2 = o.distance({"a": 0.0, "b": 1.0}, {"a": 1.0, "b": 0.0})
    assert d1 == pytest.approx(d2, abs=1e-9)


def test_lego_moves_distance_toward_target():
    lego = LEGO(eta=0.1)
    u = {"a": 1.0, "b": 0.0}
    v = {"a": 0.0, "b": 1.0}
    d_before = lego.distance(u, v)
    for _ in range(50):
        lego.learn_pair(u, v, y=0.01)
    d_after = lego.distance(u, v)
    assert d_after < d_before


def test_lego_percentile_targets_returns_ordered_values():
    X = np.array([[0.0, 0.0], [0.1, 0.0], [10.0, 10.0], [10.1, 10.0]])
    y = np.array([0, 0, 1, 1])
    near, far = LEGO.percentile_targets(X, y)
    assert near < far


def test_pola_stays_psd_and_b_ge_1():
    pola = POLA()
    a = {"x": 0.0}
    b = {"x": 1.0}
    for _ in range(20):
        pola.learn_pair(a, b, +1)
        pola.learn_pair(a, b, -1)
        eigs = np.linalg.eigvalsh(pola.A)
        assert eigs.min() >= -1e-9
        assert pola.b >= 1.0


def test_metric_knn_classifier_predicts_nearest_class():
    learner = OASIS(C=1e6)
    x_a = {"x": 1.0}
    x_b = {"x": 0.0}
    learner.learn_triplet(x_a, x_a, x_b)
    knn = MetricKNNClassifier(learner, n_neighbors=1)
    knn.learn_one(x_a, "A")
    knn.learn_one(x_b, "B")
    assert knn.predict_one({"x": 0.95}) == "A"
    assert knn.predict_one({"x": 0.05}) == "B"


def test_metric_knn_window_is_fifo():
    knn = MetricKNNClassifier(OASIS(C=0.1), n_neighbors=1, window_size=2)
    knn.learn_one({"x": 0.0}, "A")
    knn.learn_one({"x": 1.0}, "B")
    knn.learn_one({"x": 2.0}, "C")
    assert len(knn._window) == 2
    assert all(y != "A" for _, y in knn._window)


def test_metric_knn_passes_check_estimator():
    from river.checks import check_estimator
    check_estimator(MetricKNNClassifier(OASIS(C=0.1)))


def test_docstring_examples():
    assert doctest.testmod(ml).failed == 0


def test_missing_river_raises_clear_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "river", None)
    monkeypatch.delitem(sys.modules, "dense_armor.learn.metric_learning", raising=False)
    with pytest.raises(ModuleNotFoundError, match=r"dense-armor\[river\]"):
        importlib.import_module("dense_armor.learn.metric_learning")


def test_base_class_methods_are_abstract():
    from dense_armor.learn.metric_learning import MetricLearner
    m = MetricLearner()
    for call in (
        lambda: m.learn_triplet({}, {}, {}),
        lambda: m.learn_pair({}, {}, 1),
        lambda: m.distance({}, {}),
    ):
        with pytest.raises(NotImplementedError):
            call()


def test_lego_skips_identical_pair():
    lego = LEGO()
    u = {"a": 1.0, "b": 2.0}
    lego.learn_pair(u, u, 5.0)
    assert np.array_equal(lego.A, np.identity(2))


def test_pola_projection_and_distance():
    pola = POLA()
    a, b, c = {"x": 0.0, "y": 0.0}, {"x": 3.0, "y": 0.0}, {"x": 0.0, "y": 3.0}
    pola.learn_pair(a, b, -1)
    pola.learn_pair(a, c, +1)
    assert np.linalg.eigvalsh(pola.A).min() >= -1e-9
    assert pola.distance(a, b) > pola.distance(a, c)


def test_predict_proba_covers_all_seen_classes():
    knn = MetricKNNClassifier(OASIS(), n_neighbors=1)
    knn.learn_one({"x": 0.0}, "A")
    knn.learn_one({"x": 5.0}, "B")
    assert knn.predict_proba_one({"x": 0.1}) == {"A": 1.0, "B": 0.0}


def test_pola_projects_negative_eigenvalue():
    pola = POLA(b_init=0.5)
    pola.learn_pair({"x": 0.0, "y": 0.0}, {"x": 1.0, "y": 1.0}, +1)
    assert np.linalg.eigvalsh(pola.A).min() >= -1e-9


def test_metric_knn_learns_online_when_enabled():
    learner = OASIS(C=0.5)
    knn = MetricKNNClassifier(learner, n_neighbors=1, learn_metric=True, seed=42)
    for i in range(30):
        x = {"x": float(i % 5), "y": float((i * 2) % 7)}
        knn.learn_one(x, i % 2)
    assert learner.W is not None


def test_metric_knn_does_not_learn_when_disabled():
    learner = OASIS(C=0.5)
    knn = MetricKNNClassifier(learner, n_neighbors=1, learn_metric=False, seed=42)
    for i in range(30):
        x = {"x": float(i % 5), "y": float((i * 2) % 7)}
        knn.learn_one(x, i % 2)
    assert learner.W is None


def test_metric_knn_same_seed_reproducible():
    def run():
        learner = OASIS(C=0.5)
        knn = MetricKNNClassifier(learner, n_neighbors=1, learn_metric=True, seed=42)
        preds = []
        for i in range(30):
            x = {"x": float(i % 5), "y": float((i * 2) % 7)}
            preds.append(knn.predict_one(x))
            knn.learn_one(x, i % 2)
        return preds
    assert run() == run()


@pytest.mark.parametrize("learner_cls", [OASIS, LEGO, POLA])
def test_check_estimator_all_learners(learner_cls):
    from river.checks import check_estimator
    check_estimator(MetricKNNClassifier(learner_cls()))