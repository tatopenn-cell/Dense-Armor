"""Tests for the k-nearest-neighbour classifiers and regressors."""

import doctest

import numpy as np
import pytest

from dense_armor.checks import check_estimator
from dense_armor.utility.neighbors import KNNClassifier, KNNRegressor, LazySearch
from dense_armor.utility.neighbors import knn as knn_mod


def test_doctests():
    r = doctest.testmod(knn_mod)
    assert r.failed == 0


def test_minkowski_p1_vs_p2():
    """p=2 gives Euclidean, p=1 gives Manhattan.

    Window has (0,0) and (3,4). Query (6,8):
        p=2: distance to (3,4) = sqrt(9+16) = 5
             distance to (0,0) = sqrt(36+64) = 10
        p=1: distance to (3,4) = 3+4 = 7
             distance to (0,0) = 6+8 = 14
    The nearest neighbour is (3,4) in both cases, with different
    distances.
    """
    s2 = LazySearch(window_size=10, n_neighbors=1, p=2.0)
    s1 = LazySearch(window_size=10, n_neighbors=1, p=1.0)
    s2.add({"a": 0.0, "b": 0.0}, 0)
    s2.add({"a": 3.0, "b": 4.0}, 1)
    s1.add({"a": 0.0, "b": 0.0}, 0)
    s1.add({"a": 3.0, "b": 4.0}, 1)
    _y2, d2 = s2.neighbours({"a": 6.0, "b": 8.0})
    _y1, d1 = s1.neighbours({"a": 6.0, "b": 8.0})
    assert abs(d2[0] - 5.0) < 1e-12
    assert abs(d1[0] - 7.0) < 1e-12


def test_forget_is_exact():
    s = LazySearch(window_size=10, n_neighbors=2, p=2.0)
    for i in range(5):
        s.add({"x": float(i)}, i)
    before_len = len(s)
    y_forgotten = s.forget()
    assert y_forgotten == 0
    assert len(s) == before_len - 1
    ys, _ds = s.neighbours({"x": 1.0})
    assert 0 not in ys


def test_window_drops_oldest_when_full():
    s = LazySearch(window_size=3, n_neighbors=1, p=2.0)
    for i in range(5):
        s.add({"x": float(i)}, i)
    assert len(s) == 3
    ys, _ds = s.neighbours({"x": 0.0})
    assert ys == [2]


def test_knn_classifier_majority():
    clf = KNNClassifier(window_size=20, n_neighbors=3)
    for _ in range(5):
        clf.learn_one({"a": 0.0, "b": 0.0}, "A")
        clf.learn_one({"a": 1.0, "b": 1.0}, "B")
    assert clf.predict_one({"a": 0.05, "b": 0.05}) == "A"
    assert clf.predict_one({"a": 0.95, "b": 0.95}) == "B"


def test_knn_classifier_distance_weights():
    clf = KNNClassifier(window_size=10, n_neighbors=3, weights="distance")
    clf.learn_one({"x": 0.0}, "A")
    clf.learn_one({"x": 0.1}, "A")
    clf.learn_one({"x": 10.0}, "B")
    proba = clf.predict_proba_one({"x": 0.0})
    assert proba["A"] > proba["B"]


def test_knn_regressor_mean():
    reg = KNNRegressor(window_size=10, n_neighbors=3)
    for i in range(10):
        reg.learn_one({"x": float(i)}, float(i))
    assert abs(reg.predict_one({"x": 4.4}) - 4.0) < 1e-9


def test_knn_regressor_return_std():
    reg = KNNRegressor(window_size=10, n_neighbors=3)
    for i in range(10):
        reg.learn_one({"x": float(i)}, float(i))
    out, std = reg.predict_one({"x": 4.4}, return_std=True)
    assert isinstance(out, float)
    assert std >= 0.0


def test_knn_nan_and_missing_counted():
    clf = KNNClassifier(window_size=10, n_neighbors=1)
    for i in range(5):
        clf.learn_one({"a": float(i)}, i)
    n0 = clf.n_missing
    clf.learn_one({"a": float("nan")}, 5)
    clf.learn_one({}, 5)
    clf.learn_one("not a dict", 5)
    assert clf.n_missing == n0 + 3
    assert clf.predict_proba_one({"a": float("nan")}) == {}

    reg = KNNRegressor(window_size=10, n_neighbors=1)
    for i in range(5):
        reg.learn_one({"a": float(i)}, float(i))
    n0 = reg.n_missing
    reg.learn_one({"a": float("nan")}, 5.0)
    reg.learn_one({"a": 1.0}, float("nan"))
    assert reg.n_missing == n0 + 2
    assert reg.predict_one({"a": float("nan")}) == 0.0


def test_check_estimator_passes():
    check_estimator(KNNClassifier(window_size=20, n_neighbors=3))
    check_estimator(KNNRegressor(window_size=20, n_neighbors=3))


def test_bad_parameters():
    with pytest.raises(ValueError, match="window_size"):
        KNNClassifier(window_size=0)
    with pytest.raises(ValueError, match="n_neighbors"):
        KNNClassifier(window_size=5, n_neighbors=10)
    with pytest.raises(ValueError, match="p must be"):
        KNNClassifier(p=0.5)
    with pytest.raises(ValueError, match="weights"):
        KNNClassifier(weights="banana")
    with pytest.raises(ValueError, match="eps"):
        KNNClassifier(eps=0.0)


def test_clone_not_overwritten():
    clf = KNNClassifier(window_size=20, n_neighbors=3, p=1.5, weights="distance")
    for i in range(10):
        clf.learn_one({"a": float(i)}, i % 2)
    c = clf.clone()
    assert c.window_size == 20
    assert c.n_neighbors == 3
    assert c.p == 1.5
    assert c.weights == "distance"


def test_optional_metric():
    class ZeroMetric:
        def distance(self, x, x2):
            return 0.0

    clf = KNNClassifier(window_size=5, n_neighbors=1, metric=ZeroMetric())
    clf.learn_one({"a": 0.0}, "A")
    clf.learn_one({"a": 1.0}, "B")
    assert clf.predict_one({"a": 0.5}) in ("A", "B")


def test_lazy_search_edge_cases():
    with pytest.raises(ValueError):
        LazySearch(n_neighbors=0)
    s = LazySearch(window_size=5, n_neighbors=1, p=3.0)
    assert s.forget() is None
    assert s.neighbours({"a": 0.0}) == ([], [])
    assert s._distance_to_window(np.zeros(1)).size == 0
    s.add({"a": 0.0, "b": 0.0}, "A")
    ys, ds = s.neighbours({"a": 1.0, "b": 1.0})
    assert ys == ["A"] and ds[0] == pytest.approx(2.0 ** (1.0 / 3.0))
    assert s.forget() == "A"


def test_classifier_skips_unusable_samples_and_answers_empty():
    clf = KNNClassifier(window_size=5, n_neighbors=1)
    assert clf.predict_proba_one({"a": 1.0}) == {}
    for bad in ({}, "a", {"a": "word"}, {"a": float("nan")}):
        clf.learn_one(bad, "A")
    assert clf.n_missing == 4
    clf.learn_one({"a": 0.0}, "A")
    assert clf.predict_proba_one({"a": "word"}) == {}
    assert clf.predict_proba_one({"a": float("nan")}) == {}
    assert clf.predict_one({"a": 0.1}) == "A"


def test_regressor_skips_unusable_samples_and_answers_zero():
    reg = KNNRegressor(window_size=5, n_neighbors=2)
    assert reg.predict_one({"a": 1.0}) == 0.0
    for x, y in (
        ({}, 1.0),
        ({"a": "word"}, 1.0),
        ({"a": 1.0}, "word"),
        ({"a": 1.0}, float("nan")),
    ):
        reg.learn_one(x, y)
    assert reg.n_missing == 4
    reg.learn_one({"a": 0.0}, 1.0)
    reg.learn_one({"a": 2.0}, 3.0)
    assert reg.predict_one({"a": "word"}) == 0.0
    assert reg.predict_one({"a": float("nan")}) == 0.0
    assert reg.predict_one({"a": 1.0}) == pytest.approx(2.0)


def test_regressor_distance_weights_and_bad_parameters():
    reg = KNNRegressor(window_size=5, n_neighbors=2, weights="distance")
    reg.learn_one({"a": 0.0}, 0.0)
    reg.learn_one({"a": 4.0}, 4.0)
    assert reg.predict_one({"a": 1.0}) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        KNNRegressor(weights="other")
    with pytest.raises(ValueError):
        KNNRegressor(eps=0.0)


def test_no_check_is_skipped():
    assert KNNClassifier()._unit_test_skips() == set()
    assert KNNRegressor()._unit_test_skips() == set()
