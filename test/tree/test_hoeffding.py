import numpy as np

from dense_armor.checks import check_estimator
from dense_armor.utility.tree.hoeffding import HoeffdingTreeClassifier


def _two_blob_stream(n, rng):
    for _ in range(n):
        y = int(rng.random() < 0.5)
        c = 0.0 if y == 0 else 5.0
        yield (
            {
                "x0": float(rng.normal(c, 0.4)),
                "x1": float(rng.normal(0.0, 1.0)),
            },
            y,
        )


def test_split_occurs_and_separates_blobs():
    rng = np.random.default_rng(0)
    h = HoeffdingTreeClassifier(grace_period=30)
    for x, y in _two_blob_stream(600, rng):
        h.learn_one(x, y=y)
    assert h.n_nodes_ > 1
    assert h.depth_ >= 1
    p0 = h.predict_one({"x0": 0.0, "x1": 0.0})
    p1 = h.predict_one({"x0": 5.0, "x1": 0.0})
    assert p0 == 0 and p1 == 1


def test_explain_returns_path():
    rng = np.random.default_rng(1)
    h = HoeffdingTreeClassifier(grace_period=20)
    for x, y in _two_blob_stream(400, rng):
        h.learn_one(x, y=y)
    path = h.explain_one({"x0": 0.0, "x1": 0.0})
    assert isinstance(path, list)
    for step in path:
        assert "feature" in step and "threshold" in step and "side" in step


def test_nan_counted():
    h = HoeffdingTreeClassifier()
    h.learn_one({"x0": 0.0}, y=0)
    h.learn_one({"x0": float("nan")}, y=1)
    assert h.n_missing == 1


def test_max_depth_caps_tree():
    rng = np.random.default_rng(2)
    h = HoeffdingTreeClassifier(grace_period=10, max_depth=2)
    for x, y in _two_blob_stream(500, rng):
        h.learn_one(x, y=y)
    assert h.depth_ <= 2


def test_max_nodes_default_finite():
    h = HoeffdingTreeClassifier()
    assert h.max_nodes == 10000


def test_check_estimator_hoeffding():
    check_estimator(HoeffdingTreeClassifier(grace_period=20))
