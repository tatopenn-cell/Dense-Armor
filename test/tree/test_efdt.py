import numpy as np

from dense_armor.checks import check_estimator
from dense_armor.utility.tree.efdt import HoeffdingAnytimeTreeClassifier


def _iter_nodes(root):
    stack = [root]
    while stack:
        n = stack.pop()
        yield n
        if n.left_ is not None:
            stack.append(n.left_)
        if n.right_ is not None:
            stack.append(n.right_)


def _blobs(rng, n):
    for _ in range(n):
        y = int(rng.random() < 0.5)
        c = 0.0 if y == 0 else 5.0
        yield {"x0": float(rng.normal(c, 0.4))}, y


def test_efdt_root_split_follows_best_attribute():
    rng = np.random.default_rng(0)
    model = HoeffdingAnytimeTreeClassifier(grace_period=100, max_depth=1, delta=1e-3)
    for i in range(4000):
        y = int(rng.random() < 0.5)
        c = 0.0 if y == 0 else 5.0
        if i < 1500:
            d = {
                "x0": float(rng.normal(c, 0.4)),
                "x1": float(rng.normal(0.0, 1.0)),
            }
        else:
            d = {"x0": 0.0, "x1": float(rng.normal(c, 0.4))}
        model.learn_one(d, y=y)
    root = model.root_
    assert root is not None and not root.is_leaf
    assert model.n_reevals_ > 0
    assert root.split_feature_ == "x1"
    assert model.n_replaced_ >= 1


def test_efdt_internal_counts_grow():
    rng = np.random.default_rng(1)
    model = HoeffdingAnytimeTreeClassifier(grace_period=100, delta=1e-5)
    for _ in range(5000):
        y = int(rng.random() < 0.5)
        c = 0.0 if y == 0 else 5.0
        model.learn_one({"x0": float(rng.normal(c, 0.4))}, y=y)
    internal = [n for n in _iter_nodes(model.root_) if not n.is_leaf]
    assert internal
    for n in internal:
        assert n.n_ >= model.grace_period


def test_efdt_smoke():
    rng = np.random.default_rng(2)
    model = HoeffdingAnytimeTreeClassifier(grace_period=30)
    for x, y in _blobs(rng, 600):
        model.learn_one(x, y=y)
    assert model.predict_one({"x0": 0.0}) == 0
    assert model.predict_one({"x0": 5.0}) == 1


def test_efdt_nan_counted():
    model = HoeffdingAnytimeTreeClassifier()
    model.learn_one({"x0": 0.0}, y=0)
    model.learn_one({"x0": float("nan")}, y=1)
    assert model.n_missing == 1


def test_check_estimator_efdt():
    check_estimator(HoeffdingAnytimeTreeClassifier(grace_period=20))
