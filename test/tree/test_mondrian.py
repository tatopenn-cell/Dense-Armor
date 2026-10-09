import numpy as np

from dense_armor.checks import check_estimator
from dense_armor.utility.tree.mondrian import (
    MondrianForestClassifier,
    MondrianForestRegressor,
)


def _stream(rng, n):
    for _ in range(n):
        y = int(rng.random() < 0.5)
        c = 0.0 if y == 0 else 5.0
        yield {"x0": float(rng.normal(c, 0.4))}, y


def test_mondrian_regressor_estimate_var():
    rng = np.random.default_rng(0)
    f = MondrianForestRegressor(n_trees=10, seed=0)
    for _ in range(500):
        x = float(rng.uniform(-1.0, 1.0))
        y = float(np.sin(3.0 * x))
        f.learn_one({"x": x}, y=y)
    est_near = f.predict_one({"x": 0.0}, return_estimate=True)
    est_far = f.predict_one({"x": 5.0}, return_estimate=True)
    print(f"var near = {est_near.var:.4f}, var far = {est_far.var:.4f}")
    assert est_near.var >= 0.0
    assert est_far.var >= 0.0
    assert est_far.var > est_near.var


def test_mondrian_split_times_are_valid():
    rng = np.random.default_rng(1)
    f = MondrianForestRegressor(n_trees=3, seed=0)
    for _ in range(200):
        x = float(rng.uniform(0.0, 1.0))
        f.learn_one({"x": x}, y=x)
    tree = f.trees_[0]

    def walk(node, parent_tau):
        if node.is_leaf:
            return
        assert node.tau >= parent_tau
        walk(node.left, node.tau)
        walk(node.right, node.tau)

    walk(tree.root, 0.0)


def test_mondrian_online_root_split_feature_stable():
    n_seeds = 30
    stream = [{"x0": float(i)} for i in range(1, 8)]
    feat_set = set()
    for seed in range(n_seeds):
        f = MondrianForestRegressor(n_trees=1, seed=seed)
        for d in stream:
            f.learn_one(d, y=float(d["x0"]))
        r = f.trees_[0].root
        if not r.is_leaf:
            feat_set.add(r.feature)
    print(f"root features over {n_seeds} seeds: {feat_set}")
    assert feat_set
    assert feat_set == {"x0"}


def test_mondrian_classifier_smoke():
    rng = np.random.default_rng(2)
    f = MondrianForestClassifier(n_trees=15, seed=0)
    for x, y in _stream(rng, 800):
        f.learn_one(x, y=y)
    p0 = f.predict_one({"x0": 0.0})
    p1 = f.predict_one({"x0": 5.0})
    assert p0 == 0 and p1 == 1


def test_mondrian_nan_counted():
    f = MondrianForestRegressor(n_trees=3, seed=0)
    f.learn_one({"x0": 0.0}, y=0.0)
    f.learn_one({"x0": float("nan")}, y=0.0)
    assert f.n_missing == 1


def test_mondrian_max_nodes_default_finite():
    f = MondrianForestRegressor(n_trees=3, seed=0)
    assert f.max_nodes == 10000


def test_check_estimator_mondrian_regressor():
    check_estimator(MondrianForestRegressor(n_trees=3, seed=0))


def test_check_estimator_mondrian_classifier():
    check_estimator(MondrianForestClassifier(n_trees=3, seed=0))


def _walk(node):
    stack = [node]
    while stack:
        n = stack.pop()
        yield n
        stack.extend(c for c in (n.left, n.right) if c is not None)


def test_mondrian_extension_inserts_parent_with_cut_in_extension():
    rng = np.random.default_rng(0)
    f = MondrianForestRegressor(n_trees=5, seed=0)
    for _ in range(50):
        f.learn_one({"a": float(rng.random()), "b": float(rng.random())}, y=0.0)
    f.learn_one({"a": 10.0, "b": 0.5}, y=1.0)
    roots = [t.root for t in f.trees_]
    assert any(r.feature == "a" and 1.0 <= r.threshold <= 10.0 for r in roots)
    for t in f.trees_:
        assert t.root.leaf.n_ == 51
        for n in _walk(t.root):
            if n.is_leaf:
                assert n.tau == t.lifetime


def test_mondrian_branch_probability_uses_time_difference():
    from dense_armor.utility.tree.mondrian import _MNode

    f = MondrianForestRegressor(n_trees=1, seed=0)
    f.learn_one({"a": 0.0}, y=0.0)
    tree = f.trees_[0]
    node = _MNode(0, 2.0)
    node.lx_ = {"a": 0.0}
    node.ux_ = {"a": 1.0}
    p = tree._branch_prob(node, {"a": 2.0}, parent_tau=1.5)
    assert abs(p - (1.0 - np.exp(-0.5))) < 1e-12
    assert tree._branch_prob(node, {"a": 0.5}, parent_tau=1.5) == 0.0
