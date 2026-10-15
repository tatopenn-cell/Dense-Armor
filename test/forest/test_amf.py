"""Tests for the aggregated Mondrian forests."""

import doctest
import math

import numpy as np
import pytest

from dense_armor.checks import check_estimator
from dense_armor.utility.forest import amf as amf_mod
from dense_armor.utility.forest.amf import (
    AggregatedMondrianForestClassifier,
    AggregatedMondrianForestRegressor,
    _AmfTree,
    _Frequency,
    _Mean,
)


def _cells(tree):
    out = {}

    def rec(node, lo, hi):
        out[id(node)] = (node, lo, hi)
        if node.left is not None:
            h = hi.copy()
            h[node.j] = node.s
            rec(node.left, lo, h)
            lo2 = lo.copy()
            lo2[node.j] = node.s
            rec(node.right, lo2, hi)

    rec(tree.root, tree.low.copy(), tree.high.copy())
    return out


def _prunings(node):
    if node.left is None:
        return [([node], 1)]
    out = [([node], 1)]
    for la, a in _prunings(node.left):
        for lb, b in _prunings(node.right):
            out.append((la + lb, 1 + a + b))
    return out


def _inside(x, lo, hi):
    return bool(np.all(x >= lo) and np.all(x <= hi))


def _equation_6(tree, X, Y, xq):
    """Equation 6 of the paper, summed over every pruning of the final tree.

    A leaf of the final tree stands for all its refinements, which have
    the same loss, so its prior mass is 1 instead of 1/2.
    """
    fc = tree.fc
    n = len(X)
    cells = _cells(tree)
    member = {
        k: [_inside(X[s], lo, hi) for s in range(n)] for k, (_, lo, hi) in cells.items()
    }

    def stat_before(node, s):
        st = fc.empty()
        for u in range(s):
            if member[id(node)][u]:
                fc.update(st, Y[u])
        return st

    logw, preds = [], []
    for leaves, size in _prunings(tree.root):
        loss = 0.0
        for s in range(n):
            lf = next(v for v in leaves if member[id(v)][s])
            loss += fc.loss(stat_before(lf, s), Y[s])
        absorbed = sum(1 for v in leaves if v.left is None)
        logw.append((absorbed - size) * math.log(2.0) - tree.step * loss)
        lf = next(v for v in leaves if _inside(xq, cells[id(v)][1], cells[id(v)][2]))
        preds.append(fc.predict(stat_before(lf, n)))
    w = np.exp(np.array(logw) - max(logw))
    w /= w.sum()
    if isinstance(preds[0], dict):
        return {c: float(sum(wi * p[c] for wi, p in zip(w, preds))) for c in preds[0]}
    return float(sum(wi * p for wi, p in zip(w, preds)))


def _build(kind, step, seed, n=7, d=2):
    rng = np.random.default_rng(seed)
    X = rng.random((n, d))
    if kind == "reg":
        Y = list(rng.uniform(-1.0, 1.0, n))
        fc = _Mean()
    else:
        Y = [int(v) for v in rng.integers(0, 3, n)]
        fc = _Frequency(0.5, [0, 1, 2])
    tree = _AmfTree(d, 0.0, 1.0, step, fc, np.random.default_rng(100 + seed))
    for x, y in zip(X, Y):
        tree.add_point(x)
        tree.update_weights(x, y)
    return tree, X, Y


def test_doctests():
    res = doctest.testmod(amf_mod)
    assert res.failed == 0 and res.attempted > 0


def test_check_estimator_passes():
    check_estimator(AggregatedMondrianForestClassifier(n_trees=3, seed=0))
    check_estimator(AggregatedMondrianForestRegressor(n_trees=3, seed=0))


def test_collapsed_weights_equal_equation_6():
    worst = 0.0
    for kind in ("reg", "cls"):
        for step in (0.125, 1.0, 5.0, 200.0):
            for seed in range(3):
                tree, X, Y = _build(kind, step, seed)
                for q in range(len(X)):
                    a = tree.predict(X[q])
                    b = _equation_6(tree, X, Y, X[q])
                    if kind == "cls":
                        worst = max(worst, max(abs(a[c] - b[c]) for c in a))
                    else:
                        worst = max(worst, abs(a - b))
    assert worst < 1e-9


def _prequential(kind, seed, n, structured):
    rng = np.random.default_rng(seed)
    d = 2
    X = rng.random((n, d))
    if kind == "cls":
        k = 3
        Y = [int(v) for v in rng.integers(0, k, n)]
        if structured:
            Y = [int(a > 0.5) + int(rng.random() < 0.1) for a in X[:, 0]]
        fc = _Frequency(0.5, list(range(k)))
        step = 1.0
    else:
        Y = np.clip(0.3 * rng.normal(size=n), -1.0, 1.0)
        if structured:
            Y = np.clip(np.sin(6 * X[:, 0]) + 0.1 * rng.normal(size=n), -1.0, 1.0)
        fc = _Mean()
        step = 1.0 / 8.0
    tree = _AmfTree(d, 0.0, 1.0, step, fc, np.random.default_rng(seed + 1000))
    root = fc.empty()
    agg = root_loss = 0.0
    for x, y in zip(X, Y):
        tree.add_point(x)
        p = tree.predict(x)
        agg += -math.log(p[y]) if kind == "cls" else (p - y) ** 2
        root_loss += fc.loss(root, y)
        fc.update(root, y)
        tree.update_weights(x, y)
    if kind == "cls":
        freq = np.bincount(Y, minlength=3) / n
        const = -sum(math.log(freq[y]) for y in Y)
    else:
        const = float(((np.asarray(Y) - np.mean(Y)) ** 2).sum())
    return agg, root_loss, const, step


def test_lemma_1_regret_against_the_root_forecaster():
    for kind in ("cls", "reg"):
        for structured in (False, True):
            for seed in range(3):
                agg, root, _, step = _prequential(kind, seed, 300, structured)
                assert agg - root <= math.log(2.0) / step + 1e-9


def test_corollaries_1_and_2_against_the_best_constant():
    n = 300
    for structured in (False, True):
        for seed in range(3):
            agg, _, const, _ = _prequential("cls", seed, n, structured)
            assert agg - const <= math.log(2.0) + 2 * (3 - 1) / 4 * math.log(4 * n)
            agg, _, const, _ = _prequential("reg", seed, n, structured)
            assert agg - const <= 4.0 * 2 * math.log(n)


def test_large_step_tends_to_the_unaggregated_prediction_on_a_clean_step():
    f = AggregatedMondrianForestRegressor(n_trees=1, step=1e4, seed=3)
    rng = np.random.default_rng(0)
    xs = rng.random(200)
    for x in xs:
        f.learn_one({"x": float(x)}, 1.0 if x > 0.5 else -1.0)
    for x in xs[:50]:
        q = {"x": float(x)}
        assert f.predict_one(q) == pytest.approx(
            f.predict_unaggregated_one(q), abs=1e-9
        )


def test_prediction_does_not_change_the_trees():
    f = AggregatedMondrianForestRegressor(n_trees=2, seed=0)
    for i in range(40):
        f.learn_one({"a": (i % 7) / 7.0, "b": (i % 5) / 5.0}, float(i % 3))
    before = [(t.n_nodes, t.root.lw, t.root.lwb) for t in f.trees_]
    for _ in range(20):
        f.predict_one({"a": 0.31, "b": 0.77})
    assert [(t.n_nodes, t.root.lw, t.root.lwb) for t in f.trees_] == before


def test_memory_is_about_four_nodes_per_sample():
    f = AggregatedMondrianForestRegressor(n_trees=1, seed=0)
    rng = np.random.default_rng(0)
    for _ in range(100):
        f.learn_one({"a": float(rng.random()), "b": float(rng.random())}, 0.0)
    assert f.trees_[0].n_nodes <= 5 * 100


def test_predict_before_learning():
    c = AggregatedMondrianForestClassifier(n_trees=2, seed=0)
    assert c.predict_one({"x": 0.5}) is None and c.predict_proba_one({"x": 0.5}) == {}
    r = AggregatedMondrianForestRegressor(n_trees=2, seed=0)
    assert r.predict_one({"x": 0.5}) == 0.0


def test_probabilities_sum_to_one_and_follow_the_labels():
    f = AggregatedMondrianForestClassifier(n_trees=5, seed=0)
    for i in range(200):
        x = (i % 20) / 19.0
        f.learn_one({"x": x}, "low" if x < 0.5 else "high")
    p = f.predict_proba_one({"x": 0.05})
    assert sum(p.values()) == pytest.approx(1.0)
    assert f.predict_one({"x": 0.05}) == "low"
    assert f.predict_one({"x": 0.95}) == "high"


def test_nan_and_missing_are_skipped_and_counted():
    for f, y in (
        (AggregatedMondrianForestClassifier(n_trees=2, seed=0), 1),
        (AggregatedMondrianForestRegressor(n_trees=2, seed=0), 1.0),
    ):
        for i in range(30):
            f.learn_one({"a": i / 30.0, "b": 0.5}, y)
        n0, seen = f.n_missing, f.n_seen_
        f.learn_one({"a": float("nan"), "b": 0.5}, y)
        f.learn_one({"a": float("inf"), "b": 0.5}, y)
        f.learn_one({"a": 0.5}, y)
        f.learn_one({}, y)
        f.learn_one({"a": 0.5, "b": 0.5}, None)
        assert f.n_missing == n0 + 5 and f.n_seen_ == seen
        for tree in f.trees_:
            assert math.isfinite(tree.root.lw) and math.isfinite(tree.root.lwb)
        f.learn_one({"a": 0.2, "b": 0.5}, y)
        assert f.n_seen_ == seen + 1


def test_regressor_skips_a_non_numeric_target():
    f = AggregatedMondrianForestRegressor(n_trees=1, seed=0)
    f.learn_one({"a": 0.1}, "x")
    f.learn_one({"a": 0.1}, float("nan"))
    assert f.n_missing == 2 and f.n_seen_ == 0


def test_inputs_outside_the_box_are_clipped():
    f = AggregatedMondrianForestRegressor(n_trees=1, seed=0)
    f.learn_one({"a": 7.0}, 1.0)
    f.learn_one({"a": 1.0}, 1.0)
    assert f.n_seen_ == 2 and f.trees_[0].n_nodes == 1


def test_fixed_classes_skip_an_unknown_label():
    f = AggregatedMondrianForestClassifier(n_trees=1, classes=["a", "b"], seed=0)
    f.learn_one({"x": 0.1}, "a")
    f.learn_one({"x": 0.9}, "c")
    assert f.n_missing == 1 and f.n_seen_ == 1
    assert set(f.predict_proba_one({"x": 0.1})) == {"a", "b"}


def test_constructor_arguments_not_overwritten_by_learning():
    f = AggregatedMondrianForestClassifier(
        n_trees=3, step=0.7, dirichlet=0.3, low=-1.0, high=2.0, seed=4
    )
    before = f.get_params()
    for i in range(50):
        f.learn_one({"x": (i % 9) / 3.0 - 1.0}, i % 2)
    assert f.get_params() == before
    c = f.clone()
    assert c.get_params() == before and c.trees_ == [] and c.n_seen_ == 0
    assert f.alphabet_ == [0, 1]


def test_bad_parameters_raise():
    for kw in (
        {"n_trees": 0},
        {"step": 0.0},
        {"dirichlet": 0.0},
        {"low": 1.0, "high": 1.0},
    ):
        with pytest.raises(ValueError):
            AggregatedMondrianForestClassifier(**kw)
    for kw in ({"n_trees": 0}, {"step": -1.0}, {"low": 2.0, "high": 1.0}):
        with pytest.raises(ValueError):
            AggregatedMondrianForestRegressor(**kw)


def test_same_seed_gives_the_same_forest():
    outs = []
    for _ in range(2):
        f = AggregatedMondrianForestRegressor(n_trees=3, seed=11)
        for i in range(60):
            f.learn_one({"x": (i % 13) / 13.0}, float(i % 4))
        outs.append([f.predict_one({"x": 0.4}) for _ in range(3)])
    assert outs[0] == outs[1]


def test_the_dirichlet_forecaster_is_the_kt_forecaster_by_default():
    fc = _Frequency(0.5, ["a", "b"])
    st = fc.empty()
    for y in ("a", "a", "b"):
        fc.update(st, y)
    p = fc.predict(st)
    assert p["a"] == pytest.approx((2 + 0.5) / (3 + 1.0))
    assert p["b"] == pytest.approx((1 + 0.5) / (3 + 1.0))
    assert sum(p.values()) == pytest.approx(1.0)


def test_max_nodes_bounds_every_tree_and_learning_goes_on():
    rng = np.random.default_rng(0)
    for cls, y_of in (
        (AggregatedMondrianForestClassifier, lambda a: int(a > 0.5)),
        (AggregatedMondrianForestRegressor, lambda a: 1.0 if a > 0.5 else -1.0),
    ):
        f = cls(n_trees=2, max_nodes=21, seed=0)
        for _ in range(300):
            a = float(rng.random())
            f.learn_one({"a": a, "b": float(rng.random())}, y_of(a))
        assert f.n_seen_ == 300
        assert all(t.n_nodes <= 21 for t in f.trees_)
        assert max(t.n_nodes for t in f.trees_) >= 19
        for t in f.trees_:
            assert math.isfinite(t.root.lw) and math.isfinite(t.root.lwb)
        p = f.predict_one({"a": 0.9, "b": 0.5})
        assert p == 1 if cls is AggregatedMondrianForestClassifier else p > 0.0


def test_max_nodes_of_one_keeps_a_single_leaf_that_averages():
    f = AggregatedMondrianForestRegressor(n_trees=1, max_nodes=1, seed=0)
    for i in range(10):
        f.learn_one({"a": i / 10.0}, float(i))
    assert f.trees_[0].n_nodes == 1
    assert f.predict_one({"a": 0.3}) == pytest.approx(4.5)


def test_max_nodes_none_lifts_the_limit_and_zero_is_refused():
    f = AggregatedMondrianForestRegressor(n_trees=1, max_nodes=None, seed=0)
    rng = np.random.default_rng(1)
    for _ in range(200):
        f.learn_one({"a": float(rng.random())}, 0.0)
    assert f.trees_[0].n_nodes > 300
    for cls in (AggregatedMondrianForestClassifier, AggregatedMondrianForestRegressor):
        with pytest.raises(ValueError):
            cls(max_nodes=0)


def test_no_check_is_skipped():
    for f in (
        AggregatedMondrianForestClassifier(),
        AggregatedMondrianForestRegressor(),
    ):
        assert f._unit_test_skips() == set()


def test_none_input_is_counted_and_answered_empty():
    c = AggregatedMondrianForestClassifier(n_trees=1, seed=0)
    c.learn_one(None, 0)
    assert c.n_missing == 1
    c.learn_one({"x": 0.2}, 0)
    assert c.predict_proba_one(None) == {}
    r = AggregatedMondrianForestRegressor(n_trees=1, seed=0)
    r.learn_one(None, 1.0)
    assert r.n_missing == 1
    r.learn_one({"x": 0.2}, 1.0)
    assert r.predict_one(None) == 0.0


def test_unaggregated_label_is_the_leaf_label():
    f = AggregatedMondrianForestClassifier(n_trees=3, seed=0)
    for i in range(40):
        x = (i % 20) / 19.0
        f.learn_one({"x": x}, 0 if x < 0.5 else 1)
    assert f.predict_unaggregated_one({"x": 0.05}) == 0
    assert f.predict_unaggregated_one({"x": 0.95}) == 1
    assert (
        AggregatedMondrianForestClassifier().predict_unaggregated_one({"x": 0.1})
        is None
    )


def test_a_new_label_is_seen_by_the_forecasters_after_a_restore():
    import pickle

    f = AggregatedMondrianForestClassifier(n_trees=2, seed=0)
    for i in range(30):
        f.learn_one({"x": (i % 10) / 10.0}, i % 2)
    via_state = AggregatedMondrianForestClassifier(n_trees=2, seed=0)
    via_state.load_state_dict(f.state_dict())
    for g in (
        via_state,
        pickle.loads(pickle.dumps(f)),
        f.clone(include_attributes=True),
    ):
        g.learn_one({"x": 0.5}, "new")
        p = g.predict_proba_one({"x": 0.5})
        assert set(p) == {0, 1, "new"}
        assert sum(p.values()) == pytest.approx(1.0)
