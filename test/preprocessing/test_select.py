from dense_armor.checks import check_estimator
from dense_armor.utility.preprocessing.select import (
    SelectKBest,
    VarianceThreshold,
)


def test_variance_threshold_keeps_high_variance():
    vt = VarianceThreshold(threshold=0.5)
    for v in [1.0, 2.0, 3.0, 4.0, 5.0]:
        vt.learn_one({"x": v, "c": 7.0})
    out = vt.transform_one({"x": 3.0, "c": 7.0})
    assert "x" in out
    assert "c" not in out


def test_select_k_best_correlation():
    sk = SelectKBest(k=1)
    for i in range(30):
        sk.learn_one({"a": float(i), "b": float(-i)}, y=float(i))
    out = sk.transform_one({"a": 1.0, "b": 1.0})
    assert "a" in out
    assert "b" not in out


def test_check_estimator_variance():
    check_estimator(VarianceThreshold())


def test_check_estimator_select():
    check_estimator(SelectKBest())
