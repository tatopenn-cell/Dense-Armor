"""River bridge tests: interop and the missing-river error message."""
import importlib
import sys

import pytest

pytest.importorskip("river")

from dense_armor.anomaly.filters import HampelFilter
from dense_armor.learn.online_classifiers import OnlineGaussianNB


def test_bridge_raises_without_river(monkeypatch):
    monkeypatch.setitem(sys.modules, "river", None)
    monkeypatch.delitem(sys.modules, "dense_armor.bridges.river",
                       raising=False)
    with pytest.raises(ModuleNotFoundError,
                       match=r"dense-armor\[river\]"):
        importlib.import_module("dense_armor.bridges.river")


def test_bridge_helpers_are_identity():
    from dense_armor.bridges import river as bridge
    m = OnlineGaussianNB()
    assert bridge.to_river(m) is m
    assert bridge.from_river(m) is m


def test_our_classifier_passes_river_check_estimator():
    from river.checks import check_estimator as river_check
    river_check(OnlineGaussianNB())


def test_our_classifier_inside_river_pipeline():
    from river import compose, preprocessing
    pipe = compose.Pipeline(
        preprocessing.StandardScaler(),
        OnlineGaussianNB(),
    )
    for i in range(60):
        pipe.learn_one({"a": float(i % 2), "b": float(i % 3)}, i % 2)
    p = pipe.predict_proba_one({"a": 0.0, "b": 0.0})
    assert 0 in p and 1 in p


def test_river_estimator_inside_our_pipeline():
    from river import linear_model
    pipe = (HampelFilter(radius=3, feature="a")
            | linear_model.LogisticRegression())
    for i in range(80):
        pipe.learn_one({"a": float(i % 2) - 0.5, "b": 1.0}, i % 2)
    pred = pipe.predict_one({"a": 0.4, "b": 1.0})
    assert pred in (0, 1)
