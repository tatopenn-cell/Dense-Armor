"""Edge paths shared by the factorization models: inputs, labels, abstract losses."""

import numpy as np
import pytest

from dense_armor.utility.facto import (
    FFMClassifier,
    FFMRegressor,
    FMClassifier,
    FMRegressor,
    FwFMClassifier,
    FwFMRegressor,
    HOFMClassifier,
    HOFMRegressor,
)
from dense_armor.utility.optim import SGD

FIELDS = {"a": "F1", "b": "F1", "c": "F2"}
ALL = [FMRegressor, FMClassifier, FFMRegressor, FFMClassifier,
       FwFMRegressor, FwFMClassifier, HOFMRegressor, HOFMClassifier]
CLASSIFIERS = [FMClassifier, FFMClassifier, FwFMClassifier, HOFMClassifier]


def _make(cls, **kw):
    name = cls.__name__
    if name.startswith("FFM"):
        return cls(fields=FIELDS, k=2, lr=0.1, seed=0, **kw)
    if name.startswith("FwFM"):
        return cls(optimizer=SGD(lr=0.05), fields=FIELDS, k=2, seed=0, **kw)
    if name.startswith("HOFM"):
        return cls(optimizer=SGD(lr=0.05), m=3, k=2, seed=0, **kw)
    return cls(optimizer=SGD(lr=0.05), k=2, seed=0, **kw)


class _WithToDict:
    def to_dict(self):
        return {"a": 1.0}


@pytest.mark.parametrize("cls", ALL)
def test_as_dict_accepts_to_dict_objects_and_rejects_other_types(cls):
    m = _make(cls)
    assert m._as_dict(_WithToDict()) == {"a": 1.0}
    with pytest.raises(TypeError):
        m._as_dict(3)


@pytest.mark.parametrize("cls", ALL)
def test_feature_keys_argument_fixes_the_keys(cls):
    m = _make(cls, feature_keys=["a", "b", "c"])
    m.learn_one({"a": 1.0, "b": 2.0, "c": 0.5}, 1.0)
    assert m._keys == ["a", "b", "c"]


@pytest.mark.parametrize("cls", ALL)
def test_non_numeric_value_or_label_is_counted_as_missing(cls):
    m = _make(cls)
    m.learn_one({"a": "x", "b": 1.0}, 1.0)
    m.learn_one({"a": 1.0, "b": 1.0}, "bad")
    assert m.n_missing == 2


@pytest.mark.parametrize("cls", ALL)
def test_the_loss_of_the_base_class_is_abstract(cls):
    m = _make(cls)
    name = "_grad_error" if cls.__name__.startswith("FFM") else "_loss_grad"
    base = next(c for c in cls.__mro__[1:] if name in c.__dict__)
    with pytest.raises(NotImplementedError):
        getattr(base, name)(m, 0.0, 1.0)


@pytest.mark.parametrize("cls", CLASSIFIERS)
def test_classifier_predicts_a_label_after_learning(cls):
    rng = np.random.default_rng(0)
    m = _make(cls)
    x = {"a": 1.0, "b": 0.5, "c": 0.2}
    assert m.predict_one(x) is None
    for _ in range(40):
        row = {k: float(rng.normal()) for k in x}
        m.learn_one(row, int(row["a"] > 0))
    assert m.predict_one(x) in (0, 1)


def test_fm_naive_pairwise_form_before_and_after_learning():
    m = FMRegressor(optimizer=SGD(lr=0.05), k=2, seed=0)
    assert m._pairwise_naive({"a": 1.0}) == 0.0
    x = {"a": 1.0, "b": 2.0, "c": 0.5}
    for _ in range(5):
        m.learn_one(x, 1.0)
    assert np.isclose(m._pairwise_naive(x), m._y_hat(x))


@pytest.mark.parametrize("make", [
    lambda: FFMRegressor(fields={"a": "F1", "b": "F2"}, feature_keys=["a", "b", "c"], k=2, seed=0),
    lambda: FwFMRegressor(optimizer=SGD(lr=0.05), fields={"a": "F1", "b": "F2"},
                          feature_keys=["a", "b", "c"], k=2, seed=0),
])
def test_unknown_keys_and_keys_without_a_field_are_skipped(make):
    m = make()
    m.learn_one({"a": 1.0, "b": 1.0, "c": 1.0, "zzz": 1.0}, 1.0)
    assert m.n_missing == 0


def test_ffm_with_l2_updates_the_pairs():
    m = FFMRegressor(fields=FIELDS, k=2, lr=0.1, l2=0.1, seed=0)
    m.learn_one({"a": 1.0, "b": 2.0, "c": 0.5}, 1.0)
    assert m.n_seen_ == 1
