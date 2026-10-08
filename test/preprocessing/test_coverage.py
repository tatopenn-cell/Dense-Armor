"""Edge paths of the preprocessing package: unsupported values, missing keys, thin state."""
import numpy as np
import pytest

from dense_armor.utility.preprocessing.joints import JointDerivatives, JointPower
from dense_armor.utility.preprocessing.scale import (
    EWScaler,
    MinMaxScaler,
    RobustScaler,
    StandardScaler,
    _to_vec,
)
from dense_armor.utility.preprocessing.select import (
    SelectKBest,
    VarianceThreshold,
    _CorrStats,
)
from dense_armor.utility.preprocessing.text import TFIDF, Tokenizer

SCALERS = [StandardScaler, EWScaler, RobustScaler, MinMaxScaler]


def test_to_vec_branches():
    assert _to_vec(True) == (None, True)
    assert _to_vec(np.float64(2.0))[0].tolist() == [2.0]
    assert _to_vec(np.array([1, 2]))[0].tolist() == [1.0, 2.0]
    assert _to_vec(np.array(["a"], dtype=object)) == (None, True)
    assert _to_vec(["a", "b"]) == (None, True)
    assert _to_vec({"a": 1}) == (None, True)


@pytest.mark.parametrize("cls", SCALERS, ids=lambda c: c.__name__)
def test_scaler_edge_inputs(cls):
    sc = cls()
    assert sc.transform_one({"x": 1.0}) == {"x": 1.0}
    sc.learn_one({"x": 1.0, "s": "text", "q": [0.0, 0.0]})
    sc.learn_one({"q": [1.0, 2.0, 3.0]})
    sc.learn_one({"x": 3.0, "q": [2.0, 4.0]})
    sc.learn_one({"x": 5.0, "q": [4.0, 8.0]})
    out = sc.transform_one({"x": "bad", "q": [1.0, 2.0, 3.0], "new": 1.0})
    assert out["x"] == "bad"
    assert out["q"] == [1.0, 2.0, 3.0]
    assert out["new"] == 1.0
    flat = cls()
    for _ in range(3):
        flat.learn_one({"q": [1.0, 1.0]})
    assert flat.transform_one({"q": [1.0, 1.0]})["q"] == [1.0, 1.0]
    assert flat.transform_one({})== {}


def test_joint_derivatives_edge_inputs():
    jd = JointDerivatives(order=2)
    jd.learn_one({"q": ["a"]}, t=0.0)
    jd.learn_one({"q": 1.0}, t=0.5)
    assert jd.n_missing_ == 1
    assert "qd" not in jd.transform_one({})
    jd.learn_one({"q": 3.0}, t=1.0)
    assert jd.transform_one({})["qd"] == [4.0]


def test_joint_power_missing_and_mismatch():
    jp = JointPower()
    assert "power" not in jp.transform_one({"tau": [1.0]})
    assert "power" not in jp.transform_one({"tau": [1.0], "qd": [1.0, 2.0]})
    assert jp.n_missing_ == 2


def test_text_inputs_and_precomputed_counts():
    assert Tokenizer().transform_one("Hi there")["tokens"] == ["hi", "there"]
    assert Tokenizer().transform_one({"text": 42})["tokens"] == ["42"]
    tf = TFIDF()
    tf.learn_one({"counts": {"a": 1, "b": 1}})
    tf.learn_one({"counts": {"a": 2}})
    out = tf.transform_one({"counts": {"b": 1}})
    assert out["scores"]["b"] == pytest.approx(np.log(2.0))


def test_select_thin_state_and_passthrough():
    assert _CorrStats().corr == 0.0
    vt, sk = VarianceThreshold(), SelectKBest(k=1)
    assert vt.transform_one({"a": 1.0}) == {"a": 1.0}
    assert sk.transform_one({"a": 1.0}) == {"a": 1.0}
    vt.learn_one({"a": 1.0})
    vt.learn_one({"a": 2.0})
    assert vt.transform_one({"a": 1.0, "z": 5.0}) == {"a": 1.0, "z": 5.0}
