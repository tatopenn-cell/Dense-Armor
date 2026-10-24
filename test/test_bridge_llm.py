"""Tests for the LLM bridge, with a stub embed. No network."""
import json

import pytest

from dense_armor.bridges.llm import LLMEmbeddingClassifier
from dense_armor.utility.learn.online_classifiers import OnlineGaussianNB


def _stub_embed(text: str):
    """A cheap, deterministic 3-dim embedding for tests."""
    return [len(text), sum(map(ord, text)) % 7, 1.0]


def test_learn_and_predict_on_two_classes():
    clf = LLMEmbeddingClassifier(_stub_embed, OnlineGaussianNB())
    for _ in range(50):
        clf.learn_one("buongiorno", "it")
        clf.learn_one("hello", "en")
    assert clf.predict_one("ciao") in ("it", "en")
    proba = clf.predict_proba_one("ciao")
    assert set(proba) <= {"it", "en"}
    assert abs(sum(proba.values()) - 1.0) < 1e-9


def test_predict_before_learning_is_none():
    clf = LLMEmbeddingClassifier(_stub_embed, OnlineGaussianNB())
    assert clf.predict_one("anything") is None
    assert clf.predict_proba_one("anything") == {}


def test_describe_is_json_serialisable_and_names_embed():
    clf = LLMEmbeddingClassifier(_stub_embed, OnlineGaussianNB())
    d = clf.describe()
    json.dumps(d)
    assert d["embedding"] == "_stub_embed"
    assert "classifier" in d["parameters"]


def test_learn_json_and_predict_json_roundtrip():
    clf = LLMEmbeddingClassifier(_stub_embed, OnlineGaussianNB())
    for _ in range(30):
        clf.learn_json(json.dumps({"text": "buongiorno", "label": "it"}))
        clf.learn_json(json.dumps({"text": "hello", "label": "en"}))
    out = json.loads(clf.predict_json(json.dumps({"text": "ciao"})))
    assert out["prediction"] in ("it", "en")
    proba_out = json.loads(clf.predict_proba_json(json.dumps({"text": "ciao"})))
    for k in proba_out["proba"]:
        assert isinstance(k, str)


def test_works_with_an_arbitrary_wrapped_estimator():
    from dense_armor.utility.learn.metric_learning import OASIS, MetricKNNClassifier

    clf = LLMEmbeddingClassifier(
        _stub_embed,
        MetricKNNClassifier(OASIS(C=1e6), n_neighbors=1),
    )
    clf.learn_one("aaa", "A")
    clf.learn_one("zzz", "B")
    assert clf.predict_one("aa") in ("A", "B")


def test_embed_is_called_once_per_call():
    calls = {"n": 0}

    def counting_embed(text):
        calls["n"] += 1
        return [float(len(text))]

    clf = LLMEmbeddingClassifier(counting_embed, OnlineGaussianNB())
    clf.learn_one("x", 0)
    assert calls["n"] == 1
    clf.predict_one("x")
    assert calls["n"] == 2


@pytest.mark.parametrize("bad", ["", "{not json"])
def test_learn_json_rejects_bad_input(bad):
    clf = LLMEmbeddingClassifier(_stub_embed, OnlineGaussianNB())
    with pytest.raises((json.JSONDecodeError, KeyError)):
        clf.learn_json(bad)


@pytest.mark.parametrize("payload", [{"text": "ciao"}, {"text": "ciao", "label": None}])
def test_learn_json_rejects_missing_label(payload):
    clf = LLMEmbeddingClassifier(_stub_embed, OnlineGaussianNB())
    with pytest.raises(ValueError, match="label"):
        clf.learn_json(json.dumps(payload))
    assert clf.predict_one("ciao") is None


@pytest.mark.parametrize("out, exc", [(None, TypeError), ("123", TypeError),
                                      ([[1.0, 2.0], [3.0, 4.0]], ValueError)])
def test_embed_output_is_validated(out, exc):
    clf = LLMEmbeddingClassifier(lambda text: out, OnlineGaussianNB())
    with pytest.raises(exc, match="embed must return"):
        clf.learn_one("x", 0)


def test_predict_proba_json_keys_match_predict_json():
    clf = LLMEmbeddingClassifier(_stub_embed, OnlineGaussianNB())
    for _ in range(5):
        clf.learn_one("aa", True)
        clf.learn_one("bbbbbb", False)
    pred = json.loads(clf.predict_json(json.dumps({"text": "aa"})))["prediction"]
    proba = json.loads(clf.predict_proba_json(json.dumps({"text": "aa"})))["proba"]
    assert json.dumps(pred) in proba
    assert set(proba) == {"true", "false"}


def test_describe_names_a_callable_instance_by_its_class():
    class Embedder:
        def __call__(self, text):
            return [1.0]

    d = LLMEmbeddingClassifier(Embedder(), OnlineGaussianNB()).describe()
    assert d["embedding"] == "Embedder"
