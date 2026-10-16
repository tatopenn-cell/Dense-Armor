"""LLM bridge: text features from an ``embed`` callable, JSON-in / JSON-out.

Wraps any Dense-Armor estimator so each input is text. ``embed(text) ->
sequence[float]`` turns a text into a numeric vector; the features are
named ``e0``, ``e1``, ... and passed to the wrapped estimator.
``describe()`` provides the JSON tool description for an LLM agent;
``learn_json`` / ``predict_json`` / ``predict_proba_json`` accept and
return JSON strings for a tool-calling loop.

The bridge has no extra dependency: ``embed`` is supplied by the caller.
``pip install dense-armor[llm]`` is declared in ``pyproject.toml`` for
forward compatibility, but the module imports without it.
"""
import json
from typing import Any, Callable, Dict, Optional, Sequence

from dense_armor.roles.estimator import Estimator


class LLMEmbeddingClassifier(Estimator):
    """Classifier whose input is text and whose features come from ``embed``.

    Args:
        embed: callable ``str -> sequence[float]``. Called once per
            ``learn_one`` and once per ``predict_one``.
        classifier: the wrapped classifier. Any object with ``learn_one``,
            ``predict_proba_one`` and ``predict_one`` is accepted.

    Examples:
        >>> from dense_armor.bridges.llm import LLMEmbeddingClassifier
        >>> from dense_armor.utility.learn.online_classifiers import OnlineGaussianNB
        >>> def fake_embed(text):
        ...     return [len(text), sum(map(ord, text)) % 7, 1.0]
        >>> clf = LLMEmbeddingClassifier(fake_embed, OnlineGaussianNB())
        >>> for _ in range(30):
        ...     _ = clf.learn_one("buongiorno", "it")
        ...     _ = clf.learn_one("hello", "en")
        >>> clf.predict_one("ciao") in ("it", "en")
        True
    """

    def __init__(self, embed: Callable[[str], Sequence[float]],
                 classifier: Estimator):
        self.embed = embed
        self.classifier = classifier

    def _features(self, text: str) -> Dict[str, float]:
        vec = self.embed(text)
        return {f"e{i}": float(v) for i, v in enumerate(vec)}

    def learn_one(self, text: str, y: Any, t: Optional[float] = None):
        x = self._features(text)
        try:
            self.classifier.learn_one(x, y, t=t)
        except TypeError:
            self.classifier.learn_one(x, y)
        return self

    def predict_proba_one(self, text: str,
                          t: Optional[float] = None) -> Dict[Any, float]:
        x = self._features(text)
        try:
            return self.classifier.predict_proba_one(x, t=t)
        except TypeError:
            return self.classifier.predict_proba_one(x)

    def predict_one(self, text: str, t: Optional[float] = None):
        p = self.predict_proba_one(text, t=t)
        if not p:
            return None
        return max(p, key=p.get)

    def describe(self) -> Dict[str, Any]:
        """JSON-serialisable description for an LLM agent."""
        inner = self.classifier.describe()
        return {
            "name": type(self).__name__,
            "module": type(self).__module__,
            "embedding": getattr(self.embed, "__name__", repr(self.embed)),
            "parameters": {"classifier": inner},
            "methods": ["learn_json", "predict_json", "predict_proba_json",
                        "describe"],
            "input_schema": {
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
            },
            "output_schema": {"type": "any"},
        }

    def learn_json(self, payload: str) -> str:
        """JSON-in / JSON-out: ``{"text": ..., "label": ...}``."""
        obj = json.loads(payload)
        self.learn_one(str(obj["text"]), obj.get("label"))
        return json.dumps({"ok": True})

    def predict_json(self, payload: str) -> str:
        obj = json.loads(payload)
        pred = self.predict_one(str(obj["text"]))
        return json.dumps({"prediction": pred})

    def predict_proba_json(self, payload: str) -> str:
        obj = json.loads(payload)
        p = self.predict_proba_one(str(obj["text"]))
        return json.dumps({"proba": {str(k): float(v) for k, v in p.items()}})
