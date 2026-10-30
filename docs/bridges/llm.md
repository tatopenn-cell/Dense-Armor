# LLM bridge: how do I classify a text with a Dense-Armor classifier, and call it from an LLM tool?

A Dense-Armor classifier takes a dict of numbers. A text is not a dict of numbers. `LLMEmbeddingClassifier` closes the gap: you give it a function `embed` that turns a text into a list of numbers, and it passes those numbers, named `e0`, `e1`, and so on, to the classifier it wraps. It also reads and writes JSON strings, which is the form a tool-calling LLM loop uses.

The bridge has no extra package: `embed` is yours. The examples use the length of the text as the only number.

## 1. A text in, a label out

`learn_one(text, label)` and `predict_one(text)` take texts. The four texts below are long or short Italian and English words.

```python
from dense_armor.bridges.llm import LLMEmbeddingClassifier
from dense_armor.utility.learn.online_classifiers import OnlineGaussianNB

def embed(t):
    return [len(t)]

clf = LLMEmbeddingClassifier(embed, OnlineGaussianNB())
d = [("buongiorno", "long"), ("buonasera", "long"), ("hello", "short"), ("hi", "short")]
for _ in range(10):
    for t, y in d:
        clf.learn_one(t, y)
print(clf.predict_one("ciao"), clf.predict_one("arrivederci"))
```

```
short long
```

`ciao` has 4 characters and is closer to the short class; `arrivederci` has 11 and is closer to the long one.

## 2. JSON in, JSON out

`learn_json` takes `{"text": ..., "label": ...}`; `predict_json` and `predict_proba_json` take `{"text": ...}`. Each returns a JSON string, so an agent can call them as tools.

```python
import json
from dense_armor.bridges.llm import LLMEmbeddingClassifier
from dense_armor.utility.learn.online_classifiers import OnlineGaussianNB

def embed(t):
    return [len(t)]

c = LLMEmbeddingClassifier(embed, OnlineGaussianNB())
d = [("buongiorno", "long"), ("buonasera", "long"), ("hello", "short"), ("hi", "short")]
for _ in range(10):
    for t, y in d:
        c.learn_json(json.dumps({"text": t, "label": y}))
print(c.predict_json('{"text": "ciao"}'))
print({k: round(v, 3) for k, v in json.loads(c.predict_proba_json('{"text": "ciao"}'))["proba"].items()})
```

```
{"prediction": "short"}
{'long': 0.0, 'short': 1.0}
```

A `label` that is missing or `null` raises `ValueError`, because a label-less sample cannot teach the classifier.

## 3. A description for the agent

`describe()` returns a JSON-serialisable description of the object, with the input and output schema, for the tool list of an LLM.

```python
from dense_armor.bridges.llm import LLMEmbeddingClassifier
from dense_armor.utility.learn.online_classifiers import OnlineGaussianNB

def embed(t):
    return [len(t)]

c = LLMEmbeddingClassifier(embed, OnlineGaussianNB())
print(sorted(c.describe()))
```

```
['embedding', 'input_schema', 'methods', 'module', 'name', 'output_schema', 'parameters']
```

## Details

### References

None: the bridge is an adapter, and the three blocks above are the check of its behaviour.

### Choices stated explicitly

- **`embed` is supplied by the caller.** The bridge does not call any model, and has no dependency beyond NumPy. The `llm` extra in `pyproject.toml` is empty.
- **Feature names.** The vector returned by `embed` becomes the dict `{"e0": ..., "e1": ...}`. `embed` must return a one-dimensional sequence of numbers: a string, `None` or an array with another shape raises `TypeError` or `ValueError`.
- **Labels in JSON.** A string label stays a string; any other label is its JSON text (`True` becomes `"true"`, `1` becomes `"1"`) in the keys of `predict_proba_json`.

### Measured cost

Not measured.
