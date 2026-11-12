# Text into any primitive

`TextInput` gives documents to a primitive that already exists in the library, in the form that primitive takes. Nothing is copied: the text becomes numbers by the existing [text vectorizer](text_vectorizer.md), and which method of the primitive is called is decided by the same adapter as the [audio one](../audio/index.md), which `TextInput` extends.

```python
from dense_armor.core.signals.preprocessing.text_input import TextInput
```

## What the primitive receives

Each document is learned by a `TextVectorizer` and transformed. The primitive receives the flat dict of its `scores`: `{term: float}`. The text itself is not passed (some primitives refuse a string).

The adapter looks at the methods of the primitive:

| the primitive has | `feed(text, y)` does |
| --- | --- |
| `update` and `drift_detected` (a drift detector) | gives it one number, and reports whether it fired |
| `score_one` | scores the document, then learns it |
| `predict_one` and `learn_one` | predicts, then learns with the label `y` (`NeedsLabels` if there is none) |
| `transform_one` | learns, then transforms |
| only `learn_one` | learns |
| only `update_many` (a filter of samples) | raises `ValueError`: a document has no samples |

With a classifier or a regressor the prediction is made before the document is learned, so the answers of a stream are the prequential ones.

The one number given to a drift detector, or to any primitive that cannot take the dict, is the entry `key`. By default it is `"n_terms"`, the number of terms the document has; it can also be a term. A term the document does not have reads as `0.0` and is not added to the dict.

## A classifier and a drift detector

```python
from dense_armor.core.online.drift.adwin import ADWIN
from dense_armor.core.online.learn.online_classifiers import OnlineGaussianNB

pos = ["great fast reliable product", "love it works well", "very happy with the result"]
neg = ["terrible slow broken product", "hate it does not work", "very unhappy with the result"]
docs = [(t, 1) for t in pos] * 4 + [(t, 0) for t in neg] * 4

a = TextInput(OnlineGaussianNB())
answers = [a.feed(t, y) for t, y in docs]
print(a.mode, a.n)          # supervised 24

d = TextInput(ADWIN())
for t, _ in docs:
    d.feed(t)
print(d.mode, d.n, d.alarms)  # drift 24 0
```

This shows that the primitives run on text. It does not measure how well they learn: that depends on the primitive and the data.

## Checked

All the 40 classifiers, regressors, anomaly scorers and drift detectors of `core/online` that are built without arguments were fed 24 documents through `TextInput` without an error. The 28 that need arguments were not tried. The test is `test/preprocessing/test_text_input.py`.
