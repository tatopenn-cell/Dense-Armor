# river bridge: how do I use a Dense-Armor classifier in a river loop, and a river model after a Dense-Armor scaler?

river and Dense-Armor both learn one sample at a time, and both name the same methods: `learn_one`, `predict_one`, `predict_proba_one`. Because the interfaces match, the bridge does not convert anything: `to_river(est)` and `from_river(est)` return the object they receive. The two functions exist to say, in the code, which side the object is used from, and to raise a clear error when river is not installed.

The two steps below use a synthetic stream of 60 samples: 30 of class 0 with `a` near 0 to 3, and 30 of class 1 with `a` near 5 to 8.

## 1. A Dense-Armor classifier inside a river loop

`to_river` hands a Dense-Armor classifier to river code. The loop is the prequential one of river: predict first, then learn.

```python
from river import metrics
from dense_armor.bridges.river import to_river
from dense_armor.utility.learn.online_classifiers import OnlineGaussianNB

s = [({"a": 0.1 * i, "b": 1.0}, 0) for i in range(30)]
s += [({"a": 5 + 0.1 * i, "b": 1.0}, 1) for i in range(30)]
m = to_river(OnlineGaussianNB())
acc = metrics.Accuracy()
for x, y in s:
    p = m.predict_one(x)
    if p is not None:
        acc.update(y, p)
    m.learn_one(x, y)
print(round(acc.get(), 4), m.predict_one({"a": 5.5, "b": 1.0}), m.predict_one({"a": 0.2, "b": 1.0}))
```

```
0.9661 1 0
```

The first sample has no prediction (the classifier has seen nothing), so the accuracy counts the other 59. After the stream, a point at `a = 5.5` is class 1 and a point at `a = 0.2` is class 0.

## 2. A river classifier after a Dense-Armor scaler

`from_river` hands a river model to Dense-Armor code. The `|` operator of Dense-Armor chains a scaler and the river classifier in one pipeline.

```python
from river import linear_model
from dense_armor.bridges.river import from_river
from dense_armor.utility.preprocessing.scale import StandardScaler

s = [({"a": 0.1 * i, "b": 1.0}, False) for i in range(30)]
s += [({"a": 5 + 0.1 * i, "b": 1.0}, True) for i in range(30)]
pipe = StandardScaler() | from_river(linear_model.LogisticRegression())
for x, y in s:
    pipe.learn_one(x, y)
print(pipe.predict_one({"a": 5.5, "b": 1.0}), pipe.predict_one({"a": 0.2, "b": 1.0}))
```

```
True False
```

The river logistic regression is binary and takes `True` and `False` as labels. The scaler of Dense-Armor standardises each feature before it reaches the river model.

## Details

### References

- river documentation, the `learn_one` and `predict_one` methods of its classifiers. Not read for this page: the two blocks above are the check that the interfaces match, on the installed version.

### Choices stated explicitly

- **The bridge is the identity.** No wrapper class is written, because the objects already have the other side's methods. A wrapper would add a layer where the two interfaces are equal.
- **The extra.** `pip install dense-armor[river]` installs `river>=0.26` on Python 3.11 or newer, as `pyproject.toml` declares. Importing `dense_armor.bridges.river` without river raises a `ModuleNotFoundError` that names the extra.

### Measured cost

Not measured: the functions return their argument.

### Not covered

River estimators that need a method Dense-Armor classifiers do not have, and Dense-Armor classifiers that need a method river does not call. The two blocks above run on river 0.26.1.
