# scikit-learn bridge

scikit-learn fits a model on the whole table at once; Dense-Armor learns one sample at a time. A bridge lets each side call the other in the interface it expects. A Dense-Armor classifier becomes a scikit-learn estimator, and a scikit-learn estimator becomes a Dense-Armor estimator, without rewriting the algorithm.

## 1. A Dense-Armor classifier as a scikit-learn estimator

`SklearnClassifier` wraps any Dense-Armor classifier. Its `fit` learns the rows in order with `learn_one`, and its `predict` / `predict_proba` call `predict_one` / `predict_proba_one` on each row.

```python
import numpy as np
from dense_armor.bridges.sklearn import SklearnClassifier
from dense_armor.utility.learn.online_classifiers import OnlineGaussianNB

X = np.array([[0.0], [1.0], [2.0], [3.0], [4.0], [5.0]])
y = np.array([0, 0, 0, 1, 1, 1])
m = SklearnClassifier(OnlineGaussianNB).fit(X, y)
print(m.predict(np.array([[0.5], [4.5]])).tolist())
```

```
[0, 1]
```

`classes_` and `n_features_in_` are set by `fit`, as a scikit-learn user expects. The wrapper passes every input check the scikit-learn library asks of an estimator (sparse data, complex data, NaN, 1D input, empty input, `y` with one class, `y` with a continuous target for a classifier), and its error messages carry the text scikit-learn reads.

The same pattern covers a Dense-Armor regressor with `SklearnRegressor` and a Dense-Armor transformer with `SklearnTransformer`.

```python
import numpy as np
from dense_armor.bridges.sklearn import SklearnRegressor
from dense_armor.utility.linear.glm import LinearRegression
from dense_armor.utility.optim import SGD

X = np.array([[0.0], [1.0], [2.0], [3.0], [4.0], [5.0]])
y = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
m = SklearnRegressor(LinearRegression(optimizer=SGD(lr=0.01))).fit(X, y)
print(m.predict(np.array([[0.5], [4.5]])).round(3).tolist())
```

```
[0.58, 3.483]
```

## 2. A scikit-learn estimator inside Dense-Armor

`from_sklearn(est)` returns a Dense-Armor `Estimator`. If the wrapped estimator has `partial_fit`, `learn_one` calls it on the current row. If it has no `partial_fit` (a batch estimator), `refit_every` is required and the rows are collected and refit every `refit_every` rows.

```python
import numpy as np
from sklearn.linear_model import SGDClassifier
from dense_armor.bridges.sklearn import from_sklearn

rng = np.random.default_rng(0)
X = rng.normal(size=(40, 2))
y = (X[:, 0] + X[:, 1] > 0.0).astype(int)
wrapped = from_sklearn(SGDClassifier(loss="log_loss", random_state=0), classes=[0, 1])
for row, yi in zip(X, y):
    wrapped.learn_one({"0": float(row[0]), "1": float(row[1])}, int(yi))
print(wrapped.predict_one({"0": 0.0, "1": 0.0}))
```

```
1
```

The first call to `partial_fit` of a scikit-learn classifier must carry the class list. Passing `classes=[0, 1]` to `from_sklearn` uses it from the first row. Leaving it out buffers the rows until two distinct classes have been seen, then calls `partial_fit` once with those classes and switches to one-row calls.

## 3. The agreement test

For three Dense-Armor classes (a classifier, a regressor, a transformer) the wrapper through `fit` / `predict` returns the same prediction as the same object driven by `learn_one` / `predict_one`, exactly. In the other direction, `from_sklearn` on a scikit-learn partial-fit classifier returns the same prediction as `partial_fit` called directly on the same rows.

## 4. What passes and what does not

All the scikit-learn `check_estimator` checks pass for the three wrappers: 154 checks (55 + 52 + 47) on scikit-learn 1.6.1, no check skipped, no `expected_failed_checks` entry. The list of checks that cannot hold for an online learner is empty.

The import of the bridge without scikit-learn installed raises a `ModuleNotFoundError` that names the extra:

```
ModuleNotFoundError: dense_armor.bridges.sklearn needs scikit-learn: pip install dense-armor[sklearn]
```

## Details

### References

- scikit-learn 1.6.1, *Developing scikit-learn estimators*: <https://scikit-learn.org/1.6/developers/develop.html>.
- scikit-learn 1.6.1, `check_estimator`: <https://scikit-learn.org/1.6/modules/generated/sklearn.utils.estimator_checks.check_estimator.html>.
- scikit-learn 1.6.1, `parametrize_with_checks`: <https://scikit-learn.org/1.6/modules/generated/sklearn.utils.estimator_checks.parametrize_with_checks.html>.

### Choices stated explicitly

- **The wrapper uses the mixin order scikit-learn requires**: `ClassifierMixin` first, then the Dense-Armor wrapper, then `BaseEstimator`. The reverse order (Dense-Armor first) is what `check_mixin_order` rejects.
- **`predict` returns the class with the largest probability, taken from `classes_`**, never the value of a member or `None`. On this stream a member that has not seen the class returns `{}` from `predict_proba_one`, and the wrapper still returns a class from `classes_`.
- **`from_sklearn` treats `partial_fit` as an online method.** When the estimator has `partial_fit`, `learn_one` calls it on the current row, and `refit_every` must be `None`. When the estimator is batch (no `partial_fit`), `refit_every` must be given, because the alternative is a `fit` over a growing table at every row.
- **The class list of a `partial_fit` classifier**: passing `classes` to `from_sklearn` uses it from the first row; leaving it out buffers the rows until two distinct classes have been seen. The buffer is the only correct option when the row stream starts with a single class, because `SGDClassifier.partial_fit` with one class and no `classes` raises.
- **The optional dependency is installed as an extra**: `pip install dense-armor[sklearn]`. The bridge module imports scikit-learn inside a `try/except` and raises a `ModuleNotFoundError` that names the extra, as the river bridge does. `import dense_armor` works without scikit-learn.

### Measured cost

The bridge adds one `partial_fit` call per row on the wrapped scikit-learn estimator, and one `predict_one` per row on the wrapper. The Dense-Armor `utility/` classifiers it wraps are O(window) or O(1) per `learn_one`. A benchmark is not part of this page: the numbers above are correctness only.

### Development diary

The first version of the wrapper inherited from the Dense-Armor classes first (`_DenseWrapper` as a parent), and `check_mixin_order` rejected it. The order is now `ClassifierMixin, _DenseWrapper, BaseEstimator` per wrapper.

The first version of `from_sklearn` used a `fit` over the growing table for every row, which raised on a stream whose first row carries one class: `LogisticRegression` with the default solver `lbfgs` has no `partial_fit`, and `SGDClassifier.partial_fit` needs the class list on the first call. The wrapper now chooses `partial_fit` when the estimator has it, and buffers the first rows otherwise.
