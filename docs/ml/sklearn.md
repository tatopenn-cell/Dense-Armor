# scikit-learn tools on the classes of ml/: how do I put a Dense-Armor classifier in a Pipeline, a cross-validation and a grid search?

The classes of `dense_armor.ml` read the whole table at once: `fit(X, y)`, then `predict(X)` or `transform(X)`. A scikit-learn `Pipeline`, `cross_val_score` or `GridSearchCV` asks for three things more: the tags that say what kind of estimator the object is, a `score` method, and a `fit_transform(X, y)` that accepts the target. `MLBase`, the base of all these classes, now has them. No algorithm and no default changes, and `import dense_armor.ml` works without scikit-learn: the library is imported only when the tags are asked for.

The three steps below use a synthetic table of 200 rows and 2 columns, the first around 1000 and the second around 5.

## 1. A scaler and a classifier in a Pipeline

`make_pipeline` fits the scaler, passes its output to the classifier, and calls `score` of the last step.

```python
import numpy as np
from sklearn.pipeline import make_pipeline
from dense_armor.ml.linear import LogisticRegression
from dense_armor.ml.preprocessing import StandardScaler

rng = np.random.default_rng(0)
X = rng.normal([1e3, 5.0], [2.0, 1.0], size=(200, 2))
y = (X[:, 1] > 5.0).astype(int)
pipe = make_pipeline(StandardScaler(), LogisticRegression()).fit(X, y)
print(pipe.score(X, y), pipe.predict(X[:5]).tolist())
```

```
1.0 [0, 1, 1, 1, 0]
```

The pipeline calls `StandardScaler.fit_transform(X, y)`, with the target, as it does for every transformer; the scaler does not use `y`. `score` of the classifier is the accuracy.

## 2. A cross-validation of a regressor

`cross_val_score` splits the table in 3 parts, fits a clone of the estimator on 2 and scores it on the third. For a regressor the score is the coefficient of determination.

```python
import numpy as np
from sklearn.model_selection import cross_val_score
from dense_armor.ml.linear import LinearRegression

rng = np.random.default_rng(0)
X = rng.normal([1e3, 5.0], [2.0, 1.0], size=(200, 2))
y = X @ np.array([0.5, 2.0]) + rng.normal(0, 0.1, 200)
s = cross_val_score(LinearRegression(), X, y, cv=3)
print(np.round(s, 4).tolist())
```

```
[0.9977, 0.9981, 0.9982]
```

The three scores are the same as `r2_score` of scikit-learn on the same predictions, which a test checks.

## 3. A grid search on a classifier

`GridSearchCV` clones the classifier for every value of `n_neighbors`, scores each by cross-validation and keeps the best.

```python
import numpy as np
from sklearn.model_selection import GridSearchCV
from dense_armor.ml.neighbors import KNNClassifier

rng = np.random.default_rng(0)
X = rng.normal([1e3, 5.0], [2.0, 1.0], size=(200, 2))
y = (X[:, 1] > 5.0).astype(int)
gs = GridSearchCV(KNNClassifier(), {"n_neighbors": [1, 3, 5]}, cv=3).fit(X, y)
print(gs.best_params_, round(gs.best_score_, 4))
```

```
{'n_neighbors': 1} 0.9649
```

Because the two columns have a different scale, the nearest neighbour of a row is decided mostly by the first column, which is why `n_neighbors=1` wins here: the page shows the call, not a good model.

## Details

### References

- scikit-learn documentation: the `__sklearn_tags__` method (scikit-learn 1.6 and later) and the `r2_score` convention for a constant target (1.0 for a perfect prediction, 0.0 otherwise), read from the docstring of the installed version. The three blocks above and `test/ml/test_sklearn_compat.py` are the check that the classes work in the tools, on scikit-learn 1.6.1 and 1.9.0.

### Choices stated explicitly

- **One mixin on `MLBase`.** `dense_armor/ml/_sklearn_compat.py` gives `__sklearn_tags__`, `score` and `fit_transform` to the 66 public classes of `dense_armor.ml`; `score` and `fit_transform` exist only where they make sense (`hasattr` is false otherwise).
- **The kind of a class** comes from its name and its methods: `predict_proba` or a name ending in `Classifier` is a classifier; a name ending in `Regressor` or `Regression` is a regressor; `transform` is a transformer; anything else has no kind.
- **`score`** is the accuracy for a classifier and the coefficient of determination for a regressor, with `sample_weight`.
- **`fit_transform(X, y=None, **fit_params)`** replaces `fit_transform(X)` on the scalers and on `FeatureHasher`, with the new parameters last; a transformer without its own gets `fit(X, y).transform(X)`.

### What the audit found

`tools/audit_ml_sklearn.py` lists the 66 public classes of `dense_armor.ml`: 29 have `predict`, 18 `predict_proba`, 9 `transform`. With the mixin 28 have `score` and 10 have `fit_transform`. `clone` works on 51; the other 15 need constructor arguments (ensembles, multi-class and multi-output wrappers, `PCA`, `RandomProjection`, `ShrinkageCovariance`). In a `Pipeline` of two steps 6 classes work and 1 fails: `TextVectorizer`, which takes texts.

### Declared limits

The classes do not follow the estimator contract of scikit-learn: the three tools above work, but the checks of `sklearn.utils.estimator_checks.check_estimator` fail for `LogisticRegression` (16 distinct checks), `LinearRegression` (15) and `StandardScaler` (11), with the same sets on scikit-learn 1.6.1 and 1.9.0. The checks are listed one by one, with the reason, in `test/ml/test_sklearn_compat.py`, and the test fails when a check that is not listed fails or a listed one passes. The reasons are: `n_features_in_` is not set; predict before fit does not raise `NotFittedError`; complex, empty, 1-D, one-sample and sparse input are not rejected with the messages of scikit-learn; non-finite values are dropped by the class and not rejected; a 2-D `y` and string labels are not supported; and `transform` with another number of features is not rejected. The other 63 classes were not run through `check_estimator`.

### Measured cost

Not measured.
