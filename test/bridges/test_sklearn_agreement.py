"""Item C: the wrapper and the direct online loop give the same prediction.

For three Dense-Armor classes (one classifier, one regressor, one
transformer) on the same synthetic data, the prediction of the wrapper
through ``fit`` / ``predict`` equals the prediction of the same object
driven by ``learn_one`` / ``predict_one``, exactly. The other direction
holds for a scikit-learn partial-fit classifier: ``from_sklearn`` on a
scikit-learn classifier gives the same prediction as calling
``partial_fit`` directly on the same rows.
"""

import numpy as np
import pytest

pytest.importorskip("sklearn")

from sklearn.linear_model import SGDClassifier

from dense_armor.bridges.sklearn import (
    SklearnClassifier,
    SklearnRegressor,
    SklearnTransformer,
    from_sklearn,
)
from dense_armor.utility.learn.online_classifiers import OnlineGaussianNB
from dense_armor.utility.linear.glm import LinearRegression
from dense_armor.utility.optim import SGD
from dense_armor.utility.preprocessing.scale import MinMaxScaler


def _data(seed=0, n=40, d=3):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    y_cls = (X[:, 0] + X[:, 1] > 0.0).astype(int)
    y_reg = X[:, 0] - 0.5 * X[:, 1] + 0.1 * X[:, 2]
    return X, y_cls, y_reg


def test_classifier_agreement():
    X, y, _ = _data()
    wrapper = SklearnClassifier(OnlineGaussianNB).fit(X, y)
    direct = OnlineGaussianNB()
    for row, yi in zip(X, y):
        direct.learn_one({i: float(v) for i, v in enumerate(row)}, int(yi))
    direct_pred = np.asarray(
        [direct.predict_one({i: float(v) for i, v in enumerate(row)}) for row in X]
    )
    assert wrapper.predict(X).tolist() == direct_pred.tolist()


def test_regressor_agreement():
    X, _, y = _data()
    wrapper = SklearnRegressor(LinearRegression(optimizer=SGD(lr=0.01))).fit(X, y)
    direct = LinearRegression(optimizer=SGD(lr=0.01))
    for row, yi in zip(X, y):
        direct.learn_one({i: float(v) for i, v in enumerate(row)}, float(yi))
    direct_pred = np.asarray(
        [direct.predict_one({i: float(v) for i, v in enumerate(row)}) for row in X]
    )
    np.testing.assert_allclose(wrapper.predict(X), direct_pred, rtol=1e-12, atol=1e-12)


def test_transformer_agreement():
    X, _, _ = _data()
    wrapper = SklearnTransformer(MinMaxScaler).fit(X)
    direct = MinMaxScaler()
    for row in X:
        direct.learn_one({i: float(v) for i, v in enumerate(row)})
    keys = sorted(
        direct.transform_one({i: float(v) for i, v in enumerate(X[0])}).keys()
    )
    direct_out = np.asarray(
        [
            [
                float(direct.transform_one({i: float(v) for i, v in enumerate(row)})[k])
                for k in keys
            ]
            for row in X
        ]
    )
    np.testing.assert_allclose(wrapper.transform(X), direct_out, rtol=1e-12, atol=1e-12)


def test_from_sklearn_partial_fit_agreement():
    X, y, _ = _data()
    est = SGDClassifier(loss="log_loss", random_state=0)
    wrapped = from_sklearn(est, classes=[0, 1])
    direct = SGDClassifier(loss="log_loss", random_state=0)
    classes = np.array([0, 1])
    first = True
    for row, yi in zip(X, y):
        kwargs = {"classes": classes} if first else {}
        direct.partial_fit(row.reshape(1, -1), np.array([int(yi)]), **kwargs)
        wrapped.learn_one({i: float(v) for i, v in enumerate(row)}, int(yi))
        first = False
    got = [wrapped.predict_one({i: float(v) for i, v in enumerate(row)}) for row in X]
    assert got == direct.predict(X).tolist()
