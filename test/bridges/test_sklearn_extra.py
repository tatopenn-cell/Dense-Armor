"""Extra tests for the uncovered branches of the scikit-learn bridge.

These are the input-validation paths, the partial_fit contract, the
agreement with a batch scikit-learn estimator, and the error paths of
``_as_array``, ``_y_classifier``, ``_y_regressor`` and ``from_sklearn``.
"""

import numpy as np
import pytest

pytest.importorskip("sklearn")

from sklearn.linear_model import Ridge, SGDClassifier
from sklearn.tree import DecisionTreeClassifier

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


# ---- _as_array error paths ----


def test_classifier_rejects_sparse():
    import scipy.sparse as sp

    X = sp.csr_matrix(np.array([[0.0], [1.0], [2.0], [3.0]]))
    y = np.array([0, 0, 1, 1])
    with pytest.raises(ValueError, match="Sparse"):
        SklearnClassifier(OnlineGaussianNB).fit(X, y)


def test_classifier_rejects_complex():
    X = np.array([[0.0 + 0.0j], [1.0 + 0.0j], [2.0 + 0.0j], [3.0 + 0.0j]])
    y = np.array([0, 0, 1, 1])
    with pytest.raises(ValueError):
        SklearnClassifier(OnlineGaussianNB).fit(X, y)


def test_classifier_rejects_1d():
    with pytest.raises(ValueError, match="Reshape"):
        SklearnClassifier(OnlineGaussianNB).fit(
            np.array([0.0, 1.0, 2.0]), np.array([0, 0, 1])
        )


def test_classifier_rejects_nan():
    X = np.array([[0.0], [np.nan], [2.0], [3.0]])
    y = np.array([0, 0, 1, 1])
    with pytest.raises(ValueError):
        SklearnClassifier(OnlineGaussianNB).fit(X, y)


def test_classifier_rejects_empty():
    with pytest.raises(ValueError):
        SklearnClassifier(OnlineGaussianNB).fit(
            np.array([]).reshape(0, 1), np.array([])
        )


# ---- _y_classifier error paths ----


def test_classifier_rejects_y_none():
    with pytest.raises(ValueError, match="requires y"):
        SklearnClassifier(OnlineGaussianNB).fit(np.array([[0.0], [1.0]]), None)


def test_classifier_one_class_fits_and_predicts():
    """A single class is allowed: classes_ has one element and predict returns it."""
    X = np.array([[0.0], [1.0], [2.0]])
    y = np.array([0, 0, 0])
    m = SklearnClassifier(OnlineGaussianNB).fit(X, y)
    assert m.classes_.tolist() == [0]
    assert m.predict(np.array([[1.5]])).tolist() == [0]


# ---- _y_regressor error paths ----


def test_regressor_rejects_y_none():
    with pytest.raises(ValueError, match="requires y"):
        SklearnRegressor(LinearRegression(optimizer=SGD(lr=0.01))).fit(
            np.array([[0.0], [1.0]]), None
        )


def test_regressor_rejects_y_inf():
    X = np.array([[0.0], [1.0]])
    y = np.array([0.0, np.inf])
    with pytest.raises(ValueError):
        SklearnRegressor(LinearRegression(optimizer=SGD(lr=0.01))).fit(X, y)


# ---- n_features_in_ mismatch ----


def test_classifier_rejects_wrong_n_features():
    m = SklearnClassifier(OnlineGaussianNB).fit(
        np.array([[0.0], [1.0], [2.0], [3.0]]), np.array([0, 0, 1, 1])
    )
    with pytest.raises(ValueError, match="features"):
        m.predict(np.array([[0.0, 1.0]]))


# ---- partial_fit classes contract ----


def test_partial_fit_classes_change_raises():
    m = SklearnClassifier(OnlineGaussianNB)
    m.partial_fit(np.array([[0.0], [1.0]]), np.array([0, 1]), classes=[0, 1])
    with pytest.raises(ValueError, match="classes changed"):
        m.partial_fit(np.array([[2.0]]), np.array([1]), classes=[0, 1, 2])


def test_partial_fit_no_classes_on_second_call_ok():
    m = SklearnClassifier(OnlineGaussianNB)
    m.partial_fit(np.array([[0.0], [1.0]]), np.array([0, 1]), classes=[0, 1])
    m.partial_fit(np.array([[2.0]]), np.array([1]))
    assert m.classes_.tolist() == [0, 1]


# ---- _fresh variants ----


def test_fresh_from_instance():
    m = SklearnClassifier(OnlineGaussianNB(alpha=0.5))
    m.fit(np.array([[0.0], [1.0], [2.0], [3.0]]), np.array([0, 0, 1, 1]))
    assert m.model_.alpha == 0.5


# ---- from_sklearn error paths ----


def test_from_sklearn_partial_with_refit_every_raises():
    with pytest.raises(ValueError, match="partial_fit"):
        from_sklearn(SGDClassifier(loss="log_loss"), refit_every=5)


def test_from_sklearn_batch_without_refit_every_raises():
    with pytest.raises(ValueError, match="refit_every"):
        from_sklearn(Ridge())


def test_from_sklearn_buffer_two_classes():
    w = from_sklearn(SGDClassifier(loss="log_loss", random_state=0))
    w.learn_one({"a": 0.0}, 0)
    assert not w._partial_ready
    w.learn_one({"a": 1.0}, 1)
    assert w._partial_ready


def test_from_sklearn_unknown_label():
    w = from_sklearn(SGDClassifier(loss="log_loss", random_state=0), classes=[0, 1])
    w.learn_one({"a": 0.0}, 0)
    with pytest.raises(ValueError, match="not in the class list"):
        w.learn_one({"a": 1.0}, 2)


# ---- _DenseArmorAdapter branches ----


def test_from_sklearn_batch_predict_before_learn_refits():
    w = from_sklearn(Ridge(), refit_every=5)
    w.learn_one({"a": 0.0}, 0.0)
    pred = w.predict_one({"a": 1.0})
    assert np.isfinite(pred)


def test_from_sklearn_predict_proba_one():
    w = from_sklearn(SGDClassifier(loss="log_loss", random_state=0), classes=[0, 1])
    for i in range(10):
        w.learn_one({"a": float(i)}, i % 2)
    p = w.predict_proba_one({"a": 0.5})
    assert set(p.keys()) == {0, 1}
    assert abs(sum(p.values()) - 1.0) < 1e-9


# ---- transformer partial_fit ----


def test_transformer_partial_fit():
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    m = SklearnTransformer(MinMaxScaler).fit(X)
    m.partial_fit(np.array([[4.0]]))
    out = m.transform(np.array([[4.0]]))
    assert out.shape == (1, 1)


# ---- predict on unfitted ----


def test_classifier_predict_unfitted_raises():
    from sklearn.exceptions import NotFittedError

    m = SklearnClassifier(OnlineGaussianNB)
    with pytest.raises(NotFittedError):
        m.predict(np.array([[0.0]]))


def test_regressor_predict_unfitted_raises():
    from sklearn.exceptions import NotFittedError

    m = SklearnRegressor(LinearRegression(optimizer=SGD(lr=0.01)))
    with pytest.raises(NotFittedError):
        m.predict(np.array([[0.0]]))


# ---- key order independence ----


def test_predict_one_key_order_independent():
    w = from_sklearn(SGDClassifier(loss="log_loss", random_state=0), classes=[0, 1])
    for i in range(20):
        w.learn_one({"a": float(i), "b": float(-i)}, i % 2)
    # stesso x con chiavi in ordine diverso
    p1 = w.predict_one({"a": 0.5, "b": -0.5})
    p2 = w.predict_one({"b": -0.5, "a": 0.5})
    assert p1 == p2
