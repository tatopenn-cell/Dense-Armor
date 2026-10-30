"""Smoke and agreement tests for the scikit-learn bridge.

Item A: the three wrappers (SklearnClassifier, SklearnRegressor,
SklearnTransformer) fit and predict. Item B: from_sklearn wraps a
scikit-learn estimator as a Dense-Armor Estimator. The import of
``dense_armor.bridges.sklearn`` without scikit-learn installed raises
a ModuleNotFoundError that names the extra, as the other bridges do.

The ``partial_fit`` branch of ``from_sklearn`` is tested with
``SGDClassifier``: ``LogisticRegression`` with the default solver
``lbfgs`` has no ``partial_fit`` in scikit-learn 1.6, so it exercises
the ``refit_every`` branch instead.
"""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("sklearn")

from sklearn.linear_model import Ridge, SGDClassifier

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


def test_classifier_fit_predict():
    X = np.array([[0.0], [1.0], [2.0], [3.0], [4.0], [5.0]])
    y = np.array([0, 0, 0, 1, 1, 1])
    m = SklearnClassifier(OnlineGaussianNB).fit(X, y)
    assert m.classes_.tolist() == [0, 1]
    assert m.n_features_in_ == 1
    assert m.predict(np.array([[0.5], [4.5]])).tolist() == [0, 1]
    p = m.predict_proba(np.array([[0.5]]))
    assert p.shape == (1, 2)
    assert abs(p.sum() - 1.0) < 1e-9


def test_regressor_fit_predict():
    X = np.array([[0.0], [1.0], [2.0], [3.0], [4.0], [5.0]])
    y = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    m = SklearnRegressor(LinearRegression(optimizer=SGD(lr=0.01))).fit(X, y)
    assert m.n_features_in_ == 1
    pred = m.predict(np.array([[0.5], [4.5]]))
    assert pred.shape == (2,)
    assert np.all(np.isfinite(pred))


def test_transformer_fit_transform():
    X = np.array([[0.0], [1.0], [2.0], [3.0], [4.0], [5.0]])
    m = SklearnTransformer(MinMaxScaler).fit(X)
    out = m.transform(np.array([[0.5], [4.5]]))
    assert out.shape == (2, 1)
    assert np.all(np.isfinite(out))
    assert m.get_feature_names_out().tolist() == ["0"]


def test_from_sklearn_partial_fit_estimator():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(50, 2))
    y = (X[:, 0] + X[:, 1] > 0.0).astype(int)

    est = SGDClassifier(loss="log_loss", random_state=0)
    wrapped = from_sklearn(est, classes=[0, 1])
    direct = SGDClassifier(loss="log_loss", random_state=0)
    classes = np.array([0, 1])
    first = True
    for row, yi in zip(X, y):
        kwargs = {"classes": classes} if first else {}
        direct.partial_fit(row.reshape(1, -1), np.array([int(yi)]), **kwargs)
        wrapped.learn_one({"0": float(row[0]), "1": float(row[1])}, int(yi))
        first = False
    got = [wrapped.predict_one({"0": float(r[0]), "1": float(r[1])}) for r in X]
    expected = direct.predict(X).tolist()
    assert got == expected


def test_from_sklearn_batch_estimator_requires_refit_every():
    with pytest.raises(ValueError, match="refit_every"):
        from_sklearn(Ridge())


def test_from_sklearn_batch_estimator():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 2))
    y = X[:, 0] - 0.5 * X[:, 1]
    wrapped = from_sklearn(Ridge(), refit_every=10)
    for row, yi in zip(X, y):
        wrapped.learn_one({"0": float(row[0]), "1": float(row[1])}, float(yi))
    pred = wrapped.predict_one({"0": float(X[0, 0]), "1": float(X[0, 1])})
    assert np.isfinite(pred)


def test_from_sklearn_partial_fit_with_refit_every_raises():
    with pytest.raises(ValueError, match="partial_fit"):
        from_sklearn(SGDClassifier(loss="log_loss"), refit_every=10)


def test_import_without_sklearn():
    code = r"""
import sys

class _Block:
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in ("sklearn", "scipy"):
            raise ModuleNotFoundError(f"blocked: {name}")
        return None

sys.meta_path.insert(0, _Block())

try:
    import dense_armor.bridges.sklearn  # noqa: F401
except ModuleNotFoundError as exc:
    msg = str(exc)
    assert "pip install dense-armor[sklearn]" in msg, msg
    print("OK:", msg)
else:
    raise SystemExit("expected ModuleNotFoundError")
"""
    r = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(Path(__file__).resolve().parents[2]),
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert r.returncode == 0, r.stderr[-400:]
    assert "pip install dense-armor[sklearn]" in r.stdout


def test_partial_fit_honours_classes():
    """classes fixes the class list at the first call."""
    m = SklearnClassifier(OnlineGaussianNB)
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([0, 1, 0, 1])
    m.partial_fit(X, y, classes=[0, 1, 2])
    assert m.classes_.tolist() == [0, 1, 2]
    p = m.predict_proba(np.array([[0.5]]))
    assert p.shape == (1, 3)


def test_from_sklearn_rejects_unknown_label():
    """A new label after the first call raises a clear ValueError."""
    from sklearn.linear_model import SGDClassifier

    w = from_sklearn(SGDClassifier(loss="log_loss", random_state=0), classes=[0, 1])
    w.learn_one({"a": 0.0}, 0)
    with pytest.raises(ValueError, match="not in the class list"):
        w.learn_one({"a": 1.0}, 2)
