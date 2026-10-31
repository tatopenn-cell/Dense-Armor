"""The batch classes of ``dense_armor.ml`` inside scikit-learn tools.

A ``Pipeline``, ``cross_val_score`` and ``GridSearchCV`` ask for the tags,
``score`` and ``fit_transform(X, y)``. The mixin of
``dense_armor/ml/_sklearn_compat.py`` gives them without changing an
algorithm. Each test fails when its part of the mixin is removed.
"""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("sklearn")

from sklearn.base import clone, is_classifier, is_regressor  # noqa: E402
from sklearn.metrics import r2_score  # noqa: E402
from sklearn.model_selection import GridSearchCV, cross_val_score  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.utils import get_tags  # noqa: E402

from dense_armor.ml.linear import LinearRegression, LogisticRegression  # noqa: E402
from dense_armor.ml.neighbors import KNNClassifier, KNNRegressor  # noqa: E402
from dense_armor.ml.preprocessing import (  # noqa: E402
    EWScaler,
    MinMaxScaler,
    RobustScaler,
    StandardScaler,
    VarianceThreshold,
)
from dense_armor.ml.proba import Gaussian  # noqa: E402


def _table(n=120, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal([1e3, 5.0, -2.0], [2.0, 1.0, 0.5], size=(n, 3))
    yc = (X[:, 1] > 5.0).astype(int)
    yr = X @ np.array([0.5, 2.0, 3.0])
    return X, yc, yr


def test_tags_name_the_kind_of_each_class():
    assert get_tags(LogisticRegression()).estimator_type == "classifier"
    assert get_tags(LinearRegression()).estimator_type == "regressor"
    assert get_tags(StandardScaler()).estimator_type is None
    assert get_tags(StandardScaler()).transformer_tags is not None
    assert is_classifier(KNNClassifier())
    assert is_regressor(LinearRegression())
    assert not is_classifier(LinearRegression())


def test_pipeline_of_a_scaler_and_a_classifier_equals_the_two_steps():
    X, yc, _ = _table()
    pipe = make_pipeline(StandardScaler(), LogisticRegression()).fit(X, yc)
    sc = StandardScaler().fit(X)
    lr = LogisticRegression().fit(sc.transform(X), yc)
    assert np.array_equal(pipe.predict(X), lr.predict(sc.transform(X)))


@pytest.mark.parametrize(
    "scaler", [StandardScaler, MinMaxScaler, RobustScaler, EWScaler]
)
def test_fit_transform_takes_the_target(scaler):
    X, yc, _ = _table()
    a = scaler().fit_transform(X)
    b = scaler().fit_transform(X, yc)
    assert np.array_equal(a, b)


def test_fit_transform_default_of_a_transformer_without_its_own():
    X, yc, _ = _table()
    out = VarianceThreshold().fit_transform(X, yc)
    assert out.shape[0] == X.shape[0]


def test_score_of_a_classifier_is_the_accuracy():
    X, yc, _ = _table()
    m = LogisticRegression().fit(X, yc)
    assert m.score(X, yc) == pytest.approx(float(np.mean(m.predict(X) == yc)))
    w = np.linspace(1.0, 2.0, len(yc))
    assert m.score(X, yc, sample_weight=w) == pytest.approx(
        float(np.average(m.predict(X) == yc, weights=w))
    )


def test_score_of_a_regressor_is_r2_of_scikit_learn():
    X, _, yr = _table()
    m = LinearRegression().fit(X, yr + np.random.default_rng(1).normal(size=len(yr)))
    assert m.score(X, yr) == pytest.approx(r2_score(yr, m.predict(X)))


def test_score_of_a_regressor_on_a_constant_target_follows_scikit_learn():
    X, _, _ = _table()
    m = KNNRegressor(n_neighbors=3).fit(X, np.full(len(X), 3.0))
    perfect, off = np.full(len(X), 3.0), np.full(len(X), 4.0)
    assert m.score(X, perfect) == 1.0 == r2_score(perfect, m.predict(X))
    assert m.score(X, off) == 0.0 == r2_score(off, m.predict(X))


def test_score_and_fit_transform_exist_only_where_they_make_sense():
    assert not hasattr(Gaussian(), "score")
    assert not hasattr(Gaussian(), "fit_transform")
    assert not hasattr(LogisticRegression(), "fit_transform")
    assert not hasattr(StandardScaler(), "score")
    assert hasattr(LogisticRegression(), "score")
    assert hasattr(StandardScaler(), "fit_transform")


def test_cross_val_score_of_a_classifier_and_a_regressor_is_finite():
    X, yc, yr = _table()
    sc = cross_val_score(LogisticRegression(), X, yc, cv=3)
    sr = cross_val_score(LinearRegression(), X, yr, cv=3)
    assert sc.shape == (3,) and np.all(np.isfinite(sc))
    assert sr.shape == (3,) and np.all(np.isfinite(sr))


def test_grid_search_runs_on_a_classifier():
    X, yc, _ = _table()
    gs = GridSearchCV(KNNClassifier(), {"n_neighbors": [1, 3, 5]}, cv=3).fit(X, yc)
    assert gs.best_params_["n_neighbors"] in (1, 3, 5)
    assert np.isfinite(gs.best_score_)


def test_clone_keeps_the_parameters():
    m = clone(KNNClassifier(n_neighbors=7))
    assert m.n_neighbors == 7


def test_import_without_scikit_learn_works_and_the_tags_need_it():
    code = r"""
import sys

class _Block:
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] == "sklearn":
            raise ModuleNotFoundError(f"blocked: {name}")
        return None

sys.meta_path.insert(0, _Block())

import dense_armor.ml
from dense_armor.ml.linear import LinearRegression

m = LinearRegression()
try:
    m.__sklearn_tags__()
except ModuleNotFoundError:
    print("OK")
else:
    raise SystemExit("expected ModuleNotFoundError")
"""
    r = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(Path(__file__).resolve().parents[2]),
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert r.returncode == 0, r.stderr[-400:]
    assert "OK" in r.stdout


REASONS = {
    "check_n_features_in": "fit does not set n_features_in_",
    "check_n_features_in_after_fitting": "fit does not set n_features_in_",
    "check_estimators_unfitted": "predict before fit does not raise NotFittedError",
    "check_complex_data": "complex data is not rejected",
    "check_estimators_empty_data_messages": "empty data is not rejected with the scikit-learn message",
    "check_fit1d": "a 1-D X is accepted as one column, not rejected",
    "check_fit2d_1sample": "one sample is not rejected with the scikit-learn message",
    "check_fit2d_predict1d": "a 1-D X in predict is not rejected with the scikit-learn message",
    "check_requires_y_none": "y=None in fit is not rejected with the scikit-learn message",
    "check_estimators_nan_inf": "non-finite values are dropped by the class, not rejected",
    "check_supervised_y_no_nan": "a non-finite y is dropped by the class, not rejected",
    "check_supervised_y_2d": "a 2-D y is not supported",
    "check_estimator_sparse_tag": "sparse input is not supported",
    "check_estimator_sparse_array": "sparse input is not rejected with the scikit-learn message",
    "check_estimator_sparse_matrix": "sparse input is not rejected with the scikit-learn message",
    "check_classifiers_classes": "string labels are not supported",
    "check_classifiers_train": "the training checks of scikit-learn on its own data set do not hold",
    "check_classifiers_regression_target": "a continuous target is not rejected",
    "check_transformer_general": "transform with another number of features is not rejected",
}

DECLARED = {
    LogisticRegression: {
        "check_classifiers_classes",
        "check_classifiers_regression_target",
        "check_classifiers_train",
        "check_complex_data",
        "check_estimator_sparse_array",
        "check_estimator_sparse_matrix",
        "check_estimator_sparse_tag",
        "check_estimators_empty_data_messages",
        "check_estimators_nan_inf",
        "check_estimators_unfitted",
        "check_fit1d",
        "check_fit2d_predict1d",
        "check_n_features_in",
        "check_n_features_in_after_fitting",
        "check_supervised_y_2d",
        "check_supervised_y_no_nan",
    },
    LinearRegression: {
        "check_complex_data",
        "check_estimator_sparse_array",
        "check_estimator_sparse_matrix",
        "check_estimator_sparse_tag",
        "check_estimators_empty_data_messages",
        "check_estimators_nan_inf",
        "check_estimators_unfitted",
        "check_fit1d",
        "check_fit2d_1sample",
        "check_fit2d_predict1d",
        "check_n_features_in",
        "check_n_features_in_after_fitting",
        "check_requires_y_none",
        "check_supervised_y_2d",
        "check_supervised_y_no_nan",
    },
    StandardScaler: {
        "check_complex_data",
        "check_estimator_sparse_array",
        "check_estimator_sparse_matrix",
        "check_estimator_sparse_tag",
        "check_estimators_empty_data_messages",
        "check_estimators_nan_inf",
        "check_fit1d",
        "check_fit2d_predict1d",
        "check_n_features_in",
        "check_n_features_in_after_fitting",
        "check_transformer_general",
    },
}


@pytest.mark.parametrize("cls", list(DECLARED), ids=lambda c: c.__name__)
def test_check_estimator_fails_only_on_the_declared_checks(cls):
    from sklearn.utils.estimator_checks import check_estimator

    results = check_estimator(cls(), on_fail=None)
    failed = {r["check_name"] for r in results if r["status"] == "failed"}
    assert failed == DECLARED[cls]
    assert all(name in REASONS for name in failed)
