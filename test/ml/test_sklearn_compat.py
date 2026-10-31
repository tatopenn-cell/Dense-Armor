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
from dense_armor.ml.neighbors import KNNClassifier  # noqa: E402
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


def test_score_of_a_regressor_on_a_constant_target_is_finite():
    X, _, _ = _table()
    m = LinearRegression().fit(X, np.full(len(X), 3.0))
    assert m.score(X, np.full(len(X), 3.0)) == 1.0
    assert m.score(X, np.full(len(X), 4.0)) == 0.0


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
        if name.split(".")[0] in ("sklearn", "scipy"):
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
