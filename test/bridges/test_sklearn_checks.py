"""Item A: the three wrappers pass check_estimator with no skipped check.

The check runs in the scikit-learn test suite, in the version pinned by
the ``sklearn`` extra (scikit-learn >= 1.6). The list of checks that
cannot hold for an online learner is empty: all checks pass, so no
check is skipped and no ``expected_failed_checks`` entry is needed.

Documentation consulted for the version in use:
- https://scikit-learn.org/1.6/developers/develop.html
- https://scikit-learn.org/1.6/modules/generated/sklearn.utils.estimator_checks.parametrize_with_checks.html
"""

import pytest

pytest.importorskip("sklearn")

from sklearn.utils.estimator_checks import parametrize_with_checks

from dense_armor.bridges.sklearn import (
    SklearnClassifier,
    SklearnRegressor,
    SklearnTransformer,
)
from dense_armor.utility.learn.online_classifiers import OnlineGaussianNB
from dense_armor.utility.linear.glm import LinearRegression
from dense_armor.utility.optim import SGD
from dense_armor.utility.preprocessing.scale import MinMaxScaler


@parametrize_with_checks(
    [
        SklearnClassifier(OnlineGaussianNB),
        SklearnRegressor(LinearRegression(optimizer=SGD(lr=0.01))),
        SklearnTransformer(MinMaxScaler),
    ]
)
@pytest.mark.heavy
def test_sklearn_check(estimator, check):
    check(estimator)
