"""The refusals added to the metrics: merge, joints, rolling."""

import numpy as np
import pytest

from dense_armor.utility.metrics import (
    F1,
    Accuracy,
    FBeta,
    MeanAbsoluteError,
    MeanSquaredError,
    Precision,
    R2Score,
    Recall,
    RollingAUC,
    RootMeanSquaredError,
)
from dense_armor.utility.metrics.rolling import Rolling

JOINTS = [MeanAbsoluteError, MeanSquaredError, RootMeanSquaredError]


@pytest.mark.parametrize("cls", [Precision, Recall, FBeta, F1])
def test_merge_refuses_a_different_positive(cls):
    left, right = cls(positive=1), cls(positive=0)
    left.update(1, 1)
    right.update(0, 0)
    with pytest.raises(ValueError):
        left.merge(right)


@pytest.mark.parametrize("cls", [Precision, Recall, FBeta, F1])
def test_merge_refuses_a_different_average(cls):
    with pytest.raises(ValueError):
        cls(average="macro").merge(cls(average="micro"))


@pytest.mark.parametrize("cls", [Precision, Recall, FBeta, F1])
def test_merge_with_the_same_configuration_adds_the_counts(cls):
    left, right = cls(positive=1), cls(positive=1)
    left.update(1, 1)
    right.update(1, 0)
    assert left.merge(right).n_ == 2


@pytest.mark.parametrize("cls", JOINTS)
def test_a_scalar_after_a_vector_is_refused(cls):
    m = cls()
    m.update(np.array([1.0, 2.0]), np.array([1.5, 2.5]))
    with pytest.raises(ValueError):
        m.update(3.0, 3.5)


@pytest.mark.parametrize("cls", JOINTS)
def test_a_vector_after_a_scalar_is_refused(cls):
    m = cls()
    m.update(3.0, 3.5)
    with pytest.raises(ValueError):
        m.update(np.array([1.0, 2.0]), np.array([1.5, 2.5]))


@pytest.mark.parametrize("cls", JOINTS)
def test_a_refused_update_leaves_the_state_unchanged(cls):
    m = cls()
    m.update(np.array([1.0, 2.0]), np.array([1.5, 2.5]))
    before = m.get()
    with pytest.raises(ValueError):
        m.update(3.0, 3.5)
    assert m.get() == before


@pytest.mark.parametrize("cls", JOINTS)
def test_merge_refuses_a_different_number_of_joints(cls):
    left, right = cls(), cls()
    left.update(np.array([1.0, 2.0]), np.array([1.5, 2.5]))
    right.update(3.0, 3.5)
    with pytest.raises(ValueError):
        left.merge(right)


@pytest.mark.parametrize("cls", [R2Score, RollingAUC])
def test_rolling_refuses_a_metric_that_cannot_revert(cls):
    with pytest.raises(ValueError):
        Rolling(cls(), window=3)


def test_rolling_window_s_needs_the_timestamp():
    m = Rolling(Accuracy(), window_s=1.0)
    with pytest.raises(ValueError):
        m.update(1, 1)
    assert m.count == 0
