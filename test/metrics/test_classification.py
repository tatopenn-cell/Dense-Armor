"""Tests for dense_armor/utility/metrics/{base,classification}.py."""
import doctest

import numpy as np
import pytest
from sklearn import metrics as skm

import dense_armor.utility.metrics.base as base_mod
import dense_armor.utility.metrics.classification as cls_mod
from dense_armor.roles import Classifier, Regressor
from dense_armor.utility.metrics import (
    F1,
    Accuracy,
    BalancedAccuracy,
    BrierScore,
    CohenKappa,
    ConfusionMatrix,
    FBeta,
    LogLoss,
    Precision,
    Recall,
)


def _stream(n: int = 500, seed: int = 0):
    rng = np.random.default_rng(seed)
    y_true = rng.integers(0, 2, n).tolist()
    y_pred = [int(not y) if rng.random() < 0.25 else int(y) for y in y_true]
    return y_true, y_pred


def test_accuracy_matches_sklearn():
    yt, yp = _stream()
    m = Accuracy()
    for a, b in zip(yt, yp):
        m.update(a, b)
    assert m.get() == pytest.approx(skm.accuracy_score(yt, yp))


def test_accuracy_revert_restores():
    m = Accuracy()
    m.update(1, 1)
    m.update(0, 0)
    before = m.get()
    m.update(0, 1)
    m.revert(0, 1)
    assert m.get() == pytest.approx(before)


def test_accuracy_merge_matches_whole():
    yt, yp = _stream(400)
    whole = Accuracy()
    for a, b in zip(yt, yp):
        whole.update(a, b)
    left, right = Accuracy(), Accuracy()
    for a, b in zip(yt[:200], yp[:200]):
        left.update(a, b)
    for a, b in zip(yt[200:], yp[200:]):
        right.update(a, b)
    merged = left.merge(right)
    assert merged.get() == pytest.approx(whole.get())


def test_balanced_accuracy_matches_sklearn():
    yt, yp = _stream(300)
    m = BalancedAccuracy()
    for a, b in zip(yt, yp):
        m.update(a, b)
    assert m.get() == pytest.approx(skm.balanced_accuracy_score(yt, yp))


def test_precision_matches_sklearn_binary():
    yt, yp = _stream(300)
    m = Precision(positive=1, average="binary")
    for a, b in zip(yt, yp):
        m.update(a, b)
    assert m.get() == pytest.approx(skm.precision_score(yt, yp, pos_label=1))


def test_recall_matches_sklearn_binary():
    yt, yp = _stream(300)
    m = Recall(positive=1, average="binary")
    for a, b in zip(yt, yp):
        m.update(a, b)
    assert m.get() == pytest.approx(skm.recall_score(yt, yp, pos_label=1))


def test_f1_matches_sklearn_binary():
    yt, yp = _stream(300)
    m = F1(positive=1)
    for a, b in zip(yt, yp):
        m.update(a, b)
    assert m.get() == pytest.approx(skm.f1_score(yt, yp, pos_label=1))


@pytest.mark.parametrize("avg", ["macro", "micro", "weighted"])
def test_f1_multiclass_averages(avg):
    rng = np.random.default_rng(3)
    yt = rng.integers(0, 3, 400).tolist()
    yp = [int(np.random.default_rng(i).integers(0, 3)) for i in range(400)]
    m = F1(average=avg)
    for a, b in zip(yt, yp):
        m.update(a, b)
    assert m.get() == pytest.approx(skm.f1_score(yt, yp, average=avg))


def test_fbeta_matches_sklearn():
    yt, yp = _stream(400)
    for beta in (0.5, 1.0, 2.0):
        m = FBeta(beta=beta, positive=1)
        for a, b in zip(yt, yp):
            m.update(a, b)
        assert m.get() == pytest.approx(
            skm.fbeta_score(yt, yp, beta=beta, pos_label=1)
        )


def test_cohen_kappa_matches_sklearn():
    yt, yp = _stream(500)
    m = CohenKappa()
    for a, b in zip(yt, yp):
        m.update(a, b)
    assert m.get() == pytest.approx(skm.cohen_kappa_score(yt, yp))


def test_confusion_matrix_matches_sklearn():
    yt, yp = _stream(300)
    m = ConfusionMatrix()
    for a, b in zip(yt, yp):
        m.update(a, b)
    cm = m.get()
    expected = skm.confusion_matrix(yt, yp)
    for i, a in enumerate([0, 1]):
        for j, b in enumerate([0, 1]):
            assert cm[a][b] == pytest.approx(float(expected[i, j]))


def test_logloss_matches_sklearn():
    rng = np.random.default_rng(4)
    yt = rng.integers(0, 2, 500).tolist()
    pp = rng.uniform(0.05, 0.95, 500).tolist()
    m = LogLoss()
    for a, b in zip(yt, pp):
        m.update(a, b)
    assert m.get() == pytest.approx(skm.log_loss(yt, pp, labels=[0, 1]))


def test_brier_matches_sklearn():
    rng = np.random.default_rng(5)
    yt = rng.integers(0, 2, 500).tolist()
    pp = rng.uniform(0.0, 1.0, 500).tolist()
    m = BrierScore()
    for a, b in zip(yt, pp):
        m.update(a, b)
    assert m.get() == pytest.approx(skm.brier_score_loss(yt, pp))


def test_missing_pairs_are_skipped_and_counted():
    m = Accuracy()
    m.update(1, 1)
    m.update(None, 1)
    m.update(1, None)
    m.update(float("nan"), 1)
    m.update(1, float("nan"))
    m.update(0, 0)
    assert m.n == 2
    assert m.n_missing == 4
    assert m.get() == pytest.approx(1.0)


def test_weights_respected():
    m = Accuracy()
    m.update(1, 1, w=2.0)
    m.update(0, 1, w=1.0)
    assert m.get() == pytest.approx(2.0 / 3.0)


def test_works_with_role():
    assert Accuracy().works_with(Classifier())
    assert not Accuracy().works_with(Regressor())


def test_merge_type_error():
    with pytest.raises(TypeError):
        Accuracy().merge("x")
    with pytest.raises(TypeError):
        Precision().merge(Recall())
    with pytest.raises(TypeError):
        Accuracy().merge(BalancedAccuracy())


def test_merge_across_many_shards():
    yt, yp = _stream(900)
    whole = F1(positive=1)
    for a, b in zip(yt, yp):
        whole.update(a, b)
    parts = [F1(positive=1) for _ in range(3)]
    for i, (a, b) in enumerate(zip(yt, yp)):
        parts[i % 3].update(a, b)
    merged = parts[0].merge(parts[1]).merge(parts[2])
    assert merged.get() == pytest.approx(whole.get())


def test_doctests():
    assert doctest.testmod(base_mod).failed == 0
    assert doctest.testmod(cls_mod).failed == 0
