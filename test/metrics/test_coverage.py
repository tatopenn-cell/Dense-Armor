"""Revert, merge, empty state and vector paths of every online metric."""
import numpy as np
import pytest

from dense_armor.roles import Classifier, Regressor
from dense_armor.utility.evaluate import (
    OnlineModelSelection,
    progressive_val_proba_score,
)
from dense_armor.utility.metrics import (
    F1,
    Accuracy,
    BalancedAccuracy,
    BrierScore,
    CohenKappa,
    ConfusionMatrix,
    FBeta,
    GaussianNLL,
    IntervalCoverage,
    LogLoss,
    MeanAbsoluteError,
    MeanIntervalWidth,
    MeanSquaredError,
    Metric,
    Precision,
    R2Score,
    Recall,
    RollingAUC,
    RootMeanSquaredError,
)

LABELS = [(0, 0), (1, 1), (0, 1), (2, 2), (1, 0), (2, 1)]
PROBS = [(1, 0.9), (0, 0.2), (1, 0.6), (0, 0.4)]
REG = [(1.0, 1.5), (2.0, 2.5), (3.0, 2.0), (4.0, 4.5)]
INT = [(0.5, (0.0, 1.0)), (2.0, (0.0, 1.0)), (1.0, (0.5, 3.0))]
NLL = [(0.0, (0.0, 1.0)), (1.0, (0.5, 2.0))]

CASES = [
    (Accuracy, LABELS),
    (BalancedAccuracy, LABELS),
    (CohenKappa, LABELS),
    (ConfusionMatrix, LABELS),
    (lambda: Precision(average="macro"), LABELS),
    (lambda: Recall(average="weighted"), LABELS),
    (lambda: Precision(average="micro"), LABELS),
    (lambda: Recall(average="micro"), LABELS),
    (lambda: F1(average="weighted"), LABELS),
    (lambda: FBeta(beta=2.0, average="micro"), LABELS),
    (LogLoss, PROBS),
    (BrierScore, PROBS),
    (MeanAbsoluteError, REG),
    (MeanSquaredError, REG),
    (RootMeanSquaredError, REG),
    (R2Score, REG),
    (IntervalCoverage, INT),
    (MeanIntervalWidth, INT),
    (GaussianNLL, NLL),
]


def _fed(factory, pairs):
    m = factory()
    for a, b in pairs:
        m.update(a, b)
    return m


@pytest.mark.parametrize("factory,pairs", CASES)
def test_merge_equals_whole(factory, pairs):
    whole = _fed(factory, pairs)
    half = len(pairs) // 2
    merged = _fed(factory, pairs[:half]).merge(_fed(factory, pairs[half:]))
    assert merged.n == whole.n
    if isinstance(whole.get(), dict):
        assert merged.get() == whole.get()
    else:
        assert merged.get() == pytest.approx(whole.get())


@pytest.mark.parametrize("factory,pairs", CASES)
def test_revert_or_refuse(factory, pairs):
    m = _fed(factory, pairs[:-1])
    before = m.get()
    m.update(*pairs[-1])
    try:
        m.revert(*pairs[-1])
    except NotImplementedError:
        return
    if isinstance(before, dict):
        assert {k: {c: v for c, v in r.items() if v} for k, r in m.get().items()} == before
    else:
        assert m.get() == pytest.approx(before)


@pytest.mark.parametrize("factory,pairs", CASES)
def test_empty_and_wrong_merge(factory, pairs):
    m = factory()
    assert m.n == 0
    m.get()
    repr(m)
    with pytest.raises(TypeError):
        m.merge(object())


@pytest.mark.parametrize("cls", [MeanAbsoluteError, MeanSquaredError, RootMeanSquaredError])
def test_vector_targets_update_revert_merge(cls):
    a = cls()
    a.update([1.0, 2.0], [1.5, 2.5])
    a.update([2.0, 4.0], [2.5, 3.0])
    before = a.get()
    a.update([0.0, 0.0], [9.0, 9.0])
    a.revert([0.0, 0.0], [9.0, 9.0])
    assert a.get()["mean"] == pytest.approx(before["mean"])
    b = cls()
    b.update([1.0, 1.0], [1.0, 2.0])
    merged = a.merge(b)
    assert len(merged.get()["per_joint"]) == 2


def test_role_checks():
    assert Accuracy().works_with(Classifier())
    assert MeanAbsoluteError().works_with(Regressor())
    assert Metric().works_with(object())


def test_rolling_auc_merge_and_errors():
    a, b = RollingAUC(window=4), RollingAUC(window=4)
    for y, s in [(0, 0.1), (1, 0.9)]:
        a.update(y, s)
    for y, s in [(0, 0.3), (1, 0.2)]:
        b.update(y, s)
    assert a.merge(b).get() == pytest.approx(0.75)
    assert RollingAUC(window=4).get() == 0.5
    with pytest.raises(TypeError):
        a.merge(object())
    with pytest.raises(ValueError):
        a.merge(RollingAUC(window=5))
    with pytest.raises(ValueError):
        RollingAUC(window=0)


class _Probas:
    def predict_proba_one(self, x):
        return {0: 0.2, 1: 0.8}

    def learn_one(self, x, y):
        return self


def test_proba_score_with_delay_and_trace():
    s = [({}, i % 2) for i in range(20)]
    m, tr = progressive_val_proba_score(s, _Probas(), BrierScore(), delay=3, every=5)
    assert m.n == 20
    assert len(tr) == 4


def test_model_selection_errors_and_best_model():
    with pytest.raises(ValueError):
        OnlineModelSelection([], Accuracy)
    sel = OnlineModelSelection([_Probas()], MeanSquaredError)
    sel.run([({}, 0.0)])
    assert sel.best_model() is sel.models[0]


def test_nan_numpy_scalars_skipped():
    m = MeanSquaredError()
    m.update(np.float32("nan"), 1.0)
    m.update(1.0, 1.0)
    assert (m.n, m.n_missing) == (1, 1)
