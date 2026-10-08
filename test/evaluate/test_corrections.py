"""Tests for the corrections of batch 2."""

import numpy as np
import pytest
from sklearn import metrics as skm

from dense_armor.utility.evaluate import (
    best_of,
    progressive_val_proba_score,
    progressive_val_score,
)
from dense_armor.utility.metrics import (
    F1,
    Accuracy,
    BrierScore,
    MeanAbsoluteError,
    R2Score,
    Rolling,
    RollingAUC,
)


class _LastValue:
    def __init__(self) -> None:
        self._last = 0

    def learn_one(self, x, y, t=None):
        self._last = y
        return self

    def predict_one(self, x, t=None):
        return self._last


class _Shift:
    """Predict a step function: 0 before mid, 1 after."""

    def __init__(self, mid=10):
        self.mid = mid
        self.i = 0

    def learn_one(self, x, y, t=None):
        return self

    def predict_one(self, x, t=None):
        self.i += 1
        return 0 if self.i <= self.mid else 1


# ---------- prequential: delayed learning ----------
def test_delayed_label_learning_is_worse_on_step():
    stream = [({}, 0)] * 20 + [({}, 1)] * 20
    a = progressive_val_score(stream, _LastValue(), Accuracy())
    b = progressive_val_score(stream, _LastValue(), Accuracy(), delay=5)
    assert a.n == b.n
    assert b.get() <= a.get()


def test_delayed_label_learns_at_arrival():
    stream = [({}, 1)] * 10
    a = progressive_val_score(stream, _LastValue(), Accuracy())
    assert a.n == 10
    assert a.get() == pytest.approx(9.0 / 10.0)


# ---------- rolling: window_s by timestamp ----------
def test_rolling_window_s_drops_by_time():
    m = Rolling(Accuracy(), window_s=0.05)
    m.update(1, 1, t=0.0)
    m.update(1, 1, t=0.05)
    m.update(0, 1, t=0.10)
    assert m.count == 1
    assert m.get() == 0.0


def test_rolling_window_s_rejects_decreasing_t():
    m = Rolling(Accuracy(), window_s=0.03)
    m.update(1, 1, t=0.05)
    with pytest.raises(ValueError):
        m.update(1, 1, t=0.04)


# ---------- RollingAUC ----------
def test_rolling_auc_matches_sklearn():
    rng = np.random.default_rng(0)
    labels = rng.integers(0, 2, 100).tolist()
    scores = rng.uniform(0, 1, 100).tolist()
    m = RollingAUC(window=50)
    for y, s in zip(labels, scores):
        m.update(y, s)
    expected = skm.roc_auc_score(labels[-50:], scores[-50:])
    assert m.get() == pytest.approx(expected)


def test_rolling_auc_handles_ties():
    m = RollingAUC(window=4)
    for y, s in [(0, 0.5), (1, 0.5), (0, 0.5), (1, 0.5)]:
        m.update(y, s)
    assert m.get() == pytest.approx(0.5)


# ---------- F1.merge returns F1 ----------
def test_f1_merge_returns_f1():
    a = F1(positive=1)
    b = F1(positive=1)
    a.update(1, 1)
    b.update(0, 1)
    merged = a.merge(b)
    assert isinstance(merged, F1)


# ---------- R2Score Welford ----------
def test_r2_welford_matches_sklearn():
    rng = np.random.default_rng(1)
    yt = rng.normal(1e6, 1.0, 200).tolist()
    yp = [y + rng.normal(0, 0.1) for y in yt]
    m = R2Score()
    for a, b in zip(yt, yp):
        m.update(a, b)
    assert m.get() == pytest.approx(skm.r2_score(yt, yp))


# ---------- vector targets ----------
def test_vector_target_reports_per_joint_and_mean():
    m = MeanAbsoluteError()
    m.update([1.0, 2.0], [1.5, 2.5])
    m.update([2.0, 4.0], [2.5, 3.5])
    out = m.get()
    assert "per_joint" in out and "mean" in out
    assert len(out["per_joint"]) == 2
    assert out["mean"] == pytest.approx((0.5 + 0.5) / 2)


# ---------- progressive_val_proba_score ----------
class _Probas:
    def __init__(self, p=0.9):
        self.p = p

    def predict_proba_one(self, x, t=None):
        return {0: 1 - self.p, 1: self.p}

    def learn_one(self, x, y, t=None):
        return self


def test_proba_score_logloss():
    stream = [({}, 1)] * 20
    m = progressive_val_proba_score(stream, _Probas(0.9), BrierScore())
    assert m.get() == pytest.approx(0.01, abs=1e-9)


# ---------- best_of ----------
def test_best_of_picks_higher_metric():
    stream = [({}, 1)] * 20
    sel = best_of(
        [_LastValue(), _Probas()],
        Accuracy,
        stream=stream,
    )
    assert sel.current_best() == 0


# ---------- np.floating / 0-d NaN ----------
def test_np_floating_nan_is_skipped():
    m = Accuracy()
    m.update(np.float64(1.0), np.float64(1.0))
    m.update(np.float64("nan"), np.float64(1.0))
    m.update(np.float64(1.0), np.float64("nan"))
    m.update(np.array(1.0), np.array(1.0))
    m.update(np.array(np.nan), np.array(1.0))
    assert m.n == 2
    assert m.n_missing == 3
