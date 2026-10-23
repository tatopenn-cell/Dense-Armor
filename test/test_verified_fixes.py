"""Fixes of the defects found by reading: batch tree, linear models, metric learning, misc."""

import numpy as np
import pytest

from dense_armor.utility.evaluate import evaluate_events, progressive_val_score
from dense_armor.utility.evaluate.model_selection import OnlineModelSelection
from dense_armor.utility.learn.calibration import OnlinePlattScaling
from dense_armor.utility.learn.metric_learning import LEGO
from dense_armor.utility.learn.online_dynamics import RecursiveLeastSquares
from dense_armor.utility.linear.bayes import BayesianLinearRegression
from dense_armor.utility.linear.glm import LogisticRegression
from dense_armor.utility.metrics import Accuracy
from dense_armor.utility.misc.diagnostic import diag
from dense_armor.utility.misc.metro import Metro
from dense_armor.utility.tree.batch import BatchDecisionTreeClassifier


class _Const:
    def learn_one(self, x, y, t=None):
        return self

    def predict_one(self, x, t=None):
        return 1


def test_current_best_before_the_run_raises():
    sel = OnlineModelSelection([_Const(), _Const()], Accuracy)
    with pytest.raises(ValueError):
        sel.current_best()
    sel.run([({}, 1)] * 3)
    assert sel.current_best() in (0, 1)


def test_a_negative_delay_raises():
    with pytest.raises(ValueError):
        progressive_val_score([({}, 1)], _Const(), Accuracy(), delay=-5)


def test_a_delay_in_seconds_needs_timestamps():
    with pytest.raises(ValueError):
        progressive_val_score([({}, 1)] * 3, _Const(), Accuracy(), delay=1.5)


def test_an_anomaly_detector_without_t_is_called_without_it():
    class Det:
        score_threshold = 0.5

        def score_one(self, x):
            return 1.0

        def learn_one(self, x):
            return self

    stream = [(float(i), {"a": 1.0}) for i in range(5)]
    assert evaluate_events(stream, Det(), [(1.0, 3.0)])["missed"] == 0


def test_batch_fit_refuses_a_row_with_a_missing_feature():
    with pytest.raises(ValueError):
        BatchDecisionTreeClassifier().fit([{"a": 1.0, "b": 2.0}, {"a": 1.0}], [0, 1])


def test_batch_fit_refuses_nan_and_inf():
    for bad in (float("nan"), float("inf")):
        with pytest.raises(ValueError):
            BatchDecisionTreeClassifier().fit([{"a": 0.0}, {"a": bad}], [0, 1])


def test_batch_fit_refuses_a_value_that_is_not_a_number():
    with pytest.raises(ValueError):
        BatchDecisionTreeClassifier().fit([{"a": 0.0}, {"a": "x"}], [0, 1])


def test_batch_has_no_unused_class_set():
    clf = BatchDecisionTreeClassifier().fit([{"a": 0.0}, {"a": 1.0}], [0, 1])
    assert not hasattr(clf, "_classes_")


def test_logistic_treats_any_nonzero_label_as_one():
    one, two = LogisticRegression(), LogisticRegression()
    x = {"a": 1.0, "b": -0.5}
    one.learn_one(x, 1)
    two.learn_one(x, 2)
    assert np.array_equal(one._w, two._w)


def test_logistic_and_bayes_skip_an_infinite_value():
    log = LogisticRegression()
    log.learn_one({"a": float("inf")}, 1)
    assert log.n_missing == 1 and log._w is None
    bay = BayesianLinearRegression()
    bay.learn_one({"a": 1.0}, float("inf"))
    bay.learn_one({"a": float("inf")}, 1.0)
    assert bay.n_missing == 2 and bay._m is None


def test_rls_counts_a_missing_key_as_zero():
    rls = RecursiveLeastSquares(feature_keys=["a", "b"])
    rls.learn_one({"a": 1.0, "b": 2.0}, 3.0)
    rls.learn_one({"a": 1.0}, 1.0)
    assert np.isfinite(rls.predict_one({"b": 1.0}))


def test_lego_percentile_targets_refuses_one_class():
    X = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    with pytest.raises(ValueError):
        LEGO.percentile_targets(X, [1, 1, 1])
    with pytest.raises(ValueError):
        LEGO.percentile_targets(X, [0, 1, 2])
    near, far = LEGO.percentile_targets(X, [0, 0, 1])
    assert near <= far


def test_platt_asks_the_base_once_per_call():
    class Base:
        calls = 0

        def predict_proba_one(self, x, **kw):
            Base.calls += 1
            return {False: 0.3, True: 0.7}

        def learn_one(self, x, y, **kw):
            return self

    cal = OnlinePlattScaling(Base())
    cal.predict_proba_one({"x": 1.0})
    assert Base.calls == 1
    cal.learn_one({"x": 1.0}, 1)
    assert Base.calls == 2


def test_metro_keeps_the_sign():
    m = Metro()
    for v in (-1e-9, 1e-9, -3.5e-12):
        enc, fact = m.enc(v)
        assert np.sign(enc) == np.sign(v)
        assert m.dec(enc, fact) == pytest.approx(v)
    assert m.enc(0.0) == (0.0, 1.0)


def test_diag_scales_by_dtype_and_refuses_mixed_arguments():
    sig = np.sin(np.linspace(0.0, 20.0, 2000)).astype(np.float64)
    as16 = (sig * 32767).astype(np.int16)
    as32 = (sig * 2147483647).astype(np.int32)
    same = diag(as32, as16)
    ref = diag(sig.astype(np.float32), sig.astype(np.float32))
    assert same is not None and ref is not None
    with pytest.raises(TypeError):
        diag("a.wav", sig)
