"""Tests for the defects found in the review of the roles."""

from __future__ import annotations

import json

import jax.numpy as jnp
import numpy as np
import pytest

from dense_armor.roles import (
    AdaptiveConformalRegressor,
    AnomalyDetector,
    AnomalyGate,
    Protected,
    RealtimePipeline,
    Regressor,
    SafeEstimator,
    Signal,
    Transformer,
    UnitCheckedPipeline,
    call_json,
)


class _Counter(Transformer):
    """Adds 1 to every feature and remembers what it learned from."""

    def __init__(self):
        self.seen = []

    def learn_one(self, x, y=None, t=None):
        self.seen.append(dict(x))
        return self

    def transform_one(self, x, t=None):
        return {k: v + 1.0 for k, v in x.items()}


class _Last:
    def __init__(self):
        self.got = None

    def learn_one(self, x, y=None, t=None):
        self.got = (dict(x), y)
        return self

    def transform_one(self, x, t=None):
        return x


def test_unit_checked_pipeline_trains_every_transformer():
    a, b, last = _Counter(), _Counter(), _Last()
    UnitCheckedPipeline([a, b, last]).learn_one({"v": 1.0}, 5.0)
    assert a.seen == [{"v": 1.0}]
    assert b.seen == [{"v": 2.0}]
    assert last.got == ({"v": 3.0}, 5.0)


def test_unit_checked_pipeline_keeps_working_with_a_step_that_does_not_learn():
    class _Plain:
        def transform_one(self, x, t=None):
            return {k: v * 2.0 for k, v in x.items()}

    last = _Last()
    UnitCheckedPipeline([_Plain(), last]).learn_one({"v": 1.0}, 1.0)
    assert last.got == ({"v": 2.0}, 1.0)


class _Shift(Transformer):
    def __init__(self, name, by):
        self.name = name
        self.by = by
        self.n = 0

    def learn_one(self, x, y=None, t=None):
        self.n += 1
        return self

    def transform_one(self, x, t=None):
        return {self.name: x["v"] + self.by}


def test_union_chains_and_trains_every_member():
    a, b, c = _Shift("a", 1.0), _Shift("b", 2.0), _Shift("c", 3.0)
    t = (a + b) + c
    assert isinstance(t, Transformer)
    assert t.transform_one({"v": 0.0}) == {"a": 1.0, "b": 2.0, "c": 3.0}
    t.learn_one({"v": 0.0})
    assert [s.n for s in (a, b, c)] == [1, 1, 1]


def test_product_chains_and_mixes_with_union():
    a, b, c = _Shift("a", 1.0), _Shift("b", 2.0), _Shift("c", 3.0)
    assert ((a * b) * c).transform_one({"v": 1.0}) == {"a*b*c": 2.0 * 3.0 * 4.0}
    assert (a + (b * c)).transform_one({"v": 1.0}) == {"a": 2.0, "b*c": 3.0 * 4.0}
    assert ((a + b) * c).transform_one({"v": 1.0}) == {
        "a*c": 2.0 * 4.0,
        "b*c": 3.0 * 4.0,
    }


class _Scored(AnomalyDetector):
    def __init__(self, score=5.0):
        self.score = score
        self.learned = 0

    def score_one(self, x, t=None):
        return self.score

    def learn_one(self, x, t=None):
        self.learned += 1
        return self


class _WithThreshold(_Scored):
    threshold = 3.0


def test_gate_refuses_a_detector_without_threshold():
    with pytest.raises(ValueError, match="threshold"):
        AnomalyGate(_Scored())


def test_gate_explicit_threshold_wins_and_detector_threshold_is_followed():
    assert not AnomalyGate(_Scored(), threshold=6.0).classify(5.0)
    assert AnomalyGate(_Scored(), threshold=4.0).classify(5.0)
    det = _WithThreshold()
    gate = AnomalyGate(det)
    assert not gate.classify(3.0)
    assert gate.classify(3.1)
    det.threshold = 10.0
    assert not gate.classify(5.0)
    assert AnomalyGate(det, threshold=1.0).classify(5.0)


def test_gate_protects_only_above_the_threshold():
    above = _WithThreshold(score=5.0)
    AnomalyGate(above).learn_one({"a": 1.0})
    assert above.learned == 0
    below = _WithThreshold(score=2.0)
    AnomalyGate(below).learn_one({"a": 1.0})
    assert below.learned == 1


def test_a_gate_subclass_that_defines_classify_needs_no_threshold():
    class Fixed(AnomalyGate):
        def classify(self, score):
            return score > 3.0

    assert Fixed(_Scored(5.0)).classify(5.0)
    assert not Fixed(_Scored(5.0)).classify(2.0)


def test_gate_threshold_is_a_parameter():
    gate = AnomalyGate(_WithThreshold(), protect=False, threshold=2.5)
    assert gate.get_params(deep=False)["threshold"] == 2.5


class _Zero(Regressor):
    def learn_one(self, x, y, t=None):
        return self

    def predict_one(self, x, t=None, return_std=False):
        return 0.0


def test_conformal_std_is_the_rms_of_the_residuals_and_not_the_half_width():
    m = AdaptiveConformalRegressor(_Zero(), alpha=0.1, window=10)
    for y in (3.0, 4.0):
        m.learn_one({"x": 1.0}, y)
    mean, std = m.predict_one({"x": 1.0}, return_std=True)
    assert mean == 0.0
    assert std == pytest.approx(((9.0 + 16.0) / 2.0) ** 0.5)
    est = m.predict_one({"x": 1.0}, return_estimate=True)
    assert est.mean == 0.0
    assert est.var == pytest.approx(12.5)
    lo, hi = m.predict_interval({"x": 1.0})
    assert hi == pytest.approx(-lo)
    assert abs(hi - std) > 0.1


def test_conformal_std_is_zero_before_any_residual():
    m = AdaptiveConformalRegressor(_Zero())
    assert m.predict_one({"x": 1.0}, return_std=True) == (0.0, 0.0)
    assert m.predict_one({"x": 1.0}, return_estimate=True).var == 0.0


class _Const:
    def __init__(self, value):
        self.value = value

    def score_one(self, x, t=None):
        return self.value


def _refuse(constant):
    raise ValueError(f"not valid JSON: {constant}")


@pytest.mark.parametrize(
    "value,text",
    [
        (float("inf"), "Infinity"),
        (float("-inf"), "-Infinity"),
        (float("nan"), "NaN"),
        (np.float32(np.inf), "Infinity"),
        (1.5, 1.5),
    ],
)
def test_call_json_is_always_valid_json(value, text):
    out = call_json(_Const(value), {"method": "score_one", "x": {"a": 1.0}})
    assert json.loads(out, parse_constant=_refuse) == {"ok": True, "result": text}


def test_call_json_non_finite_inside_a_list():
    class _Many:
        def score_one(self, x, t=None):
            return [1.0, float("inf"), float("nan")]

    out = call_json(_Many(), {"method": "score_one", "x": {"a": 1.0}})
    assert json.loads(out, parse_constant=_refuse)["result"] == [1.0, "Infinity", "NaN"]


class _Drift:
    def __init__(self):
        self.seen = []
        self.drift_detected = False

    def update(self, value, t=None):
        self.seen.append(value)
        return self


class _Sink:
    def learn_one(self, x, y=None, t=None):
        return self


def test_protected_feeds_the_drift_detector_the_smallest_key_in_any_order():
    for x in ({"b": 2.0, "a": 1.0}, {"a": 1.0, "b": 2.0}):
        det = _Drift()
        Protected(_Sink(), det).learn_one(x)
        assert det.seen == [1.0]


def test_protected_feature_selects_the_key_and_a_missing_key_raises():
    det = _Drift()
    Protected(_Sink(), det, feature="b").learn_one({"a": 1.0, "b": 2.0})
    assert det.seen == [2.0]
    with pytest.raises(KeyError):
        Protected(_Sink(), _Drift(), feature="z").learn_one({"a": 1.0})


def test_protected_drift_detector_on_an_empty_sample_gets_zero():
    det = _Drift()
    Protected(_Sink(), det).learn_one({})
    assert det.seen == [0.0]


def test_signal_is_a_dict_of_floats_and_a_matrix_is_refused():
    single = Signal(jnp.ones(2), ["a", "b"], ["m", "m"])
    assert single["a"] == 1.0
    with pytest.raises(ValueError, match="SignalBatch"):
        Signal(jnp.ones((3, 2)), ["a", "b"], ["m", "m"])


class _B(Transformer):
    """Declares a budget and a memory class, and does nothing else."""

    threshold = 1.0

    def __init__(self, budget_s=None, memory_class="O(1)"):
        if budget_s is not None:
            self.budget_s = budget_s
        self.memory_class = memory_class

    def transform_one(self, x, t=None):
        return x


def test_wrappers_declare_the_budget_of_what_they_cost():
    assert AnomalyGate(_B(1e-3)).budget_s == pytest.approx(2e-3)
    assert Protected(_B(1e-3), _B(5e-4)).budget_s == pytest.approx(1.5e-3)
    both = Protected(_B(1e-3), _B(5e-4), train_detector=True)
    assert both.budget_s == pytest.approx(2e-3)
    assert SafeEstimator(_B(1e-3), guard=lambda x: False).budget_s == pytest.approx(
        2e-3
    )
    assert UnitCheckedPipeline(
        [_B(1e-3), _B(2e-3), _B(4e-3)]
    ).budget_s == pytest.approx(1e-2)
    assert (_B(1e-3) + _B(2e-3)).budget_s == pytest.approx(3e-3)
    assert (_B(1e-3) * _B(2e-3)).budget_s == pytest.approx(3e-3)


def test_a_wrapper_with_a_part_that_declares_no_budget_declares_none():
    assert AnomalyGate(_B()).budget_s is None
    assert Protected(_B(1e-3), _B()).budget_s is None
    assert SafeEstimator(_B(), guard=lambda x: False).budget_s is None
    assert (_B(1e-3) + _B()).budget_s is None


def test_wrappers_take_the_largest_memory_class_of_their_parts():
    assert Protected(_B(1e-3), _B(1e-3)).memory_class == "O(1)"
    assert Protected(_B(1e-3), _B(1e-3, "O(window)")).memory_class == "O(window)"
    assert Protected(_B(1e-3), _B(1e-3, "O(n)")).memory_class == "unknown"
    assert (
        SafeEstimator(_B(1e-3, "O(window)"), guard=lambda x: False).memory_class
        == "O(window)"
    )


def test_realtime_pipeline_accepts_a_wrapper_and_checks_its_real_cost():
    wrapped = SafeEstimator(_B(1e-3), guard=lambda x: False)
    assert RealtimePipeline([wrapped], period_s=1e-2).total_budget_s == pytest.approx(
        2e-3
    )
    with pytest.raises(ValueError, match="exceeds"):
        RealtimePipeline([wrapped], period_s=1.5e-3)
    with pytest.raises(ValueError, match="do not declare"):
        RealtimePipeline([SafeEstimator(_B(), guard=lambda x: False)], period_s=1.0)


class _ThresholdOnly:
    threshold = 2.0

    def __init__(self, score):
        self.score = score

    def score_one(self, x, t=None):
        return self.score


def test_protected_applies_the_threshold_of_a_scoring_detector():
    below = Protected(_Sink(), _ThresholdOnly(2.0))
    above = Protected(_Sink(), _ThresholdOnly(2.1), fallback=-1.0)
    assert not below._flagged({"v": 1.0})
    assert above._flagged({"v": 1.0})
    assert above.score_one({"v": 1.0}) == float("inf")


class _Learning(_ThresholdOnly):
    def __init__(self, score):
        super().__init__(score)
        self.learned = 0

    def learn_one(self, x, t=None):
        self.learned += 1
        return self


def test_protected_trains_the_detector_only_on_unflagged_samples_and_only_if_asked():
    normal = _Learning(1.0)
    Protected(_Sink(), normal, train_detector=True).learn_one({"v": 1.0})
    assert normal.learned == 1
    flagged = _Learning(3.0)
    Protected(_Sink(), flagged, train_detector=True).learn_one({"v": 1.0})
    assert flagged.learned == 0
    default = _Learning(1.0)
    Protected(_Sink(), default).learn_one({"v": 1.0})
    assert default.learned == 0


def test_safe_estimator_predicts_then_learns_once_each():
    calls = []

    class _Model:
        def predict_one(self, x, t=None):
            calls.append("predict")
            return 1.0

        def learn_one(self, x, y=None, t=None):
            calls.append("learn")
            return self

    SafeEstimator(_Model(), guard=lambda x: False).learn_one({"a": 1.0}, 1.0)
    assert calls == ["predict", "learn"]
