"""Tests for dense_armor/utility/metrics/events.py."""
import doctest
import math
import warnings

import numpy as np
import pytest

import dense_armor.utility.metrics.events as ev_mod
from dense_armor.utility.metrics import F1
from dense_armor.utility.metrics.events import (
    EventMetrics,
    EventWindow,
    PointAdjustedF1,
)


def test_event_window_validates():
    with pytest.raises(ValueError):
        EventWindow(start=1.0, end=0.0)


def test_event_window_contains():
    w = EventWindow(1.0, 3.0)
    assert w.contains(1.0)
    assert w.contains(2.0)
    assert w.contains(3.0)
    assert not w.contains(0.9)
    assert not w.contains(3.1)


def test_metrics_requires_at_least_one_event():
    with pytest.raises(ValueError):
        EventMetrics([])


def test_metrics_rejects_bad_bias():
    with pytest.raises(ValueError):
        EventMetrics([(0.0, 1.0)], positional_bias="weird")


def test_detection_delay_simple():
    m = EventMetrics([(10.0, 20.0)])
    m.add_alarm(12.5)
    delays = m.detection_delays_s()
    assert delays == [pytest.approx(2.5)]


def test_missed_event():
    m = EventMetrics([(10.0, 20.0), (30.0, 40.0)])
    m.add_alarm(12.5)
    assert m.missed_events() == 1
    assert m.detection_delays_s()[0] == pytest.approx(2.5)
    assert m.detection_delays_s()[1] is None


def test_alarm_after_window_is_a_miss():
    m = EventMetrics([(10.0, 20.0)])
    m.add_alarm(21.0)
    assert m.missed_events() == 1
    assert m.false_alarms() == 1


def test_false_alarms_per_hour_with_explicit_normal_time():
    m = EventMetrics([(10.0, 20.0)], normal_time_s=1800.0)
    m.add_alarm(15.0)
    m.add_alarm(100.0)
    m.add_alarm(200.0)
    assert m.false_alarms() == 2
    assert m.false_alarms_per_hour() == pytest.approx(4.0)


def test_false_alarms_per_hour_inferred():
    m = EventMetrics([(10.0, 20.0)])
    for t in [10.0, 100.0, 200.0, 300.0]:
        m.add_alarm(t)
    fa_per_h = m.false_alarms_per_hour()
    assert fa_per_h > 0.0
    assert math.isfinite(fa_per_h)


def test_range_recall_perfect():
    m = EventMetrics(
        [(0.0, 10.0)], positional_bias="flat", alpha=0.0,
        threshold_window=1.0,
    )
    for t in range(11):
        m.add_alarm(float(t))
    assert m.range_recall() == pytest.approx(1.0)
    assert m.range_precision() == pytest.approx(1.0)
    assert m.range_f1() == pytest.approx(1.0)


def test_range_recall_penalises_fragmentation():
    m = EventMetrics(
        [(0.0, 10.0)], positional_bias="flat", alpha=0.0,
        threshold_window=0.0,
    )
    for t in range(11):
        m.add_alarm(float(t))
    assert m.range_recall() == pytest.approx(1.0 / 11.0)


def test_range_recall_partial():
    m = EventMetrics([(0.0, 10.0)], positional_bias="flat", alpha=0.0)
    m.add_alarm(2.0)
    m.add_alarm(3.0)
    m.add_alarm(4.0)
    r = m.range_recall()
    assert 0.0 < r < 1.0


def test_range_precision_with_false_alarm():
    m = EventMetrics([(0.0, 5.0)], positional_bias="flat", alpha=0.0)
    m.add_alarm(1.0)
    m.add_alarm(50.0)
    p = m.range_precision()
    assert 0.0 < p < 1.0


def test_cardinality_factor_penalises_fragmentation():
    m_whole = EventMetrics([(0.0, 10.0)], positional_bias="flat", alpha=0.0,
                           cardinality=True, threshold_window=0.0)
    m_frag = EventMetrics([(0.0, 10.0)], positional_bias="flat", alpha=0.0,
                          cardinality=True, threshold_window=0.0)
    for t in [1.0, 3.0, 5.0, 7.0, 9.0]:
        m_whole.add_alarm(t)
    for t in [1.0, 3.0]:
        m_frag.add_alarm(t)
    assert m_frag.range_recall() <= m_whole.range_recall()


def test_nab_score_rewards_earlier_detection():
    early = EventMetrics([(0.0, 10.0)])
    late = EventMetrics([(0.0, 10.0)])
    early.add_alarm(0.5)
    late.add_alarm(9.5)
    assert early.nab_score() > late.nab_score()


def test_nab_perfect_and_null_detectors():
    perfect = EventMetrics([(0.0, 10.0), (50.0, 60.0)])
    perfect.add_alarm(0.0)
    perfect.add_alarm(50.0)
    null = EventMetrics([(0.0, 10.0), (50.0, 60.0)])
    assert perfect.nab_score() == pytest.approx(100.0)
    assert null.nab_score() == pytest.approx(0.0)


def test_nab_scoring_follows_figure_3():
    m = EventMetrics([(0.0, 10.0)])
    for t in (-5.0, 0.0, 5.0, 14.5, 100.0):
        m.add_alarm(t)
    s = ev_mod._nab_sigmoid
    raw = -0.11 + s(-1.0) - 0.11 * 0.8093 - 0.11 * 1.0
    assert s(0.45) == pytest.approx(-0.8093, abs=1e-4)
    assert s(9.0) == pytest.approx(-1.0, abs=1e-9)
    expected = 100.0 * (raw + 1.0) / (s(-1.0) + 1.0)
    assert m.nab_score() == pytest.approx(expected, abs=1e-3)


def test_nab_weights_bounded():
    with pytest.raises(ValueError):
        EventMetrics([(0.0, 1.0)], a_fn=-2.0)
    with pytest.raises(ValueError):
        EventMetrics([(0.0, 1.0)], a_tp=1.5)


def test_report_shape():
    m = EventMetrics([(0.0, 10.0)])
    m.add_alarm(1.0)
    m.add_alarm(50.0)
    assert set(m.report()) == {
        "detection_delays_s", "mean_delay_s", "missed", "false_alarms",
        "false_alarms_per_hour", "range_precision", "range_recall",
        "range_f1", "nab_score",
    }


def test_positional_bias_values_figure_2b():
    b = ev_mod._PositionalBias
    assert [b.front(i, 4) for i in range(1, 5)] == [4, 3, 2, 1]
    assert [b.back(i, 4) for i in range(1, 5)] == [1, 2, 3, 4]
    assert [b.middle(i, 4) for i in range(1, 5)] == [1, 2, 2, 1]
    assert [b.flat(i, 4) for i in range(1, 5)] == [1, 1, 1, 1]


def test_window_points_use_sample_period():
    assert EventWindow(0.0, 0.5).n_points(0.01) == 51


def test_precision_uses_flat_bias():
    front = EventMetrics([(0.0, 0.09)], dt=0.01, positional_bias="front")
    back = EventMetrics([(0.0, 0.09)], dt=0.01, positional_bias="back")
    for m in (front, back):
        m.add_alarm(0.0)
    assert front.range_precision() == pytest.approx(back.range_precision())
    assert front.range_recall() > back.range_recall()


def test_point_adjusted_f1_inflates_a_random_score():
    rng = np.random.default_rng(0)
    y = np.zeros(2000, dtype=int)
    for s in (200, 700, 1200, 1700):
        y[s:s + 100] = 1
    p = (rng.random(2000) < 0.05).astype(int)
    pa = PointAdjustedF1(positive=1, warn=False)
    f1 = F1(positive=1)
    for a, b in zip(y, p):
        pa.update(int(a), int(b))
        f1.update(int(a), int(b))
    assert pa.get() > 0.8
    assert f1.get() < 0.1


def test_positional_bias_front_versus_back():
    front = EventMetrics([(0.0, 10.0)], positional_bias="front", alpha=0.0)
    back = EventMetrics([(0.0, 10.0)], positional_bias="back", alpha=0.0)
    for m in (front, back):
        for t in [0.0, 1.0]:
            m.add_alarm(t)
    assert front.range_recall() >= back.range_recall()


def test_point_adjusted_f1_matches_plain_f1_when_perfect():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = PointAdjustedF1(positive=1, warn=False)
        for a, b in [(0, 0), (1, 1), (0, 0), (1, 1)]:
            m.update(a, b)
    assert m.get() == pytest.approx(1.0)


def test_point_adjusted_f1_no_detections_is_zero():
    m = PointAdjustedF1(positive=1, warn=False)
    y_true = [0, 0, 1, 1, 0, 0]
    y_pred = [0] * len(y_true)
    for a, b in zip(y_true, y_pred):
        m.update(a, b)
    assert m.get() == 0.0


def test_point_adjusted_f1_warns_by_default():
    m = PointAdjustedF1(positive=1)
    m.update(1, 1)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        _ = m.get()
        assert any("point-adjusted" in str(x.message) for x in w)


def test_doctests():
    assert doctest.testmod(ev_mod).failed == 0
