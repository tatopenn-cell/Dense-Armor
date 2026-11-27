"""Regression tests for the fixes of the review: non-finite values and Orca state."""

import logging

import numpy as np

from dense_armor.core.online.anomaly.streaming import streamDD
from dense_armor.core.online.protect.arbiter import classify_segments, route_and_correct
from dense_armor.core.online.protect.healing import healing_filter
from dense_armor.core.online.protect.orca import Orca

NAN_SERIES = np.array([1.0] * 20 + [np.nan] + [1.0] * 2 + [9.0] + [1.0] * 10)


def test_nonfinite_point_is_clean_and_a_later_spike_is_found():
    labels, _, _ = classify_segments(NAN_SERIES, radius=3, ref_mult=2)
    assert labels[20] == "clean"
    assert labels[23] == "spike"


def test_batch_and_streaming_flags_agree_with_a_nan():
    labels, _, _ = classify_segments(NAN_SERIES, radius=3, ref_mult=2)
    det = streamDD(radius=3, ref_mult=2)
    streaming = [bool(det.update(v)) for v in NAN_SERIES]
    assert streaming == [lab != "clean" for lab in labels]


def test_route_and_correct_corrects_with_the_finite_median():
    pulito, _, _ = route_and_correct(NAN_SERIES, radius=3, ref_mult=2)
    assert abs(pulito[23] - 1.0) < 1e-9
    assert np.isnan(pulito[20])


def test_run_followed_only_by_nan_is_a_regime():
    x = np.array([1.0] * 40 + [9.0, 9.0, 9.0] + [np.nan] * 10)
    labels, _, _ = classify_segments(x, radius=3, ref_mult=2)
    assert list(labels[40:43]) == ["regime"] * 3


def test_healing_filter_replaces_a_nan():
    out = healing_filter(np.r_[np.ones(5), np.nan, np.ones(30)])
    assert np.all(np.isfinite(out))
    assert out[5] == 1.0


def test_orca_hard_clamp_threshold_is_a_parameter():
    assert Orca(hard_clamp_threshold=0.01).hard_clamp_threshold == 0.01
    assert Orca().hard_clamp_threshold == 0.05


def test_orca_resets_the_arbiter_results_at_every_call():
    logging.disable(logging.CRITICAL)
    try:
        orca = Orca()
        x = 120.0 + np.random.default_rng(0).normal(0, 0.01, 40)
        x[20] = 500.0
        orca.protect_and_forward(None, x, use_arbiter=True, use_model_injection=False)
        assert orca.etichette_arbitro is not None
        orca.protect_and_forward(None, x[:10], use_arbiter=False, use_model_injection=False)
        assert orca.etichette_arbitro is None
        assert orca.incertezza_arbitro is None
        assert orca.incertezza_arbitro_media == 0.0
    finally:
        logging.disable(logging.NOTSET)
