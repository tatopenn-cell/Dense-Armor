"""Unit tests for dense_armor/utility/river_anomaly.py."""
import doctest
import importlib
import sys

import numpy as np
import pytest

import dense_armor.utility.anomaly.deviation as river_anomaly  # noqa: E402
from dense_armor.utility.anomaly.deviation import StreamingDeviationScorer  # noqa: E402
from dense_armor.utility.anomaly.streaming import StreamingDeviationDetector  # noqa: E402


def test_score_matches_streaming_detector_flags():
    rng = np.random.default_rng(0)
    stream = list(rng.normal(0, 1, 300)) + [8.0, -9.0] + list(rng.normal(0, 1, 50))
    det = StreamingDeviationDetector(radius=5, ref_mult=2, n_sigmas=3.0)
    scorer = StreamingDeviationScorer(radius=5, ref_mult=2)
    for v in stream:
        flag = det.update(v)
        assert (scorer.score_one({"v": v}) > 3.0) == flag
        scorer.learn_one({"v": v})


def test_warmup_scores_zero():
    m = StreamingDeviationScorer()
    for v in [1.0, 2.0, 3.0]:
        assert m.score_one({"v": v}) == 0.0
        m.learn_one({"v": v})


def test_flat_window_uses_absolute_deviation():
    m = StreamingDeviationScorer()
    for _ in range(10):
        m.learn_one({"v": 2.0})
    assert m.score_one({"v": 2.5}) == pytest.approx(0.5)


def test_feature_selection():
    m = StreamingDeviationScorer(feature="b")
    for v in range(10):
        m.learn_one({"a": 1000.0 * v, "b": 1.0 + 0.01 * v})
    assert m.score_one({"a": 0.0, "b": 1.05}) < 3.0


def test_docstring_example():
    assert doctest.testmod(river_anomaly).failed == 0


def test_check_estimator():
    from dense_armor.checks import check_estimator

    check_estimator(StreamingDeviationScorer())


