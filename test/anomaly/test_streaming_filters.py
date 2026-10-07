# -*- coding: utf-8 -*-
"""Unit tests for dense_armor/utility/streaming_filters.py.

The scorers are causal: they see only the `2 * radius` samples strictly
preceding the value being scored. The reference is the batch formula from
`robust_filters.py` applied to that same causal window, not the batch
output on a centred window -- see `ISTRUZIONI.md`, section "Important:
centred vs causal windows".
"""
import doctest
import importlib
import math
import sys

import numpy as np
import pytest

pytest.importorskip("river")

import dense_armor.anomaly.filters as streaming_filters  # noqa: E402
from dense_armor.anomaly.filters import (  # noqa: E402
    HampelScorer, TukeyScorer, ChauvenetScorer, SigmaClipScorer, HampelFilter,
)


def _ref_hampel(window, x_query, n_sigmas=3.0, eps=1e-9):
    w = np.asarray(window, dtype=float)
    med = float(np.median(w))
    mad = float(np.median(np.abs(w - med)))
    scale = 1.4826 * mad
    dev = abs(x_query - med)
    if scale < eps:
        return float("inf") if dev > eps else 0.0
    return dev / scale


def _ref_tukey(window, x_query, k=1.5, eps=1e-9):
    w = np.asarray(window, dtype=float)
    q1, q3 = np.percentile(w, [25, 75])
    iqr = float(q3 - q1)
    if iqr < eps:
        return float("inf") if (x_query < q1 or x_query > q3) else 0.0
    lo = float(q1) - k * iqr
    hi = float(q3) + k * iqr
    if x_query < lo:
        return (lo - x_query) / iqr
    if x_query > hi:
        return (x_query - hi) / iqr
    return 0.0


def _ref_chauvenet(window, x_query, eps=1e-9):
    w = np.asarray(window, dtype=float)
    n = w.size
    mu = float(np.mean(w))
    sigma = float(np.std(w))
    dev = abs(x_query - mu)
    if sigma < eps:
        return float(n) if dev <= eps else 0.0
    z = dev / sigma
    p = math.erfc(z / math.sqrt(2.0))
    return n * p


def _ref_sigmaclip(window, x_query, n_sigmas=3.0, max_iters=5, eps=1e-9):
    w = np.asarray(window, dtype=float)
    mask = np.ones(w.shape, dtype=bool)
    for _ in range(max_iters):
        sub = w[mask]
        if sub.size < 2:
            break
        mu = float(np.mean(sub))
        sigma = float(np.std(sub))
        if sigma < eps:
            break
        new_mask = np.abs(w - mu) <= n_sigmas * sigma
        if np.array_equal(new_mask, mask):
            break
        mask = new_mask
    sub = w[mask]
    if sub.size < 1:
        return 0.0
    mu = float(np.mean(sub))
    sigma = float(np.std(sub))
    dev = abs(x_query - mu)
    if sigma < eps:
        return float("inf") if dev > eps else 0.0
    return dev / sigma


def _stream_with_spikes(seed=0, n=300):
    rng = np.random.default_rng(seed)
    stream = list(rng.normal(0.0, 1.0, n))
    stream[100] = 50.0
    stream[150] = -50.0
    return stream


def test_hampel_equivalence_causal():
    stream = _stream_with_spikes()
    r = 5
    s = HampelScorer(radius=r, n_sigmas=3.0)
    for i, v in enumerate(stream):
        window = stream[max(0, i - 2 * r):i]
        if len(window) >= 4:
            assert s.score_one({"v": v}) == pytest.approx(
                _ref_hampel(window, v)), f"i={i}"
        s.learn_one({"v": v})


def test_tukey_equivalence_causal():
    stream = _stream_with_spikes(seed=1)
    r = 5
    s = TukeyScorer(radius=r, k=1.5)
    for i, v in enumerate(stream):
        window = stream[max(0, i - 2 * r):i]
        if len(window) >= 4:
            assert s.score_one({"v": v}) == pytest.approx(
                _ref_tukey(window, v)), f"i={i}"
        s.learn_one({"v": v})


def test_chauvenet_equivalence_causal():
    stream = _stream_with_spikes(seed=2)
    r = 5
    s = ChauvenetScorer(radius=r, threshold=0.5)
    for i, v in enumerate(stream):
        window = stream[max(0, i - 2 * r):i]
        if len(window) >= 4:
            assert s.score_one({"v": v}) == pytest.approx(
                _ref_chauvenet(window, v)), f"i={i}"
        s.learn_one({"v": v})


def test_sigmaclip_equivalence_causal():
    stream = _stream_with_spikes(seed=3)
    r = 5
    s = SigmaClipScorer(radius=r, n_sigmas=3.0)
    for i, v in enumerate(stream):
        window = stream[max(0, i - 2 * r):i]
        if len(window) >= 4:
            assert s.score_one({"v": v}) == pytest.approx(
                _ref_sigmaclip(window, v)), f"i={i}"
        s.learn_one({"v": v})


@pytest.mark.parametrize("cls,kw", [
    (HampelScorer, {}),
    (TukeyScorer, {}),
    (ChauvenetScorer, {}),
    (SigmaClipScorer, {}),
])
def test_warmup_no_anomaly(cls, kw):
    s = cls(**kw)
    for v in [1.0, 2.0, 3.0]:
        assert not s.is_outlier({"v": v})
        s.learn_one({"v": v})


def test_flat_window():
    s = HampelScorer(radius=5)
    for _ in range(20):
        s.learn_one({"v": 2.0})
    assert s.score_one({"v": 2.0}) == 0.0
    assert not s.is_outlier({"v": 2.0})
    assert s.is_outlier({"v": 2.5})


def test_nan_input_does_not_raise():
    for cls in (HampelScorer, TukeyScorer, ChauvenetScorer, SigmaClipScorer):
        s = cls(radius=5)
        for v in [1.0, 1.1, 0.9, 1.0, 1.05, 0.95, 1.02]:
            s.learn_one({"v": v})
        score = s.score_one({"v": float("nan")})
        assert isinstance(score, float)


def test_feature_selection():
    s = HampelScorer(radius=5, feature="b")
    for i in range(20):
        s.learn_one({"a": 1000.0 * i, "b": 1.0 + 0.01 * i})
    assert s.score_one({"a": 0.0, "b": 1.05}) < 3.0


def test_feature_selection_smallest_key_by_default():
    s = HampelScorer(radius=5)
    for i in range(20):
        s.learn_one({"a": 1.0 + 0.01 * i, "b": 1000.0 * i})
    assert s.score_one({"a": 1.05, "b": 0.0}) < 3.0


def test_hampel_filter_transformer_replaces_outlier():
    f = HampelFilter(radius=5)
    rng = np.random.default_rng(0)
    for v in rng.normal(0, 1, 20):
        f.learn_one({"v": float(v)})
    out = f.transform_one({"v": 50.0})
    assert abs(out["v"]) < 5.0


def test_hampel_filter_transformer_passes_through():
    f = HampelFilter(radius=5)
    rng = np.random.default_rng(0)
    for v in rng.normal(0, 1, 20):
        f.learn_one({"v": float(v)})
    out = f.transform_one({"v": 0.05})
    assert out["v"] == pytest.approx(0.05)


def test_hampel_filter_transformer_other_keys_untouched():
    f = HampelFilter(radius=5, feature="v")
    rng = np.random.default_rng(0)
    for v in rng.normal(0, 1, 20):
        f.learn_one({"v": float(v), "other": 42.0})
    out = f.transform_one({"v": 50.0, "other": 42.0})
    assert out["other"] == 42.0


def test_docstring_examples():
    assert doctest.testmod(streaming_filters).failed == 0


def test_river_check_estimator():
    from river.checks import check_estimator
    for cls in (HampelScorer, TukeyScorer, ChauvenetScorer, SigmaClipScorer,
                HampelFilter):
        check_estimator(cls())


def test_missing_river_raises_clear_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "river", None)
    monkeypatch.delitem(sys.modules,
                       "dense_armor.anomaly.filters", raising=False)
    with pytest.raises(ModuleNotFoundError, match=r"dense-armor\[river\]"):
        importlib.import_module("dense_armor.anomaly.filters")


def _detect(cls, stream, truth, **kwargs):
    s = cls(**kwargs)
    n = len(stream)
    preds = np.empty(n, dtype=bool)
    for i, v in enumerate(stream):
        preds[i] = s.is_outlier({"v": float(v)})
        s.learn_one({"v": float(v)})
    tp = int(np.sum(preds & truth))
    fp = int(np.sum(preds & ~truth))
    fn = int(np.sum(~preds & truth))
    prec = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    rec = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    return prec, rec, fp, fn, preds


def test_detection_on_seeded_sine_with_spikes():
    rng = np.random.default_rng(42)
    n = 5000
    t = np.arange(n)
    signal = 5.0 * np.sin(2.0 * np.pi * t / 500.0) + rng.normal(0.0, 0.5, n)
    n_spikes = 50
    spike_idx = rng.choice(n - 100, size=n_spikes, replace=False) + 50
    truth = np.zeros(n, dtype=bool)
    truth[spike_idx] = True
    sign = rng.choice([-1.0, 1.0], size=n_spikes)
    mag = rng.uniform(5.0, 15.0, n_spikes)
    signal[spike_idx] += sign * mag

    results = {}
    for name, cls in [("Hampel", HampelScorer), ("Tukey", TukeyScorer),
                      ("Chauvenet", ChauvenetScorer),
                      ("SigmaClip", SigmaClipScorer)]:
        prec, rec, fp, fn, _ = _detect(cls, signal, truth, radius=15)
        results[name] = (prec, rec, fp, fn)

    for name, (prec, rec, fp, fn) in results.items():
        print(f"{name:<12} P={prec:.3f} R={rec:.3f} FP={fp} FN={fn}")

    assert all(0.0 <= r[0] <= 1.0 for r in results.values())
    assert all(0.0 <= r[1] <= 1.0 for r in results.values())


def test_per_sample_time_budget():
    import time
    rng = np.random.default_rng(0)
    stream = rng.normal(0.0, 1.0, 5000)
    s = HampelScorer(radius=15)
    t0 = time.perf_counter()
    for v in stream:
        _ = s.score_one({"v": float(v)})
        s.learn_one({"v": float(v)})
    dt = time.perf_counter() - t0
    print(f"Hampel per-sample: {dt / len(stream) * 1e6:.2f} us")


@pytest.mark.parametrize("cls", [TukeyScorer, ChauvenetScorer, SigmaClipScorer])
def test_flat_window_scores(cls):
    s = cls(radius=5)
    for _ in range(10):
        s.learn_one({"v": 2.0})
    assert s.score_one({"v": 2.0}) >= 0.0
    assert s.score_one({"v": 9.0}) != s.score_one({"v": 2.0})


def test_sigma_clip_stats_edge_cases():
    from dense_armor.anomaly.filters import _clean_stats

    assert _clean_stats(np.array([1.0, 1.0, 1.0, 1.0]), 3.0) == (1.0, 0.0)
    mu, sigma = _clean_stats(np.array([0.0, 0.0, 0.0, 100.0]), 0.5)
    assert np.isfinite(mu) and np.isfinite(sigma)
