"""Tests for the drift-vs-noise pipeline."""

import numpy as np
import pytest

from pipelines.finance.drift_vs_noise import (
    DriftVsNoisePipeline,
    DriftVsNoiseResult,
    preaveraged_returns,
    raw_returns,
    synthetic_tick,
)


def test_raw_returns_is_the_difference():
    x = np.array([0.0, 1.0, 3.0, 6.0])
    np.testing.assert_allclose(raw_returns(x), [1.0, 2.0, 3.0])


def test_preaveraged_returns_window():
    x = np.arange(40, dtype=float)
    r = preaveraged_returns(x, window=10)
    assert len(r) == 2
    np.testing.assert_allclose(r[0], 10.0)
    np.testing.assert_allclose(r[1], 10.0)


def test_preaveraged_returns_ignores_remainder():
    x = np.arange(45, dtype=float)
    r = preaveraged_returns(x, window=10)
    assert len(r) == 2


def test_preaveraged_returns_rejects_bad_window():
    with pytest.raises(ValueError, match="window"):
        DriftVsNoisePipeline(window=0)


def test_pipeline_rejects_bad_warmup():
    with pytest.raises(ValueError, match="warmup"):
        DriftVsNoisePipeline(warmup=-1)


def test_synthetic_tick_gaussian():
    true, tick = synthetic_tick(n=1000, seed=0, noise_std=0.01,
                                vol=0.001, noise_kind="gaussian")
    assert len(true) == len(tick) == 1000
    assert true[0] == 0.0
    noise = tick - true
    assert abs(float(np.std(noise)) - 0.01) < 0.002


def test_synthetic_tick_student_is_heavier():
    _, g = synthetic_tick(n=5000, seed=0, noise_std=0.01,
                          vol=0.001, noise_kind="gaussian")
    _, s = synthetic_tick(n=5000, seed=0, noise_std=0.01,
                          vol=0.001, noise_kind="student", noise_df=3.0)
    kurt_g = float(np.mean((g - g.mean()) ** 4) / np.var(g) ** 2)
    kurt_s = float(np.mean((s - s.mean()) ** 4) / np.var(s) ** 2)
    assert kurt_s > kurt_g


def test_synthetic_tick_bad_noise_kind():
    with pytest.raises(ValueError, match="noise_kind"):
        synthetic_tick(n=10, seed=0, noise_kind="cauchy")


def test_synthetic_tick_with_shift():
    _, tick = synthetic_tick(n=100, seed=0, noise_std=0.0,
                             vol=0.0, drift=0.0, shift_at=50,
                             drift_after=1.0, noise_kind="gaussian")
    assert abs(float(tick[49] - tick[48])) < 1e-9
    assert abs(float(tick[50] - tick[49]) - 1.0) < 1e-9


def test_run_no_shift():
    _, tick = synthetic_tick(n=2000, seed=0, noise_std=0.01, vol=0.001)
    r = DriftVsNoisePipeline().run(tick)
    assert isinstance(r, DriftVsNoiseResult)
    assert r.change_at is None
    assert r.delay_raw_samples is None
    assert r.delay_robust_samples is None


def test_student_noise_fires_raw_more_than_robust():
    raw_total = 0
    rob_total = 0
    for s in range(5):
        _, tick = synthetic_tick(n=5000, seed=s, noise_std=0.005,
                                 vol=0.001, noise_kind="student",
                                 noise_df=3.0)
        r = DriftVsNoisePipeline().run(tick)
        raw_total += r.n_episodes_raw
        rob_total += r.n_episodes_robust
    assert raw_total > rob_total, (raw_total, rob_total)
    assert raw_total >= 1


def test_shift_detected_by_robust():
    found = 0
    for s in range(5):
        _, tick = synthetic_tick(n=5000, seed=s, noise_std=0.005,
                                 vol=0.001, shift_at=2500,
                                 drift_after=0.002, noise_kind="student",
                                 noise_df=3.0)
        r = DriftVsNoisePipeline().run(tick, change_at=2500)
        if r.delay_robust_samples is not None:
            assert r.delay_robust_samples <= 20
            found += 1
    assert found >= 3


def test_custom_cusum_kwargs_forwarded():
    _, tick = synthetic_tick(n=500, seed=0, noise_std=0.01, vol=0.001)
    r = DriftVsNoisePipeline(
        cusum_kwargs={"reference": "adaptive", "k": 0.5, "h": 20.0,
                      "reset_on_alarm": True}).run(tick)
    assert r.n_samples == 499
