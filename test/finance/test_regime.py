"""Tests for the regime pipeline (BOCPD on daily returns)."""

from pathlib import Path

import numpy as np
import pytest

from pipelines.finance.regime import (
    RegimePipeline,
    RegimeResult,
    load_log_returns,
    run_pipeline,
    run_synthetic,
    synthetic_two_regime,
)

ROOT = Path(__file__).resolve().parents[2]
GSPC = ROOT / "test" / "fixtures" / "finance" / "GSPC.csv"


def test_load_log_returns_on_gspc():
    r = load_log_returns(GSPC)
    assert r.ndim == 1
    assert len(r) == 1253
    assert np.all(np.isfinite(r))
    assert abs(float(np.mean(r))) < 0.01


def test_load_log_returns_missing_file(tmp_path):
    p = tmp_path / "empty.csv"
    p.write_text("Price,Close\n")
    with pytest.raises(ValueError, match="fewer than"):
        load_log_returns(p)


def test_load_log_returns_no_close(tmp_path):
    p = tmp_path / "bad.csv"
    p.write_text("Price,Open\nTicker,X\nDate,\n2021-01-01,1.0\n")
    with pytest.raises(ValueError, match="no Close column"):
        load_log_returns(p)


def test_load_log_returns_non_positive_close(tmp_path):
    p = tmp_path / "neg.csv"
    p.write_text(
        "Price,Close\nTicker,X\nDate,\n"
        "2021-01-01,-1.0\n2021-01-02,2.0\n")
    with pytest.raises(ValueError, match="not positive"):
        load_log_returns(p)


def test_load_log_returns_skips_short_rows(tmp_path):
    p = tmp_path / "ok.csv"
    p.write_text(
        "Price,Close\nTicker,X\nDate,\n"
        "2021-01-01,1.0\n2021-01-02\n2021-01-03,2.0\n")
    r = load_log_returns(p)
    assert len(r) == 1
    assert abs(float(r[0]) - np.log(2.0)) < 1e-12


def test_load_log_returns_skips_non_numeric_close(tmp_path):
    p = tmp_path / "ok.csv"
    p.write_text(
        "Price,Close\nTicker,X\nDate,\n"
        "2021-01-01,1.0\n2021-01-02,bad\n2021-01-03,2.0\n")
    r = load_log_returns(p)
    assert len(r) == 1


def test_synthetic_two_regime_is_deterministic():
    x1, c1 = synthetic_two_regime(n=100, seed=0, change_at=50)
    x2, c2 = synthetic_two_regime(n=100, seed=0, change_at=50)
    np.testing.assert_array_equal(x1, x2)
    assert c1 == c2 == 50


def test_synthetic_two_regime_has_different_vol():
    x, ch = synthetic_two_regime(n=1000, seed=0, change_at=500,
                                 vol_low=0.01, vol_high=0.03)
    v_low = float(np.std(x[:ch]))
    v_high = float(np.std(x[ch:]))
    assert v_high > 2.0 * v_low


def test_run_on_gspc_finds_some_alarm():
    r = run_pipeline(GSPC)
    assert isinstance(r, RegimeResult)
    assert r.n_returns == 1253
    assert r.n_alarms >= 1
    assert r.change_at is None
    assert r.delay_samples is None
    assert r.years_of_calm > 0
    assert r.alarm_rate_per_year >= 0.0


def test_run_on_synthetic_detects_change_near_500():
    x, ch = synthetic_two_regime(n=1000, seed=0, change_at=500)
    pipe = RegimePipeline()
    r = pipe.run(x, change_at=ch)
    assert r.change_at == 500
    assert r.first_alarm_after_change is not None
    assert r.delay_samples is not None
    assert r.delay_samples <= 10
    assert r.n_alarms >= 1


def test_run_synthetic_summary():
    out = run_synthetic(n_seeds=5, n=500, change_at=250)
    assert out["n_seeds"] == 5
    assert len(out["per_seed"]) == 5
    assert out["delay_median"] is not None
    assert out["delay_iqr"] is not None
    assert out["rate_median"] is not None
    assert out["n_with_detection"] <= 5


def test_run_synthetic_no_detection():
    x, ch = synthetic_two_regime(n=100, seed=0, change_at=50,
                                 vol_low=0.01, vol_high=0.01)
    pipe = RegimePipeline(lam=1e6, detection_threshold=0.99)
    r = pipe.run(x, change_at=ch)
    assert r.n_alarms == 0
    assert r.first_alarm_after_change is None
    assert r.delay_samples is None


def test_bad_warmup():
    with pytest.raises(ValueError, match="warmup"):
        RegimePipeline(warmup=-1)


def test_n_calm_zero_gives_nan_rate():
    pipe = RegimePipeline()
    r = pipe.run(np.zeros(10), change_at=0)
    assert np.isnan(r.alarm_rate_per_year)


def test_warmup_suppresses_early_alarms():
    x, ch = synthetic_two_regime(n=200, seed=0, change_at=100)
    pipe = RegimePipeline(warmup=50)
    r = pipe.run(x, change_at=ch)
    assert all(a >= 50 for a in r.alarms)
