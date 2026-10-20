"""Tests for the portfolio pipeline."""

from pathlib import Path

import numpy as np
import pytest
from scipy.optimize import minimize

from pipelines.finance.portfolio import (
    PortfolioPipeline,
    PortfolioResult,
    inverse_volatility_weights,
    load_panel,
    load_returns,
    min_variance_weights,
    run_on_fixtures,
)

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "test" / "fixtures" / "finance"


def _panel(seed=0, n=500, p=4):
    rng = np.random.default_rng(seed)
    a = rng.normal(size=(n, p))
    return a @ np.linalg.cholesky(
        np.array([[1.0, 0.4, 0.2, 0.1],
                  [0.4, 1.0, 0.3, 0.2],
                  [0.2, 0.3, 1.0, 0.25],
                  [0.1, 0.2, 0.25, 1.0]])).T


def test_load_returns_on_aapl():
    r = load_returns(FIXTURES / "AAPL.csv")
    assert len(r) > 100
    assert np.all(np.isfinite(r))


def test_load_returns_bad_file(tmp_path):
    p = tmp_path / "x.csv"
    p.write_text("Price,Close\nTicker,X\nDate,\n2021-01-01,-1.0\n")
    with pytest.raises(ValueError, match="not positive"):
        load_returns(p)


def test_load_panel_aligns_rows():
    panel = load_panel(FIXTURES)
    assert panel.ndim == 2
    assert panel.shape[1] == 4
    assert panel.shape[0] >= 100


def test_min_variance_weights_sum_to_one():
    cov = np.cov(_panel(), rowvar=False, ddof=1)
    w = min_variance_weights(cov)
    assert abs(float(w.sum()) - 1.0) < 1e-12


def test_min_variance_matches_numeric_minimisation():
    cov = np.cov(_panel(seed=1), rowvar=False, ddof=1)
    p = cov.shape[0]
    w_analytic = min_variance_weights(cov)
    res = minimize(lambda w: float(w @ cov @ w), np.full(p, 1.0 / p),
                   constraints={"type": "eq",
                                "fun": lambda w: float(w.sum() - 1)},
                   method="SLSQP", options={"ftol": 1e-15})
    np.testing.assert_allclose(w_analytic, res.x, atol=1e-6)


def test_inverse_volatility_weights_ordering():
    cov = np.diag([1.0, 4.0, 9.0, 16.0])
    w = inverse_volatility_weights(cov)
    assert abs(float(w.sum()) - 1.0) < 1e-12
    assert w[0] > w[1] > w[2] > w[3]


def test_inverse_volatility_rejects_negative_variance():
    with pytest.raises(ValueError, match="variance"):
        inverse_volatility_weights(np.diag([1.0, -1.0, 1.0, 1.0]))


def test_singular_covariance_raises():
    cov = np.ones((3, 3))
    with pytest.raises(ValueError, match="singular"):
        min_variance_weights(cov)


def test_pipeline_on_panel():
    panel = _panel(seed=2)
    r = PortfolioPipeline().run(panel)
    assert isinstance(r, PortfolioResult)
    assert r.n_samples == 500
    assert abs(float(r.weights_min_var.sum()) - 1.0) < 1e-12
    assert r.realised_vol_min_var <= r.realised_vol_equal + 1e-12
    assert r.turnover_min_var >= 0.0


def test_pipeline_bad_window():
    with pytest.raises(ValueError, match="window"):
        PortfolioPipeline(window=1)


def test_pipeline_bad_refit():
    with pytest.raises(ValueError, match="refit_every"):
        PortfolioPipeline(refit_every=0)


def test_pipeline_too_few_rows():
    with pytest.raises(ValueError, match="at least 2 rows"):
        PortfolioPipeline().run(np.zeros((1, 4)))


def test_run_on_fixtures_returns_result():
    r = run_on_fixtures(FIXTURES)
    assert r.n_samples > 100
    assert abs(float(r.weights_min_var.sum()) - 1.0) < 1e-12
    assert 0.0 < r.alpha <= 1.0


def test_load_returns_fewer_than_header(tmp_path):
    p = tmp_path / "short.csv"
    p.write_text("Price,Close\nTicker,X\nDate,\n")
    with pytest.raises(ValueError, match="fewer than"):
        load_returns(p)


def test_load_returns_no_close(tmp_path):
    p = tmp_path / "x.csv"
    p.write_text("Price,Open\nTicker,X\nDate,\n2021-01-01,1.0\n")
    with pytest.raises(ValueError, match="no Close column"):
        load_returns(p)


def test_load_returns_skips_non_numeric_and_short(tmp_path):
    p = tmp_path / "ok.csv"
    p.write_text(
        "Price,Close\nTicker,X\nDate,\n"
        "2021-01-01,1.0\n2021-01-02,bad\n2021-01-03\n2021-01-04,2.0\n")
    r = load_returns(p)
    assert len(r) == 1



def test_pipeline_survives_singular_window():
    rng = np.random.default_rng(0)
    panel = np.column_stack([
        rng.normal(size=200),
        rng.normal(size=200),
        rng.normal(size=200),
        np.zeros(200),
    ])
    r = PortfolioPipeline(window=20, refit_every=5).run(panel)
    assert r.n_samples == 200
