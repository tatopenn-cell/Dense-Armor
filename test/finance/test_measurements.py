"""Measurement tests for the finance pipelines, on real data.

Every test reads a CSV under test/fixtures/finance/ with the standard
library csv reader (the same pattern used by
dense_armor.utility.datasets.casper), and runs one pipeline on it.

The numbers asserted here are the ones reported on the doc pages
under docs/finance/. If they change, the pages must be updated in
the same commit.
"""

import csv
from pathlib import Path

import numpy as np
import pytest

from pipelines.finance import (
    ExecutionPipeline,
    FinancePipeline,
    OnlineAnalysis,
    RiskPipeline,
    SingleTickerPipeline,
    StaticAnalysis,
)

# Tutti i test di questo modulo leggono CSV da disco e fanno girare una
# pipeline su centinaia o migliaia di campioni reali. Sono i "benchmark",
# non i test unitari. Si marcano `slow` cosi' in locale basta
#     pytest -m "not slow"
# per girare solo i test veloci. La CI li fa girare tutti, perche' e' da
# loro che viene la copertura di OnlineAnalysis, StaticAnalysis e del
# metodo calibra.
pytestmark = pytest.mark.slow

FIXT = Path(__file__).resolve().parents[2] / "test" / "fixtures" / "finance"


def _leggi_azioni(ticker):
    """Read a Yahoo-Finance CSV with three header rows (Price, Ticker, Date)."""
    fname = ticker.replace("^", "").replace("/", "_")
    close = []
    volume = []
    with (FIXT / f"{fname}.csv").open() as fh:
        reader = csv.reader(fh)
        # skip the three header rows yfinance writes
        next(reader)
        next(reader)
        next(reader)
        for row in reader:
            if not row or not row[0]:
                continue
            close.append(float(row[4]))
            volume.append(float(row[5]))
    return np.array(close), np.array(volume)


def _leggi_trade_coinbase():
    """Read the Coinbase trades CSV (header, then trade_id,side,size,price,time)."""
    sizes = []
    with (FIXT / "coinbase_btcusd_trades.csv").open() as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            sizes.append(float(row["size"]))
    return np.array(sizes)


def test_surveillance_on_real_sp500():
    """On 1,254 days of S&P 500 with calibrated ranges, 42 LODA+HST
    anomalies, including quarterly-expiration Fridays and month-end.
    See docs/finance/static_vs_online.md."""
    prezzi, vol = _leggi_azioni("^GSPC")
    vol_rel = vol / np.nanmean(vol)

    n_cal = 250
    ret_cal = np.diff(prezzi[:n_cal]) / prezzi[:n_cal][:-1]
    range_ret = (
        float(np.percentile(ret_cal, 0.5)),
        float(np.percentile(ret_cal, 99.5)),
    )
    vol_cal = vol_rel[:n_cal]
    range_vol = (
        float(np.percentile(vol_cal, 0.5)),
        float(np.percentile(vol_cal, 99.5)),
    )

    p = FinancePipeline(range_ret=range_ret, range_vol=range_vol, seed=0)
    for i in range(1, n_cal):
        p.process_tick(float(prezzi[i]), float(vol_rel[i]))

    n_anom = 0
    for i in range(n_cal, len(prezzi)):
        e = p.process_tick(float(prezzi[i]), float(vol_rel[i]))
        if e.anomalous:
            n_anom += 1

    assert n_anom == 42, f"anomalies={n_anom}, expected 42"


def test_risk_on_real_sp500():
    """On 1,253 days of S&P 500: VaR 99%, VaR 99.9%, days past 5%."""
    prezzi, _ = _leggi_azioni("^GSPC")
    rend = np.diff(prezzi) / prezzi[:-1]
    perdite = np.maximum(-rend, 0.0)

    r = RiskPipeline(alpha=0.01, loss_limit=0.05, seed=0)
    esiti = [r.process_day(float(p)) for p in perdite]

    var99 = esiti[-1].var_99
    var999 = esiti[-1].var_999
    n_over = sum(1 for e in esiti if e.over_limit)

    assert 0.025 < var99 < 0.032, f"var99={var99:.4f}"
    assert 0.038 < var999 < 0.045, f"var999={var999:.4f}"
    assert n_over == 1, f"days past 5%={n_over}, expected 1"


def test_execution_on_real_coinbase_trades():
    """On 1,000 real BTC/USD Coinbase trades: most pass, some refused."""
    sizes = _leggi_trade_coinbase()
    ep = ExecutionPipeline(seed=0)
    esiti = [ep.process_order(float(s)) for s in sizes]

    assert ep.n_orders == 1000
    assert ep.n_refused >= 15, (
        f"refused={ep.n_refused}, expected at least 15 on the real distribution"
    )
    assert ep.n_passed_directly > 900, (
        f"passed direct={ep.n_passed_directly}, expected >900"
    )
    # ogni ordine e' o approvato o rifiutato
    n_ok = sum(1 for e in esiti if e.approved)
    assert n_ok + ep.n_refused == 1000


def test_single_ticker_on_real_apple():
    """On Apple over 5 years, 43 anomalous days (HST/LODA adaptive)."""
    prezzi, vol = _leggi_azioni("AAPL")
    vol_rel = vol / np.nanmean(vol)

    n_cal = 250
    ret_cal = np.diff(prezzi[:n_cal]) / prezzi[:n_cal][:-1]
    range_ret = (
        float(np.percentile(ret_cal, 0.5)),
        float(np.percentile(ret_cal, 99.5)),
    )
    vol_cal = vol_rel[:n_cal]
    range_vol = (
        float(np.percentile(vol_cal, 0.5)),
        float(np.percentile(vol_cal, 99.5)),
    )

    t = SingleTickerPipeline(range_ret=range_ret, range_vol=range_vol, seed=0)
    n_anom = 0
    for i in range(1, len(prezzi)):
        ret = (prezzi[i] - prezzi[i - 1]) / prezzi[i - 1]
        if t.process_day(float(ret), float(vol_rel[i])).anomalous:
            n_anom += 1

    assert n_anom == 43, f"anomalies={n_anom}, expected 43"


def test_static_analysis_on_real_sp500():
    """StaticAnalysis with a 3% fixed threshold on the S&P 500.

    Reads the same GSPC fixture, applies the fixed-cutoff baseline,
    and checks the number of flagged days matches the
    static_vs_online page (20 days over the whole series).
    """
    prices, _volume = _leggi_azioni("^GSPC")
    returns = [
        (prices[i] - prices[i - 1]) / prices[i - 1] for i in range(1, len(prices))
    ]

    sa = StaticAnalysis(threshold=0.03)
    r = sa.analyze(np.array(returns))

    assert r.n_days == len(returns)
    assert r.n_anomalous == 20, f"static 3% flag = {r.n_anomalous}"


def test_online_analysis_on_real_sp500():
    """OnlineAnalysis on the same S&P 500 series, both detectors.

    Checks that the pipeline runs end to end and returns the
    structure promised by the class docstring.
    """
    prices, volume = _leggi_azioni("^GSPC")
    vol_mean = float(np.mean(volume))
    relative_volume = [v / vol_mean for v in volume]
    returns = [0.0] + [
        (prices[i] - prices[i - 1]) / prices[i - 1] for i in range(1, len(prices))
    ]

    n_cal = 250
    ret_cal = np.array(returns[1:n_cal])
    range_ret = (
        float(np.percentile(ret_cal, 0.5)),
        float(np.percentile(ret_cal, 99.5)),
    )
    vol_cal = np.array(relative_volume[:n_cal])
    range_vol = (
        float(np.percentile(vol_cal, 0.5)),
        float(np.percentile(vol_cal, 99.5)),
    )

    oa = OnlineAnalysis(range_ret=range_ret, range_vol=range_vol, seed=0)
    r = oa.analyze(returns, relative_volume)

    assert r.n_days == len(returns)
    assert isinstance(r.n_anomalous, int)
    assert isinstance(r.final_loda_threshold, float)
    assert isinstance(r.final_hst_threshold, float)


def test_finance_pipeline_calibra():
    """Static method calibra returns ranges from the first fraction of a series."""
    prices, volume = _leggi_azioni("^GSPC")
    vol_mean = float(np.mean(volume))
    relative_volume = np.array([v / vol_mean for v in volume])
    prices = np.array(prices)

    range_ret, range_vol = FinancePipeline.calibra(
        prices, relative_volume, fraction=0.2
    )
    low, high = range_ret
    assert low < 0 < high
    assert range_vol[0] > 0
    assert range_vol[1] > range_vol[0]


def test_execution_dampened_branch():
    """Exercise the middle branch of the ExecutionPipeline (Orca on a 6x order).

    Warms the rolling mean to a known value, then feeds one order big
    enough to go through Orca but below the 10x refusal threshold.
    The purpose is to cover the Orca path (which the tests on real
    data alone never reach).
    """
    ep = ExecutionPipeline(seed=0)
    for _ in range(30):
        ep.process_order(5.0)
    # rolling mean ~ 5, so 6x = 30 goes through Orca but not refused
    e = ep.process_order(30.0)
    assert e.approved
    assert ep.n_through_orca >= 1


def test_execution_refusal_by_count():
    """Confirm the refusal counter and the direct-pass counter are consistent."""
    ep = ExecutionPipeline(seed=0)
    for _ in range(30):
        ep.process_order(5.0)
    ep.process_order(500.0)  # refused
    ep.process_order(6.0)  # passes directly
    assert ep.n_orders == 32
    assert ep.n_refused == 1
    assert ep.n_passed_directly >= 29


def test_fraud_pipeline_on_openml_creditcard():
    """FraudPipeline on the OpenML creditcard dataset (subsampled).

    The full dataset is 284,807 rows with 492 frauds (0.17%). It is
    subsampled to 50,000 rows keeping every fraud; 40,000 rows train,
    10,000 test. The two online trees reach 0.969 (Mondrian) and 0.926
    (SGT) ROC-AUC. At threshold 0.5 the ensemble fires 72 alerts,
    69 true positives and 3 false positives.
    """
    try:
        from sklearn.datasets import fetch_openml
    except ImportError:
        pytest.skip("scikit-learn not installed")
    from sklearn.metrics import roc_auc_score

    from pipelines.finance.fraud import FraudPipeline

    d = fetch_openml("creditcard", version=1, as_frame=True, parser="auto")
    X_full = d.data.astype(float).values
    y_full = d.target.astype(int).values

    rng = np.random.default_rng(0)
    idx_pos = np.where(y_full == 1)[0]
    idx_neg = np.where(y_full == 0)[0]
    n_neg = 50000 - len(idx_pos)
    idx = np.concatenate([idx_pos, rng.choice(idx_neg, size=n_neg, replace=False)])
    rng.shuffle(idx)
    X, y = X_full[idx], y_full[idx]

    n_train = int(len(X) * 0.8)
    X_train, y_train = X[:n_train], y[:n_train]
    X_test, y_test = X[n_train:], y[n_train:]
    nf = X_train.shape[1]

    p = FraudPipeline(n_trees=10, min_samples_split_sgt=50, threshold=0.5, seed=0)
    for i in range(len(X_train)):
        x_dict = {f"f{j}": float(X_train[i, j]) for j in range(nf)}
        p.learn_one(x_dict, int(y_train[i]))

    s_m, s_e, y_pred = [], [], []
    for i in range(len(X_test)):
        x_dict = {f"f{j}": float(X_test[i, j]) for j in range(nf)}
        r = p.score_one(x_dict)
        s_m.append(r.score_mondrian)
        s_e.append(r.score_ensemble)
        y_pred.append(1 if r.flagged else 0)

    auc_m = roc_auc_score(y_test, s_m)
    auc_e = roc_auc_score(y_test, s_e)
    tp = sum(1 for i in range(len(y_test)) if y_test[i] == 1 and y_pred[i] == 1)
    fp = sum(1 for i in range(len(y_test)) if y_test[i] == 0 and y_pred[i] == 1)

    assert 0.960 < auc_m < 0.975, f"Mondrian AUC = {auc_m:.3f}"
    assert 0.960 < auc_e < 0.975, f"ensemble AUC = {auc_e:.3f}"
    assert tp == 69, f"true positives = {tp}"
    assert fp == 3, f"false positives = {fp}"

