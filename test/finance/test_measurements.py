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

from pipelines.finance import (
    ExecutionPipeline,
    FinancePipeline,
    RiskPipeline,
    SingleTickerPipeline,
)

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
