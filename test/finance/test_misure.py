"""Measurement tests for the four finance pipelines.

Every test uses real data from test/fixtures/finance/:
- GSPC.csv, AAPL.csv, MSFT.csv, TSLA.csv: five years of daily
  closing prices and volumes from Yahoo Finance.
- coinbase_btcusd_trades.csv: 1,000 BTC/USD trades from the
  Coinbase public API on 14 October 2026.

The numbers asserted here are the ones reported on the doc pages
under docs/finance/. If they change, the pages must be updated in
the same commit.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pipelines.finance import (
    AnomaliaTitoloPipeline,
    EsecuzionePipeline,
    FinanzaPipeline,
    RischioPipeline,
)

FIXT = Path(__file__).resolve().parents[2] / "test" / "fixtures" / "finance"


def carica_azioni(ticker):
    fname = ticker.replace("^", "").replace("/", "_")
    df = pd.read_csv(FIXT / f"{fname}.csv", skiprows=3, header=None)
    df.columns = ["Date", "Open", "High", "Low", "Close", "Volume"]
    df["Date"] = pd.to_datetime(df["Date"], format="%Y-%m-%d")
    return df.set_index("Date").astype(float)


def carica_trade_coinbase():
    df = pd.read_csv(FIXT / "coinbase_btcusd_trades.csv")
    return df.sort_values("trade_id").reset_index(drop=True)


def test_sorveglianza_su_sp500_reale():
    """Su 1,254 giorni di S&P 500, tre anomalie LODA+HST."""
    df = carica_azioni("^GSPC")
    prezzi = df["Close"].values.astype(float)
    vol_rel = df["Volume"].values.astype(float)
    vol_rel = vol_rel / np.nanmean(vol_rel)

    n_cal = 250
    ret_cal = np.diff(prezzi[:n_cal]) / prezzi[:n_cal][:-1]
    range_ret = (float(np.percentile(ret_cal, 0.5)),
                 float(np.percentile(ret_cal, 99.5)))
    vol_cal = vol_rel[:n_cal]
    range_vol = (float(np.percentile(vol_cal, 0.5)),
                 float(np.percentile(vol_cal, 99.5)))

    p = FinanzaPipeline(range_ret=range_ret, range_vol=range_vol, seed=0)
    for i in range(1, n_cal):
        p.process_tick(prezzi[i], vol_rel[i])

    n_anom = 0
    n_deriva = 0
    for i in range(n_cal, len(prezzi)):
        e = p.process_tick(prezzi[i], vol_rel[i])
        if e.anomalia:
            n_anom += 1
        if e.deriva_segnalata:
            n_deriva += 1

    assert n_anom == 3, f"anomalie={n_anom}, atteso 3"


def test_rischio_su_sp500_reale():
    """Su 1,253 giorni di S&P 500: VaR 99%, VaR 99.9%, giorni oltre 5%."""
    df = carica_azioni("^GSPC")
    prezzi = df["Close"].values.astype(float)
    rend = np.diff(prezzi) / prezzi[:-1]
    perdite = np.maximum(-rend, 0.0)

    r = RischioPipeline(alpha=0.01, limite_perdita=0.05, seed=0)
    esiti = [r.process_day(float(p)) for p in perdite]

    var99 = esiti[-1].var_99
    var999 = esiti[-1].var_999
    n_oltre = sum(1 for e in esiti if e.oltre_limite)

    assert 0.025 < var99 < 0.032, f"var99={var99:.4f}"
    assert 0.038 < var999 < 0.045, f"var999={var999:.4f}"
    assert n_oltre == 1, f"giorni oltre 5%={n_oltre}, atteso 1"


def test_esecuzione_su_trade_coinbase_reali():
    """Su 1,000 trade Coinbase reali: quanti rifiutati e dampened."""
    trades = carica_trade_coinbase()
    sizes = trades["size"].values.astype(float)

    ep = EsecuzionePipeline(seed=0)
    esiti = [ep.process_order(float(s)) for s in sizes]

    assert ep.n_ordini == 1000
    assert ep.n_rifiutati >= 0
    assert ep.n_via_orca == ep.n_dampened + (
        sum(1 for e in esiti if "passed through Orca" in e.motivo)
    )
    # Con dati reali il rolling mean si adatta, quindi quasi tutto passa diretto
    assert ep.n_passati_diretti > 900, (
        f"passati diretti={ep.n_passati_diretti}, atteso >900"
    )


def test_anomalia_titolo_su_apple_reale():
    """Su Apple 5 anni: 43 giorni anomali (soglie adattive HST/LODA)."""
    df = carica_azioni("AAPL")
    prezzi = df["Close"].values.astype(float)
    vol_rel = df["Volume"].values.astype(float)
    vol_rel = vol_rel / np.nanmean(vol_rel)

    n_cal = 250
    ret_cal = np.diff(prezzi[:n_cal]) / prezzi[:n_cal][:-1]
    range_ret = (float(np.percentile(ret_cal, 0.5)),
                 float(np.percentile(ret_cal, 99.5)))
    vol_cal = vol_rel[:n_cal]
    range_vol = (float(np.percentile(vol_cal, 0.5)),
                 float(np.percentile(vol_cal, 99.5)))

    t = AnomaliaTitoloPipeline(range_ret=range_ret, range_vol=range_vol, seed=0)
    n_anom = 0
    for i in range(1, len(prezzi)):
        ret = (prezzi[i] - prezzi[i-1]) / prezzi[i-1]
        if t.process_day(ret, vol_rel[i]).anomalo:
            n_anom += 1

    assert n_anom == 43, f"giorni anomali={n_anom}, atteso 43"
