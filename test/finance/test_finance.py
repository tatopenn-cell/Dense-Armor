"""Tests for the four finance pipelines."""

import numpy as np
import pytest

from pipelines.finance import (
    ExecutionPipeline,
    FinancePipeline,
    RiskPipeline,
    SingleTickerPipeline,
)


@pytest.fixture
def pipeline():
    return FinancePipeline(range_ret=(-0.05, 0.05), range_vol=(0.5, 2.0), seed=0)


def test_finanza_import(pipeline):
    assert pipeline is not None
    assert pipeline.n_tick == 0


def test_finanza_un_tick_normale(pipeline):
    e = pipeline.process_tick(100.0, 1.0)
    assert e.tick == 0
    assert not e.discarded_by_armatura
    assert not e.anomalia


def test_finanza_scarta_picco(pipeline):
    for _ in range(30):
        pipeline.process_tick(100.0, 1.0)
    e = pipeline.process_tick(9999.0, 1.0)
    assert e.discarded_by_armatura


def test_esecuzione_ordine_piccolo():
    ep = ExecutionPipeline(seed=0)
    e = ep.process_order(1.0)
    assert e.approved
    assert e.reduction == 0.0


def test_esecuzione_rifiuta_ordine_enorme():
    ep = ExecutionPipeline(seed=0)
    for _ in range(30):
        ep.process_order(5.0)
    e = ep.process_order(5000.0)
    assert not e.approved
    assert "max_size_multiple" in e.reason


def test_rischio_var_cresce():
    rp = RiskPipeline(alpha=0.01, loss_limit=0.05, seed=0)
    rng = np.random.default_rng(0)
    for _ in range(200):
        rp.process_day(float(abs(rng.normal(0.01, 0.005))))
    e = rp.process_day(0.02)
    assert e.var_99 > 0
    assert e.var_999 >= e.var_99


def test_rischio_giorno_grande_segnalato():
    rp = RiskPipeline(alpha=0.01, loss_limit=0.05, seed=0)
    for _ in range(50):
        rp.process_day(0.005)
    e = rp.process_day(0.10)
    assert e.over_limit


def test_titolo_un_giorno():
    t = SingleTickerPipeline(range_ret=(-0.05, 0.05), range_vol=(0.5, 2.0), seed=0)
    e = t.process_day(0.001, 1.0)
    assert e.day == 0
    assert not e.anomalous


def test_titolo_200_giorni_non_crasha():
    t = SingleTickerPipeline(range_ret=(-0.05, 0.05), range_vol=(0.5, 2.0), seed=0)
    rng = np.random.default_rng(0)
    for _ in range(200):
        e = t.process_day(float(rng.normal(0, 0.01)), float(1.0 + rng.normal(0, 0.1)))
        assert isinstance(e.anomalous, bool)
