"""Base tests for the finance pipeline."""

import numpy as np
import pytest

from dense_armor.utility.finance import FinanzaPipeline


@pytest.fixture
def pipeline():
    return FinanzaPipeline(
        range_ret=(-0.05, 0.05),
        range_vol=(0.5, 2.0),
        seed=0,
    )


def test_import(pipeline):
    assert pipeline is not None
    assert pipeline.n_tick == 0


def test_one_normal_tick(pipeline):
    e = pipeline.process_tick(100.0, 1.0)
    assert e.tick == 0
    assert not e.scartato_da_armatura
    assert not e.anomalia


def test_discards_a_spike(pipeline):
    for _ in range(30):
        pipeline.process_tick(100.0, 1.0)
    e = pipeline.process_tick(9999.0, 1.0)
    assert e.scartato_da_armatura


def test_two_hundred_ticks_no_crash(pipeline):
    rng = np.random.default_rng(0)
    for _ in range(200):
        p = 100.0 + rng.normal(0, 0.5)
        v = 1.0 + rng.normal(0, 0.1)
        e = pipeline.process_tick(p, v)
        assert isinstance(e.loda_score, float)
        assert isinstance(e.hst_score, float)


def test_calibration(pipeline):
    rng = np.random.default_rng(0)
    prezzi = 100 + np.cumsum(rng.normal(0, 0.3, 500))
    vol = 1 + rng.normal(0, 0.2, 500)
    range_ret, range_vol = FinanzaPipeline.calibra(prezzi, vol, frazione=0.3)
    assert range_ret[0] < 0 < range_ret[1]
    assert range_vol[0] > 0
