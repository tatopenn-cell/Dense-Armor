import doctest

import numpy as np
import pytest

from dense_armor.utility.signals.audio import FIR, DelayLine, convolve
from dense_armor.utility.signals.audio import fir as fir_mod


def test_convolve_matches_numpy_and_hand_case():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(50)
    h = rng.standard_normal(7)
    assert np.allclose(convolve(x, h), np.convolve(x, h))
    assert convolve([1.0, 2.0, 3.0], [1.0, 1.0]).tolist() == [1.0, 3.0, 5.0, 3.0]
    assert np.allclose(convolve(x, h), convolve(h, x))
    for a, b in (([], [1.0]), ([1.0], [])):
        with pytest.raises(ValueError):
            convolve(a, b)


def test_fir_impulse_response_block_sample_and_state():
    h = np.array([0.5, -0.25, 0.125, 1.0])
    f = FIR(h)
    assert np.array_equal(f.update_many(np.r_[1.0, np.zeros(5)])[:4], h)
    x = np.random.default_rng(1).standard_normal(300)
    whole = FIR(h).update_many(x)
    assert np.allclose(whole, np.convolve(x, h)[: len(x)])
    g = FIR(h)
    parts = np.concatenate([g.update_many(x[:120]), g.update_many(x[120:])])
    assert np.array_equal(whole, parts)
    k = FIR(h)
    assert np.array_equal(np.array([k.update(v).out for v in x]), whole)
    assert f.clone().get_params()["h"].shape == (4,)
    assert f.budget_s > 0 and f.memory_class
    with pytest.raises(ValueError):
        FIR([])


def test_delay_line():
    d = DelayLine(3)
    x = np.arange(1.0, 8.0)
    assert d.update_many(x).tolist() == [0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0]
    long = np.random.default_rng(2).standard_normal(3000)
    y = DelayLine(50).update_many(long)
    assert np.array_equal(y[50:], long[:-50]) and not y[:50].any()
    with pytest.raises(ValueError):
        DelayLine(0)


def test_doctests():
    assert doctest.testmod(fir_mod).failed == 0
