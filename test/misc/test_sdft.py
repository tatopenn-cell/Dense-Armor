"""Tests for the sliding DFT (Jacobsen and Lyons 2003)."""

import doctest

import numpy as np
import pytest

from dense_armor.utility.misc import SDFT
from dense_armor.utility.misc import sdft as sdft_module


def test_matches_numpy_fft_on_every_window():
    """The SDFT of the last N samples equals numpy.fft.fft on the same N."""
    rng = np.random.default_rng(0)
    N = 8
    values = rng.normal(size=50)
    s = SDFT(n=N)
    for i, v in enumerate(values):
        _ = s.update(v)
        if i + 1 >= N:
            window = values[i + 1 - N : i + 1]
            expected = np.fft.fft(window)
            assert np.allclose(s.X, expected, atol=1e-9), f"step {i}"


def test_partial_window_is_zero_padded():
    s = SDFT(n=4)
    _ = s.update(1.0)
    expected = np.fft.fft([1.0], n=4)
    assert np.allclose(s.X, expected, atol=1e-12)
    _ = s.update(2.0)
    expected = np.fft.fft([1.0, 2.0], n=4)
    assert np.allclose(s.X, expected, atol=1e-12)


def test_is_ready_flips_at_n():
    s = SDFT(n=3)
    assert not s.is_ready
    _ = s.update(1.0)
    assert not s.is_ready
    _ = s.update(2.0)
    assert not s.is_ready
    _ = s.update(3.0)
    assert s.is_ready


def test_nan_skipped():
    s = SDFT(n=4)
    _ = s.update(1.0)
    _ = s.update(float("nan"))
    _ = s.update(2.0)
    assert s.n_seen_ == 2
    expected = np.fft.fft([1.0, 2.0], n=4)
    assert np.allclose(s.X, expected, atol=1e-12)


def test_update_many_matches_loop():
    rng = np.random.default_rng(1)
    values = rng.normal(size=20)
    a = SDFT(n=5)
    b = SDFT(n=5)
    a.update_many(values)
    for v in values:
        b.update(v)
    assert np.allclose(a.X, b.X)


def test_rejects_zero_n():
    with pytest.raises(ValueError):
        SDFT(n=0)


def test_doctest():
    res = doctest.testmod(sdft_module, verbose=False)
    assert res.failed == 0
