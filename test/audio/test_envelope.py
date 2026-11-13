"""The analytic signal and the envelope follower."""

import numpy as np
import pytest

from dense_armor.core.audio.envelope import envelope_follower, hilbert

SR = 8000.0


def test_the_hilbert_transform_of_a_cosine_is_a_sine():
    n = 256
    t = np.arange(n) / n
    z = hilbert(np.cos(2 * np.pi * 8 * t))
    assert np.allclose(z.real, np.cos(2 * np.pi * 8 * t), atol=1e-12)
    assert np.allclose(z.imag, np.sin(2 * np.pi * 8 * t), atol=1e-12)
    assert np.allclose(np.abs(z), 1.0, atol=1e-12)


@pytest.mark.parametrize("n", [1, 2, 7, 64, 257, 1000])
def test_the_real_part_is_the_signal_and_the_spectrum_has_no_negative_frequencies(n):
    x = np.random.default_rng(n).standard_normal(n)
    z = hilbert(x)
    assert np.allclose(z.real, x, atol=1e-12)
    spec = np.fft.fft(z)
    assert np.allclose(spec[(n + 1) // 2 + (0 if n % 2 else 1) :], 0.0, atol=1e-9)


@pytest.mark.parametrize("n", [8, 63, 64, 1000, 1001])
def test_hilbert_matches_scipy(n):
    signal = pytest.importorskip("scipy.signal")
    x = np.random.default_rng(n).standard_normal(n)
    assert np.max(np.abs(hilbert(x) - signal.hilbert(x))) < 1e-6


@pytest.mark.parametrize("bad", [np.zeros(0), np.zeros((2, 3)), np.ones(4) + 1j])
def test_hilbert_refuses_what_is_not_a_real_1d_signal(bad):
    with pytest.raises(ValueError, match="real 1D"):
        hilbert(bad)


def test_the_envelope_of_a_440_hz_tone_is_constant_within_five_percent():
    t = np.arange(8000) / SR
    y = envelope_follower(0.8 * np.sin(2 * np.pi * 440 * t), SR)
    assert np.abs(y[800:] / 0.8 - 1.0).max() < 0.05


def test_a_burst_rises_and_falls_with_the_time_constants():
    attack, release = 0.01, 0.1
    n = int(2.0 * SR)
    x = np.zeros(n)
    on, off = int(0.2 * SR), int(1.0 * SR)
    t = np.arange(on, off) / SR
    x[on:off] = np.sin(2 * np.pi * 440 * t)
    y = envelope_follower(x, SR, attack_s=attack, release_s=release)
    rise = y[on + int(attack * SR)]
    assert abs(rise - (1 - np.exp(-1))) < 0.10 * (1 - np.exp(-1))
    top = y[off - 1]
    fall = y[off - 1 + int(release * SR)]
    assert abs(fall / top - np.exp(-1)) < 0.10 * np.exp(-1)
    assert y[:on].max() < 0.05


def test_a_zero_time_constant_follows_the_envelope_at_once():
    x = np.random.default_rng(0).standard_normal(500)
    y = envelope_follower(x, SR, attack_s=0.0, release_s=0.0)
    assert np.allclose(y, np.abs(hilbert(x)), atol=1e-12)


def test_the_output_has_the_length_of_the_input_and_is_not_negative():
    x = np.random.default_rng(1).standard_normal(333)
    y = envelope_follower(x, SR)
    assert y.shape == x.shape and (y >= 0.0).all()


@pytest.mark.parametrize("kw", [{"attack_s": -1.0}, {"release_s": -0.1}])
def test_the_times_must_not_be_negative(kw):
    with pytest.raises(ValueError, match="must be positive"):
        envelope_follower(np.zeros(10), SR, **kw)
    with pytest.raises(ValueError, match="must be positive"):
        envelope_follower(np.zeros(10), 0.0)
