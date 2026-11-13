"""The phase vocoder, the pitch shift, PSOLA and the harmonizer."""

import numpy as np
import pytest

from dense_armor.core.audio import yin_f0
from dense_armor.core.audio.timescale import time_stretch

SR = 8000.0


def _sine(f, n=8000, phase=0.0):
    return np.sin(2 * np.pi * f * np.arange(n) / SR + phase)


def _peak(y, sr=SR, pad=16):
    spec = np.abs(np.fft.rfft(y * np.hanning(len(y)), pad * len(y)))
    k = int(np.argmax(spec))
    a, b, c = np.log(spec[k - 1 : k + 2] + 1e-300)
    return (k + 0.5 * (a - c) / (a - 2 * b + c)) * sr / (pad * len(y))


def _cents(a, b):
    return 1200 * np.log2(a / b)


def _rms(y):
    return float(np.sqrt((y**2).mean()))


@pytest.mark.parametrize("rate", [0.5, 0.75, 1.0, 1.37, 2.0, 3.0])
def test_the_length_is_the_input_length_times_the_rate(rate):
    for n in (8000, 5003, 777):
        assert len(time_stretch(_sine(440.0, n), SR, rate)) == round(n * rate)


def test_a_rate_of_one_gives_back_the_input_to_rounding_after_the_first_samples():
    x = np.random.default_rng(0).standard_normal(5000)
    y = time_stretch(x, SR, 1.0)
    assert np.max(np.abs(y[64:] - x[64:])) < 1e-9
    assert np.max(np.abs(y[:30] - x[:30])) > 1e-3


@pytest.mark.parametrize("rate", [0.5, 0.75, 1.37, 2.0])
def test_the_amplitude_is_kept_at_every_rate(rate):
    x = _sine(440.0) + 0.5 * _sine(880.0, phase=1.0)
    y = time_stretch(x, SR, rate)
    cut = min(2048, len(y) // 4)
    assert _rms(y[cut:-cut]) == pytest.approx(_rms(x), rel=0.02)


@pytest.mark.parametrize("f", [440.0, 437.3, 1000.0])
@pytest.mark.parametrize("rate", [0.5, 1.37, 2.0])
def test_a_sine_off_the_centre_of_a_bin_keeps_its_frequency(f, rate):
    x = _sine(f)
    y = time_stretch(x, SR, rate)
    assert abs(_cents(_peak(y), f)) < 2.0
    reference = yin_f0(x[:1024], SR, 80.0, 0.1)
    frames = [
        yin_f0(y[i : i + 1024], SR, 80.0, 0.1) for i in range(0, len(y) - 1024, 512)
    ]
    assert np.nanmax(np.abs(_cents(np.array(frames[2:-2]), reference))) < 5.0


def test_a_sine_at_the_centre_of_a_bin_is_the_ideal_stretched_sine():
    f = 128 * SR / 2048
    y = time_stretch(_sine(f), SR, 2.0)
    mid = y[3000:11000]
    ideal = np.sin(2 * np.pi * f * np.arange(len(mid)) / SR)
    quad = np.cos(2 * np.pi * f * np.arange(len(mid)) / SR)
    a, b = mid @ ideal, mid @ quad
    fit = (a * ideal + b * quad) / (ideal @ ideal)
    assert np.max(np.abs(mid - fit)) < 1e-9


def test_the_relative_level_of_two_partials_is_kept():
    x = _sine(440.0) + 0.5 * _sine(880.0, phase=1.0)
    for rate in (0.5, 2.0):
        y = time_stretch(x, SR, rate)
        cut = min(2048, len(y) // 4)
        spec = np.abs(
            np.fft.rfft(
                y[cut:-cut] * np.hanning(len(y) - 2 * cut), 16 * (len(y) - 2 * cut)
            )
        )
        freq = np.fft.rfftfreq(16 * (len(y) - 2 * cut), 1 / SR)
        a = spec[np.argmin(np.abs(freq - 440.0))]
        b = spec[np.argmin(np.abs(freq - 880.0))]
        assert b / a == pytest.approx(0.5, abs=0.02)


@pytest.mark.parametrize("rate", [0.5, 1.37, 2.0])
def test_time_stretch_agrees_with_librosa(rate):
    librosa = pytest.importorskip("librosa")
    x = _sine(440.0) + 0.5 * _sine(880.0, phase=1.0)
    y = time_stretch(x, SR, rate)
    ref = librosa.effects.time_stretch(x, rate=1.0 / rate, n_fft=2048, hop_length=512)
    assert len(y) == len(ref)
    cut = min(2048, len(y) // 4)
    a, b = y[cut:-cut], ref[cut:-cut]
    sa = np.abs(np.fft.rfft(a * np.hanning(len(a))))
    sb = np.abs(np.fft.rfft(b * np.hanning(len(b))))
    assert np.abs(sa / sa.max() - sb / sb.max()).max() < 0.03
    assert abs(_peak(y) - _peak(ref)) < 1.0


def test_a_short_signal_and_other_frames_work():
    x = _sine(440.0, 300)
    assert len(time_stretch(x, SR, 2.0)) == 600
    y = time_stretch(_sine(440.0), SR, 1.5, n_fft=1024, hop=256)
    assert len(y) == 12000 and abs(_cents(_peak(y), 440.0)) < 2.0


def test_time_stretch_refuses_what_it_cannot_do():
    for bad in (np.zeros(0), np.zeros((2, 3))):
        with pytest.raises(ValueError, match="non-empty 1D"):
            time_stretch(bad, SR, 2.0)
    for rate in (0.0, -1.0):
        with pytest.raises(ValueError, match="rate positive"):
            time_stretch(_sine(440.0), SR, rate)
