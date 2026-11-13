"""The phase vocoder, the pitch shift, PSOLA and the harmonizer."""

import numpy as np
import pytest

from dense_armor.core.audio import yin_f0
from dense_armor.core.audio.timescale import (
    pitch_shift,
    psola_time_stretch,
    time_stretch,
)

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


@pytest.mark.parametrize("semitones", [12.0, -12.0, 7.0, -5.0, 0.5])
@pytest.mark.parametrize("n", [8000, 5003])
def test_the_length_is_kept(semitones, n):
    assert len(pitch_shift(_sine(440.0, n), SR, semitones)) == n


@pytest.mark.parametrize("semitones", [12.0, -12.0, 7.0, -5.0])
def test_the_frequency_is_multiplied_by_the_ratio_of_the_shift(semitones):
    y = pitch_shift(_sine(440.0), SR, semitones)
    cut = 1000
    want = 440.0 * 2 ** (semitones / 12)
    assert abs(_cents(_peak(y[cut:-cut]), want)) < 20.0


def test_plus_12_semitones_doubles_the_frequency():
    y = pitch_shift(_sine(440.0), SR, 12.0)
    assert abs(_cents(_peak(y[1000:-1000]), 880.0)) < 20.0
    frames = [
        yin_f0(y[i : i + 1024], SR, 80.0, 0.1) for i in range(0, len(y) - 1024, 512)
    ]
    assert abs(_cents(np.nanmedian(frames[2:-2]), 880.0)) < 20.0


def test_no_shift_is_the_input_except_the_first_and_the_last_samples():
    x = np.random.default_rng(1).standard_normal(5000)
    y = pitch_shift(x, SR, 0.0)
    assert np.max(np.abs(y[64:-1] - x[64:-1])) < 1e-9
    assert y[-1] == 0.0


def test_the_level_is_kept():
    x = _sine(440.0)
    for semitones in (12.0, -12.0, 7.0):
        y = pitch_shift(x, SR, semitones)
        assert _rms(y[1000:-1000]) == pytest.approx(_rms(x), rel=0.05)


@pytest.mark.parametrize("semitones", [12.0, -12.0, 7.0])
def test_pitch_shift_agrees_with_librosa(semitones):
    librosa = pytest.importorskip("librosa")
    x = _sine(440.0)
    y = pitch_shift(x, SR, semitones)
    ref = librosa.effects.pitch_shift(
        x, sr=SR, n_steps=semitones, n_fft=2048, hop_length=512
    )
    assert len(y) == len(ref)
    assert abs(_peak(y[1000:-1000]) - _peak(ref[1000:-1000])) < 1.0


def _voice(n=4000, f0=200.0, harmonics=6):
    t = np.arange(n) / SR
    return sum(np.sin(2 * np.pi * f0 * k * t) / k for k in range(1, harmonics + 1))


@pytest.mark.parametrize("rate", [0.5, 1.0, 1.7, 2.0])
@pytest.mark.parametrize("n", [4000, 2503])
def test_psola_makes_the_length_the_input_length_times_the_rate(rate, n):
    assert len(psola_time_stretch(_voice(n), SR, 200.0, rate)) == round(n * rate)


@pytest.mark.parametrize("rate", [0.5, 2.0])
def test_psola_keeps_the_pitch(rate):
    y = psola_time_stretch(_sine(200.0, 4000), SR, 200.0, rate)
    assert abs(_cents(_peak(y), 200.0)) < 10.0
    frames = [
        yin_f0(y[i : i + 400], SR, 80.0, 0.1) for i in range(0, len(y) - 400, 100)
    ]
    assert np.nanmax(np.abs(_cents(np.array(frames[1:-1]), 200.0))) < 20.0


def test_the_output_is_periodic_with_the_period_of_the_input():
    y = psola_time_stretch(_voice(), SR, 200.0, 2.0)[1000:-1000]
    lag = 40
    corr = np.corrcoef(y[:-lag], y[lag:])[0, 1]
    assert corr > 0.99


def test_the_harmonics_keep_their_relative_level():
    x = _voice(4000)
    y = psola_time_stretch(x, SR, 200.0, 1.7)

    def level(s, f):
        spec = np.abs(np.fft.rfft(s * np.hanning(len(s)), 16 * len(s)))
        freq = np.fft.rfftfreq(16 * len(s), 1 / SR)
        return spec[np.argmin(np.abs(freq - f))]

    for k in (2, 3, 4):
        ratio_in = level(x, 200.0 * k) / level(x, 200.0)
        ratio_out = level(y, 200.0 * k) / level(y, 200.0)
        assert ratio_out == pytest.approx(ratio_in, rel=0.1)


def test_a_rate_of_one_gives_back_a_periodic_sound_and_keeps_its_level():
    x = _voice()
    y = psola_time_stretch(x, SR, 200.0, 1.0)
    assert np.corrcoef(x[500:-500], y[500:-500])[0, 1] > 0.99
    assert _rms(y[500:-500]) == pytest.approx(_rms(x[500:-500]), rel=0.05)


def test_a_track_of_f0_that_is_constant_gives_the_same_as_a_number():
    x = _voice(2000)
    a = psola_time_stretch(x, SR, 200.0, 1.5)
    b = psola_time_stretch(x, SR, np.full(len(x), 200.0), 1.5)
    assert np.allclose(a, b)


def test_unvoiced_samples_get_the_default_period_and_the_length_holds():
    x = _voice(3000)
    f0 = np.full(len(x), 200.0)
    f0[1000:1500] = np.nan
    f0[1500:1800] = 0.0
    y = psola_time_stretch(x, SR, f0, 1.3)
    assert len(y) == round(3000 * 1.3) and np.isfinite(y).all()


def test_a_track_that_follows_a_glide_keeps_the_glide():
    n = 6000
    t = np.arange(n) / SR
    f = 150.0 + 100.0 * t / t[-1]
    x = np.sin(2 * np.pi * np.cumsum(f) / SR)
    y = psola_time_stretch(x, SR, f, 2.0)
    first = yin_f0(y[1000:1400], SR, 80.0, 0.1)
    last = yin_f0(y[-1500:-1100], SR, 80.0, 0.1)
    assert 140.0 < first < 190.0 and 200.0 < last < 260.0 and last > first + 30.0


def test_psola_refuses_what_it_cannot_do():
    x = _voice(1000)
    with pytest.raises(ValueError, match="non-empty 1D"):
        psola_time_stretch(np.zeros(0), SR, 200.0, 2.0)
    for kw in ({"sr": 0.0}, {"rate": 0.0}, {"rate": -1.0}):
        args = {"sr": SR, "rate": 2.0, **kw}
        with pytest.raises(ValueError, match="positive"):
            psola_time_stretch(x, args["sr"], 200.0, args["rate"])
    with pytest.raises(ValueError, match="one value for each sample"):
        psola_time_stretch(x, SR, np.full(10, 200.0), 2.0)
