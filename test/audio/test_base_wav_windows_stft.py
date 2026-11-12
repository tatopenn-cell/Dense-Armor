import doctest
import wave

import numpy as np
import pytest

from dense_armor.utility.signals.audio import (
    Audio,
    StreamingSTFT,
    blackman,
    cola_ok,
    get_window,
    hamming,
    hann,
    istft,
    read_wav,
    rectangular,
    stft,
    write_wav,
)
from dense_armor.utility.signals.audio import fourier as stft_mod
from dense_armor.utility.signals.audio import signal as signal_mod
from dense_armor.utility.signals.audio import windows as windows_mod


def test_audio_properties_and_errors():
    a = Audio(np.zeros((8000, 2)), 8000, 1.5)
    assert (a.n, a.channels, a.duration, a.t0) == (8000, 2, 1.0, 1.5)
    assert a.to_mono().channels == 1
    with pytest.raises(ValueError):
        Audio(np.zeros(4), 0)
    with pytest.raises(ValueError):
        Audio(np.zeros((2, 2, 2)), 8000)


@pytest.mark.parametrize("sub,bits", [("pcm16", 16), ("pcm24", 24), ("pcm32", 32)])
def test_wav_roundtrip_within_one_step(tmp_path, sub, bits):
    t = np.arange(800) / 8000
    x = 0.9 * np.sin(2 * np.pi * 440 * t)
    p = tmp_path / "a.wav"
    write_wav(p, Audio(x, 8000), sub)
    y = read_wav(p)
    assert y.sr == 8000
    assert np.max(np.abs(y.x - x)) <= 1.0 / 2 ** (bits - 1)


def test_wav_float32_and_stereo(tmp_path):
    x = np.random.default_rng(0).uniform(-1, 1, (50, 2))
    p = tmp_path / "s.wav"
    write_wav(p, Audio(x, 4000), "float32")
    y = read_wav(p)
    assert y.x.shape == (50, 2)
    assert np.max(np.abs(y.x - x)) < 1e-6


def test_wav_agrees_with_stdlib_wave(tmp_path):
    x = np.array([0.0, 0.5, -0.5, 0.25, -1.0])
    p = tmp_path / "w.wav"
    write_wav(p, Audio(x, 8000), "pcm16")
    with wave.open(str(p)) as f:
        ints = np.frombuffer(f.readframes(5), dtype="<i2")
        assert (f.getframerate(), f.getnchannels(), f.getsampwidth()) == (8000, 1, 2)
    assert ints.tolist() == [0, 16384, -16384, 8192, -32768]
    q = tmp_path / "std.wav"
    with wave.open(str(q), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(8000)
        f.writeframes(np.array([0, 16384, -32768], dtype="<i2").tobytes())
    assert read_wav(q).x.tolist() == [0.0, 0.5, -1.0]


def test_wav_errors(tmp_path):
    p = tmp_path / "bad.wav"
    p.write_bytes(b"not a wav file at all")
    with pytest.raises(ValueError):
        read_wav(p)
    with pytest.raises(ValueError):
        write_wav(tmp_path / "x.wav", Audio(np.zeros(2), 8000), "mp3")


def test_window_hand_values():
    n = 8
    assert hann(n)[[0, 2, 4]].tolist() == pytest.approx([0.0, 0.5, 1.0], abs=1e-12)
    assert hamming(n)[0] == pytest.approx(0.54 - 0.46)
    assert hamming(n)[4] == pytest.approx(0.54 + 0.46)
    assert hamming(n, a=25 / 46)[0] == pytest.approx(25 / 46 - 21 / 46)
    b = blackman(n)
    assert b[0] == pytest.approx(0.42 - 0.50 + 0.08, abs=1e-12)
    assert b[4] == pytest.approx(0.42 + 0.50 + 0.08)
    assert b[2] == pytest.approx(0.42 - 0.08)
    assert rectangular(5).tolist() == [1.0] * 5


def test_window_symmetric_vs_periodic_and_errors():
    assert np.allclose(hann(9, periodic=False), hann(9, periodic=False)[::-1])
    assert hann(9, periodic=False)[0] == pytest.approx(0.0)
    assert hann(9, periodic=False)[4] == pytest.approx(1.0)
    assert hann(1).tolist() == [1.0]
    with pytest.raises(ValueError):
        hann(0)
    with pytest.raises(ValueError):
        get_window("kaiser", 8)


def test_cola():
    w = hann(16)
    assert cola_ok(w, 8)
    assert cola_ok(w, 4)
    assert not cola_ok(rectangular(8), 16)
    assert not cola_ok(hamming(16), 5)


def test_stft_shape_and_tone_peak():
    sr, f = 1024, 128.0
    x = np.sin(2 * np.pi * f * np.arange(4096) / sr)
    S = stft(x, 256, 128, "hann")
    assert S.shape == (1 + (4096 - 256) // 128, 129)
    assert np.all(np.abs(S).argmax(axis=1) == 32)
    assert stft(np.ones(10), 16, 4).shape == (0, 9)


@pytest.mark.parametrize(
    "name,hop", [("hann", 128), ("hann", 64), ("rectangular", 256)]
)
def test_istft_perfect_reconstruction_interior(name, hop):
    rng = np.random.default_rng(1)
    x = rng.standard_normal(4096)
    S = stft(x, 256, hop, name)
    y = istft(S, hop, name, 256)
    lo, hi = 256 - hop, len(y) - (256 - hop)
    assert np.max(np.abs(y[lo:hi] - x[lo:hi])) < 1e-9


def test_istft_rejects_non_cola():
    S = stft(np.ones(512), 64, 20, "hamming")
    with pytest.raises(ValueError):
        istft(S, 20, "hamming", 64)


def test_streaming_stft_equals_block():
    rng = np.random.default_rng(2)
    x = rng.standard_normal(1000)
    S = stft(x, 64, 16, "hann")
    got = StreamingSTFT(64, 16, "hann").update_many(x)
    assert len(got) == S.shape[0]
    assert np.max(np.abs(np.array(got) - S)) < 1e-9


def test_streaming_stft_state_and_params():
    s = StreamingSTFT(8, 4, "hann")
    assert s.spectrum is None
    s.update_many(np.arange(8.0))
    assert s.frames_ == 1 and s.spectrum is not None
    assert s.get_params()["n_fft"] == 8
    assert s.budget_s > 0 and s.memory_class
    with pytest.raises(ValueError):
        StreamingSTFT(0, 1)


def test_bounded_buffer_on_long_stream():
    rng = np.random.default_rng(3)
    x = rng.standard_normal(64 * 40)
    S = stft(x, 64, 32, "hann")
    got = StreamingSTFT(64, 32, "hann").update_many(x)
    assert np.max(np.abs(np.array(got) - S)) < 1e-9


@pytest.mark.parametrize("mod", [signal_mod, windows_mod, stft_mod])
def test_doctests(mod):
    assert doctest.testmod(mod, extraglobs={"np": np}).failed == 0
