import struct
import sys
import types

import numpy as np
import pytest

from dense_armor.core.audio import (
    YIN,
    Audio,
    AudioInput,
    AudioLiveStream,
    Oscillator,
    Resampler,
    cents_to_freq,
    cola_ok,
    hard_clip,
    hard_clip_blamp,
    note_to_freq,
    read_wav,
    rectangular,
    resample,
    run_estimators,
    stft,
)
from dense_armor.core.audio import blamp as blamp_mod

SR = 8000


def _blocks(n=3):
    t = np.arange(64)
    return [Audio(np.sin(0.3 * (k + 1) * t), SR, k * 64 / SR) for k in range(n)]


class _ScalarScorer:
    """Takes one number, not the dict: the adapter must fall back to the scalar form."""

    def __init__(self):
        self.seen = []

    def score_one(self, x):
        v = float(x)
        return abs(v - float(np.mean(self.seen))) if self.seen else 0.0

    def learn_one(self, x):
        self.seen.append(float(x))


class _ScalarScaler:
    def __init__(self):
        self.n = 0

    def learn_one(self, x):
        float(x)
        self.n += 1

    def transform_one(self, x):
        return 2.0 * float(x)


def test_audio_input_falls_back_to_the_scalar_form_and_keeps_it():
    ai = AudioInput(_ScalarScorer())
    out = ai.run(_blocks(3))
    assert ai.mode == "score" and ai.scalar and len(out) == 3
    assert out[0] == 0.0 and out[1] > 0.0 and out[2] > 0.0
    sc = AudioInput(_ScalarScaler())
    first = sc.feed(_blocks(1)[0])
    second = sc.feed(_blocks(2)[1])
    assert sc.mode == "transform" and sc.scalar and sc.primitive.n == 2
    assert first == pytest.approx(
        2.0 * AudioInput(_ScalarScaler()).features.transform(_blocks(1)[0])["b0"]
    )
    assert second > 0.0


def test_audio_input_rejects_an_object_with_no_known_method():
    with pytest.raises(ValueError, match="no known method"):
        AudioInput(object()).feed(_blocks(1)[0])
    rep = run_estimators(_blocks(2), {"x": object()})
    assert rep["x"]["error"] == "ValueError: no known method" and rep["x"]["n"] == 0


def test_hard_clip_blamp_survives_a_flat_polynomial(monkeypatch):
    s = 2.0 * np.sin(2 * np.pi * 0.05 * np.arange(60))
    monkeypatch.setattr(blamp_mod, "_fit", lambda x, na: (0.0, 0.0, 0.0, 0.5))
    assert np.allclose(hard_clip_blamp(s, 1.0), hard_clip(s, 1.0))
    assert blamp_mod._r(2.5) == 0.0 and blamp_mod._r(-2.0) == 0.0


def test_stft_takes_a_window_array_and_checks_its_arguments():
    x = np.random.default_rng(0).standard_normal(64)
    assert np.allclose(stft(x, 8, 4, np.ones(8)), stft(x, 8, 4, "rectangular"))
    with pytest.raises(ValueError):
        stft(x, 8, 4, np.ones(7))
    with pytest.raises(ValueError):
        stft(np.ones((4, 4)), 8, 4)
    with pytest.raises(ValueError):
        stft(x, 0, 4)
    with pytest.raises(ValueError):
        stft(x, 8, 0)
    with pytest.raises(ValueError):
        cola_ok(np.ones(8), 0)
    with pytest.raises(ValueError):
        rectangular(0)


def test_constructors_and_helpers_check_their_arguments():
    with pytest.raises(ValueError):
        Oscillator("sine", 0)
    with pytest.raises(ValueError):
        YIN(0.0, 80.0, 0.1)
    with pytest.raises(ValueError):
        YIN(SR, 80.0, 0.1, f0_max=50.0)
    with pytest.raises(ValueError):
        cents_to_freq(100.0, 0.0)
    with pytest.raises(ValueError):
        note_to_freq(0, 0, 0.0, -1.0)
    with pytest.raises(ValueError):
        Resampler(1.0).set_gain(float("inf"))
    with pytest.raises(ValueError):
        Resampler(1.0).set_ratio(0.0)
    with pytest.raises(ValueError):
        resample(Audio(np.zeros((8, 2)), SR), 2.0)


def _riff(*chunks):
    body = b"WAVE" + b"".join(chunks)
    return b"RIFF" + struct.pack("<I", len(body)) + body


def _chunk(cid, payload):
    return (
        cid
        + struct.pack("<I", len(payload))
        + payload
        + (b"\x00" if len(payload) % 2 else b"")
    )


def test_read_wav_names_what_is_wrong_with_a_file(tmp_path):
    pcm = struct.pack("<HHIIHH", 1, 1, SR, 2 * SR, 2, 16)
    adpcm = struct.pack("<HHIIHH", 2, 1, SR, SR, 1, 4)
    p = tmp_path / "no_data.wav"
    p.write_bytes(_riff(_chunk(b"fmt ", pcm)))
    with pytest.raises(ValueError, match="missing fmt or data"):
        read_wav(p)
    q = tmp_path / "adpcm.wav"
    q.write_bytes(_riff(_chunk(b"fmt ", adpcm), _chunk(b"data", b"\x00\x00")))
    with pytest.raises(ValueError, match="unsupported format: tag 2, 4 bits"):
        read_wav(q)
    r = tmp_path / "odd.wav"
    r.write_bytes(
        _riff(
            _chunk(b"LIST", b"abc"),
            _chunk(b"fmt ", pcm),
            _chunk(b"data", struct.pack("<2h", 0, 16384)),
        )
    )
    assert read_wav(r).x.tolist() == [0.0, 0.5]


def test_live_stream_builds_the_real_input_stream_with_the_given_arguments(monkeypatch):
    made = {}

    class Dev:
        def __init__(self, **kw):
            made.update(kw)

        def start(self):
            made["started"] = True

        def stop(self):
            pass

        def close(self):
            made["closed"] = True

    monkeypatch.setitem(
        sys.modules, "sounddevice", types.SimpleNamespace(InputStream=Dev)
    )
    s = AudioLiveStream(SR, 100, queue_size=2, device=3).start()
    s.stop()
    assert made["samplerate"] == SR and made["blocksize"] == 100
    assert made["device"] == 3 and made["channels"] == 1 and callable(made["callback"])
    assert made["started"] and made["closed"]
