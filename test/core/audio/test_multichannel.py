"""Multichannel blocks: the live input, one channel of a block, and one set of primitives per channel."""

import os

import numpy as np
import pytest

from dense_armor.core.audio import Audio, AudioLiveStream, run_channels
from dense_armor.core.online.anomaly.filters import HampelScorer
from dense_armor.core.online.drift.adwin import ADWIN

SR = 8000


def _backend(chunks):
    class Dev:
        def __init__(self, cb):
            self.cb = cb

        def start(self):
            for c in chunks:
                self.cb(c, len(c), None, None)

        def stop(self):
            pass

        def close(self):
            pass

    return lambda cb, sr, block, device: Dev(cb)


def _eight(n=2000):
    t = np.arange(n) / SR
    return np.stack([np.sin(2 * np.pi * 200 * (k + 1) * t) for k in range(8)], axis=1)


def test_a_live_stream_of_eight_channels_gives_blocks_of_eight_channels():
    x = _eight()
    chunks = [x[i : i + 400] for i in range(0, 2000, 400)]
    s = AudioLiveStream(
        SR, 400, queue_size=8, backend=_backend(chunks), channels=8
    ).start()
    got = [s.read(0.1) for _ in range(5)]
    assert all(g is not None and g.channels == 8 and g.n == 400 for g in got)
    assert np.allclose(np.concatenate([g.x for g in got]), x)
    assert [g.t0 for g in got] == pytest.approx([i * 400 / SR for i in range(5)])
    for k in range(8):
        assert np.allclose(got[0].channel(k).x, x[:400, k])
    assert s.dropped == 0 and s.channels == 8
    s.stop()


def test_one_channel_is_still_the_default_and_takes_the_first_channel():
    x = _eight(800)
    s = AudioLiveStream(SR, 400, backend=_backend([x[:400], x[400:]])).start()
    a = s.read(0.1)
    assert s.channels == 1 and a.channels == 1 and a.x.ndim == 1
    assert np.allclose(a.x, x[:400, 0])


def test_a_stream_needs_at_least_one_channel():
    with pytest.raises(ValueError, match="channels must be >= 1"):
        AudioLiveStream(SR, 100, backend=_backend([]), channels=0)


def test_a_channel_of_a_block_is_that_column_and_the_others_are_refused():
    a = Audio(np.arange(12.0).reshape(4, 3), SR, 0.5)
    assert a.channel(2).x.tolist() == [2.0, 5.0, 8.0, 11.0]
    assert a.channel(0).t0 == 0.5 and a.channel(0).channels == 1
    mono = Audio(np.arange(4.0), SR)
    assert mono.channel(0) is mono
    for k in (-1, 3):
        with pytest.raises(IndexError, match="not in 0..2"):
            a.channel(k)
    with pytest.raises(IndexError, match="not in 0..0"):
        mono.channel(1)


def _blocks(x, block=400):
    return [
        Audio(x[i : i + block], SR, i / SR) for i in range(0, len(x) - block + 1, block)
    ]


def test_each_channel_is_read_by_its_own_primitives():
    n = 16000
    x = np.zeros((n, 3))
    x[:, 0] = 0.01 * np.random.default_rng(0).standard_normal(n)
    x[:, 1] = 0.01 * np.random.default_rng(1).standard_normal(n)
    x[n // 2 :, 1] *= 100.0
    x[:, 2] = x[:, 0]
    out = run_channels(
        _blocks(x), lambda k: {"adwin": ADWIN(), "hampel": HampelScorer()}
    )
    assert len(out) == 3
    assert [o["adwin"]["error"] for o in out] == [None] * 3
    assert out[0]["adwin"]["alarms"] == 0 and out[1]["adwin"]["alarms"] >= 1
    assert out[0]["hampel"]["last"] == out[2]["hampel"]["last"]
    assert out[1]["hampel"]["mode"] == "score" and out[1]["adwin"]["mode"] == "drift"
    assert all(o["adwin"]["n"] == 40 for o in out)


def test_the_primitives_can_differ_by_channel():
    x = _eight(1600)
    made = []

    def make(k):
        made.append(k)
        return {"p": HampelScorer() if k % 2 == 0 else ADWIN()}

    out = run_channels(_blocks(x, 400), make)
    assert made == list(range(8))
    assert [o["p"]["mode"] for o in out] == ["score", "drift"] * 4


def test_a_mono_block_is_channel_zero_and_a_label_is_passed_on():
    from dense_armor.core.online.learn.online_classifiers import OnlineGaussianNB

    x = _eight(1600)[:, 0]
    out = run_channels(
        _blocks(x, 400), lambda k: {"nb": OnlineGaussianNB()}, label=lambda b: 1
    )
    assert (
        len(out) == 1
        and out[0]["nb"]["mode"] == "supervised"
        and out[0]["nb"]["error"] is None
    )


def test_blocks_with_different_channels_or_no_blocks_are_refused():
    a = Audio(np.zeros((400, 2)), SR)
    b = Audio(np.zeros((400, 3)), SR)
    with pytest.raises(ValueError, match="same number of channels"):
        run_channels([a, b], lambda k: {"h": HampelScorer()})
    with pytest.raises(ValueError, match="no block"):
        run_channels([], lambda k: {"h": HampelScorer()})


@pytest.mark.skipif(
    "DENSE_ARMOR_AUDIO_DEVICE" not in os.environ,
    reason="set DENSE_ARMOR_AUDIO_DEVICE to the index of an input device with 2 channels or more",
)
def test_a_real_input_with_two_channels():
    s = AudioLiveStream(
        44100,
        1024,
        queue_size=64,
        device=int(os.environ["DENSE_ARMOR_AUDIO_DEVICE"]),
        channels=2,
    ).start()
    blocks = [s.read(1.0) for _ in range(20)]
    s.stop()
    assert all(b is not None and b.channels == 2 and b.n == 1024 for b in blocks)
    assert s.dropped == 0
