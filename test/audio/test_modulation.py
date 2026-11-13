"""The chorus."""

import math

import numpy as np
import pytest

from dense_armor.core.audio import hilbert
from dense_armor.core.audio.modulation import Chorus


def test_the_output_equals_a_hand_written_loop():
    sr, rate, depth, delay, mix = 1000.0, 3.0, 2.0, 4.0, 0.4
    x = np.random.default_rng(0).standard_normal(400)
    got = Chorus(rate, depth, mix, sr, delay).update_many(x)
    ref = np.empty(len(x))
    for n in range(len(x)):
        d = (delay + depth * math.sin(2 * math.pi * rate * n / sr)) * 1e-3 * sr
        pos = n - d
        i0 = math.floor(pos)
        frac = pos - i0

        def at(i):
            return x[i] if i >= 0 else 0.0

        ref[n] = (1 - mix) * x[n] + mix * ((1 - frac) * at(i0) + frac * at(i0 + 1))
    assert np.max(np.abs(got - ref)) < 1e-12


def test_without_modulation_it_is_a_delay_and_mix_zero_is_the_input():
    x = np.random.default_rng(1).standard_normal(50)
    y = Chorus(0.0, 0.0, 1.0, 1000.0, 2.0).update_many(x)
    assert np.allclose(y[2:], x[:-2], atol=1e-15) and (y[:2] == 0.0).all()
    assert np.allclose(Chorus(0.5, 5.0, 0.0, 1000.0, 8.0).update_many(x), x)


def test_a_signal_in_pieces_gives_the_same_output():
    x = np.random.default_rng(2).standard_normal(900)
    whole = Chorus(1.5, 3.0, 0.5, 1000.0, 5.0).update_many(x)
    c = Chorus(1.5, 3.0, 0.5, 1000.0, 5.0)
    pieces = np.concatenate([c.update_many(x[:200]), c.update_many(x[200:])])
    assert np.allclose(pieces, whole, atol=1e-12)
    one = Chorus(1.5, 3.0, 0.5, 1000.0, 5.0)
    for v in x[:30]:
        one.update(v)
    assert one.out == pytest.approx(whole[29], abs=1e-12)
    assert one.update_many([]).shape == (0,)


def test_the_output_is_not_the_input_over_a_window_of_100_ms():
    sr = 8000.0
    x = np.sin(2 * np.pi * 440.0 * np.arange(int(3 * sr)) / sr)
    y = Chorus(0.5, 5.0, 0.5, sr).update_many(x)
    w = int(0.1 * sr)
    corr = [
        np.corrcoef(x[i : i + w], y[i : i + w])[0, 1] for i in range(w, len(x) - w, w)
    ]
    assert min(corr) < 0.9


@pytest.mark.parametrize("rate", [2.0, 1.3])
def test_the_rate_of_the_lfo_is_measured_on_the_output(rate):
    sr, f0 = 2000.0, 20.0
    x = np.sin(2 * np.pi * f0 * np.arange(int(10 * sr)) / sr)
    y = Chorus(rate, 5.0, 0.5, sr).update_many(x)
    env = np.abs(hilbert(y))[2000:-2000]
    env = env - env.mean()
    n = len(env) * 16
    spec = np.abs(np.fft.rfft(env * np.hanning(len(env)), n))
    k = int(np.argmax(spec[: n // 8]))
    a, b, c = np.log(spec[k - 1 : k + 2] + 1e-300)
    k = k + 0.5 * (a - c) / (a - 2 * b + c)
    assert abs(k * sr / n - rate) < 0.05


def test_the_chorus_parameters_are_checked_and_clone_and_state_work():
    for kw in (
        {"rate_hz": -1.0},
        {"depth_ms": -1.0},
        {"mix": 1.5},
        {"sr": 0.0},
    ):
        with pytest.raises(ValueError, match="must be >= 0"):
            Chorus(**kw)
    with pytest.raises(ValueError, match="delay_ms must be >= depth_ms"):
        Chorus(depth_ms=10.0, delay_ms=5.0)
    c = Chorus(1.0, 2.0, 0.3, 1000.0, 6.0)
    assert c.clone().get_params() == c.get_params()
    x = np.random.default_rng(3).standard_normal(400)
    c.update_many(x[:200])
    twin = c.clone()
    twin.load_state_dict(c.state_dict())
    assert np.allclose(twin.update_many(x[200:]), c.update_many(x[200:]))
