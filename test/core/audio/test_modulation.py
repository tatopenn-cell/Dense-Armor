"""The chorus and the phaser."""

import math

import numpy as np
import pytest

from dense_armor.core.audio import hilbert
from dense_armor.core.audio.modulation import (
    PHASER_F_MAX,
    PHASER_F_MIN,
    Chorus,
    Phaser,
)


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


SR = 8000.0
N_FFT = 8192


def _impulse_response(p, n=N_FFT):
    return p.update_many(np.r_[1.0, np.zeros(n - 1)])


def _notches(ir, sr=SR):
    mag = np.abs(np.fft.rfft(ir))
    idx = [
        i
        for i in range(1, len(mag) - 1)
        if mag[i] < mag[i - 1] and mag[i] < mag[i + 1] and mag[i] < 0.02
    ]
    return np.array(idx) * sr / len(ir)


def _expected_notches(stages, fc, sr=SR):
    t = math.tan(math.pi * fc / sr)
    ks = range(stages // 2)
    return np.array(
        [
            sr / math.pi * math.atan(t * math.tan((2 * k + 1) * math.pi / (2 * stages)))
            for k in ks
        ]
    )


def test_the_corner_frequency_sweeps_the_range_around_its_geometric_centre():
    p = Phaser(0.5, 0.0, 4, SR)
    assert p.corner_frequency(0) == pytest.approx(
        math.sqrt(PHASER_F_MIN * PHASER_F_MAX)
    )
    q = Phaser(1.0, 1.0, 4, SR)
    fcs = [q.corner_frequency(n) for n in range(int(SR))]
    assert min(fcs) == pytest.approx(PHASER_F_MIN, rel=1e-3)
    assert max(fcs) == pytest.approx(PHASER_F_MAX, rel=1e-3)


def test_the_static_response_matches_scipy():
    signal = pytest.importorskip("scipy.signal")
    mix, stages = 0.5, 4
    p = Phaser(0.0, 0.0, stages, SR, mix)
    ir = _impulse_response(p)
    t = math.tan(math.pi * p.corner_frequency(0) / SR)
    a = (t - 1) / (t + 1)
    grid = np.linspace(0.0, np.pi, len(ir) // 2 + 1)
    _, stage = signal.freqz([a, 1.0], [1.0, a], worN=grid)
    ref = (1 - mix) + mix * stage**stages
    assert np.max(np.abs(np.fft.rfft(ir) - ref)) < 1e-6


@pytest.mark.parametrize("stages", [2, 4, 5, 6])
def test_the_notches_are_where_the_phase_says(stages):
    p = Phaser(0.0, 0.0, stages, SR, 0.5)
    got = _notches(_impulse_response(p))
    want = _expected_notches(stages, p.corner_frequency(0))
    assert len(got) == stages // 2
    assert np.allclose(got, want, atol=2.0)


def test_one_stage_has_no_notch_below_the_nyquist_frequency():
    assert len(_notches(_impulse_response(Phaser(0.0, 0.0, 1, SR, 0.5)))) == 0


def test_the_notches_follow_the_lfo():
    p = Phaser(0.05, 1.0, 4, SR, 0.5)
    found = []
    for n in (0, int(SR / (4 * 0.05))):
        p2 = p.clone()
        p2._n = n
        got = _notches(_impulse_response(p2, 4096), SR)
        want = _expected_notches(4, p2.corner_frequency(n))
        found.append(got[0])
        assert abs(got[0] - want[0]) < 0.05 * want[0]
    assert found[1] / found[0] > 2.0


def test_mix_zero_is_the_input_and_mix_one_is_an_all_pass():
    x = np.random.default_rng(4).standard_normal(300)
    assert np.allclose(Phaser(0.5, 0.7, 4, SR, 0.0).update_many(x), x)
    ir = _impulse_response(Phaser(0.0, 0.0, 4, SR, 1.0))
    assert (ir**2).sum() == pytest.approx(1.0, abs=1e-9)


def test_the_phaser_parameters_are_checked_and_clone_and_state_work():
    for kw, msg in (
        ({"rate_hz": -1.0}, "rate_hz >= 0"),
        ({"depth": 1.5}, "rate_hz >= 0"),
        ({"mix": -0.1}, "rate_hz >= 0"),
        ({"stages": 0}, "stages must be"),
        ({"stages": 2.5}, "stages must be"),
        ({"sr": 4000.0}, "sr must be above"),
    ):
        with pytest.raises(ValueError, match=msg):
            Phaser(**kw)
    p = Phaser(0.3, 0.5, 3, SR, 0.4)
    assert p.clone().get_params() == p.get_params()
    x = np.random.default_rng(5).standard_normal(600)
    p.update_many(x[:300])
    twin = p.clone()
    twin.load_state_dict(p.state_dict())
    assert np.allclose(twin.update_many(x[300:]), p.update_many(x[300:]))
