import doctest
import importlib
import math

import numpy as np
import pytest
from scipy.signal import lfilter

from dense_armor.core.audio import (
    YIN,
    AllPole,
    Audio,
    AudioFileStream,
    AudioInput,
    Oscillator,
    Resampler,
    cmndf,
    pitch_ratio_transition,
    rational_ratio,
    reflection_to_lpc,
    write_wav,
    yin_f0,
)

SR = 8000


def _d_reference(x, tau_max):
    n = len(x)
    xp = np.concatenate([x, np.zeros(tau_max + 1)])
    d = np.array(
        [sum((xp[i] - xp[i + t]) ** 2 for i in range(n)) for t in range(tau_max + 1)]
    )
    out = np.ones(tau_max + 1)
    for t in range(1, tau_max + 1):
        out[t] = d[t] / (d[1 : t + 1].sum() / t)
    return out


def test_cmndf_equals_the_definition_of_eq_1_and_2():
    x = np.random.default_rng(0).standard_normal(40)
    got = cmndf(x, 55)
    assert got[0] == 1.0 and np.allclose(got, _d_reference(x, 55))
    assert np.all(cmndf(np.zeros(32), 10) == 1.0)
    for bad in ((np.zeros((2, 2)), 4), (np.ones(1), 4), (np.ones(8), 0)):
        with pytest.raises(ValueError):
            cmndf(*bad)


def _yin_reference(x, sr, f0_min, threshold):
    d = _d_reference(x, math.ceil(sr / f0_min))
    for t in range(1, len(d) - 1):
        if d[t] < threshold and d[t] < d[t - 1] and d[t] <= d[t + 1]:
            den = d[t - 1] - 2 * d[t] + d[t + 1]
            delta = 0.0 if den == 0 else 0.5 * (d[t - 1] - d[t + 1]) / den
            return sr / (t + min(1.0, max(-1.0, delta)))
    return float("nan")


def test_yin_f0_equals_the_steps_of_the_paper_and_is_within_15_cents():
    n = np.arange(1024)
    for f in (110.0, 440.0, 441.3, 1000.0):
        x = np.sin(2 * np.pi * f * n / SR)
        got = yin_f0(x, SR, 80.0, 0.1)
        assert got == pytest.approx(_yin_reference(x, SR, 80.0, 0.1), rel=1e-9)
        assert abs(1200 * math.log2(got / f)) < 15.0
    saw = Oscillator("sawtooth_additive", SR).update_many(np.full(1024, 200.0))
    assert abs(1200 * math.log2(yin_f0(saw, SR, 80.0, 0.1) / 200.0)) < 15.0


def test_yin_f0_unvoiced_and_errors():
    n = np.arange(1024)
    noise = np.random.default_rng(1).standard_normal(1024)
    assert np.isnan(yin_f0(noise, SR, 80.0, 0.1))
    assert np.isnan(yin_f0(np.zeros(1024), SR, 80.0, 0.1))
    for kw in ({"sr": 0.0}, {"f0_min": 0.0}, {"threshold": 0.0}, {"f0_max": 50.0}):
        args = {"sr": SR, "f0_min": 80.0, "threshold": 0.1, **kw}
        with pytest.raises(ValueError):
            yin_f0(np.sin(n), **args)


def test_yin_through_the_audio_input_on_a_file_stream(tmp_path):
    t = np.arange(SR) / SR
    x = np.where(t < 0.5, np.sin(2 * np.pi * 440 * t), np.sin(2 * np.pi * 220 * t))
    p = tmp_path / "pitch.wav"
    write_wav(p, Audio(x, SR), "float32")
    ai = AudioInput(YIN(SR, 80.0, 0.1))
    f0 = np.array(ai.run(AudioFileStream(p, 1024)))
    assert ai.mode == "samples" and len(f0) == 7
    assert f0[:3] == pytest.approx(440.0, abs=1.0)
    assert f0[-2:] == pytest.approx(220.0, abs=1.0)
    assert ai.primitive.f0_ == f0[-1]


def test_reflection_to_lpc_hand_case_and_stability_with_the_sign_of_yu():
    assert reflection_to_lpc([0.5, 0.25]).tolist() == [0.625, 0.25]
    assert reflection_to_lpc([0.3]).tolist() == [0.3]
    rng = np.random.default_rng(2)
    unstable_with_hayes_eq4_sign = 0
    for _ in range(300):
        k = rng.uniform(-0.99, 0.99, int(rng.integers(2, 9)))
        assert AllPole.from_reflection(k).stable
        a = reflection_to_lpc(k)
        unstable_with_hayes_eq4_sign += np.max(np.abs(np.roots(np.r_[1.0, -a]))) >= 1.0
    assert unstable_with_hayes_eq4_sign > 100
    assert not AllPole.from_reflection([1.2, 0.5]).stable
    for bad in ([], np.zeros((2, 2))):
        with pytest.raises(ValueError):
            reflection_to_lpc(bad)


def test_allpole_equals_scipy_lfilter_whole_in_blocks_and_time_varying():
    rng = np.random.default_rng(3)
    a = reflection_to_lpc(rng.uniform(-0.9, 0.9, 4))
    x = rng.standard_normal(300)
    ref = lfilter([1.0], np.r_[1.0, a], x)
    assert np.allclose(AllPole(a).update_many(x), ref)
    f = AllPole(a)
    parts = np.concatenate([f.update_many(x[:120]), f.update_many(x[120:])])
    assert np.allclose(parts, ref)
    const = np.tile(a, (len(x), 1))
    assert np.allclose(AllPole(a).update_many(x, const), ref)
    slow = np.linspace(-0.3, 0.6, len(x))[:, None]
    got = AllPole([0.0]).update_many(x, slow)
    want = np.empty(len(x))
    prev = 0.0
    for i, v in enumerate(x):
        prev = want[i] = v - slow[i, 0] * prev
    assert np.allclose(got, want)
    with pytest.raises(ValueError):
        AllPole([0.1, 0.2]).update(1.0, [0.1])
    with pytest.raises(ValueError):
        AllPole([])


def test_allpole_through_the_audio_input_keeps_state_between_blocks():
    a = reflection_to_lpc([0.5, -0.3])
    x = np.random.default_rng(4).standard_normal(600)
    blocks = [Audio(x[i : i + 100], SR, i / SR) for i in range(0, 600, 100)]
    ai = AudioInput(AllPole(a))
    y = np.concatenate(ai.run(blocks))
    assert ai.mode == "samples"
    assert np.allclose(y, lfilter([1.0], np.r_[1.0, a], x))


def test_state_dict_restores_allpole_and_yin_contract():
    x = np.random.default_rng(5).standard_normal(80)
    f = AllPole(reflection_to_lpc([0.4, 0.2]))
    f.update_many(x[:40])
    g = f.clone()
    g.load_state_dict(f.state_dict())
    assert np.allclose(g.update_many(x[40:]), f.update_many(x[40:]))
    y = YIN(SR, 80.0, 0.1)
    assert y.clone().get_params() == y.get_params()
    for o in (f, y):
        assert o.budget_s > 0 and o.memory_class in ("O(1)", "O(window)")


def test_resampler_gain_follows_algorithm_3():
    x = [0.0, 1.0, 2.0, 3.0, 4.0]
    assert Resampler(2.0, gain=0.5).update_many(x).tolist() == [0.0, 1.0]
    r = Resampler(2.0).set_gain(-1.0)
    assert r.update_many(x).tolist() == [-0.0, -2.0]
    with pytest.raises(ValueError):
        Resampler(1.0, gain=float("inf"))


def test_pitch_ratio_transition_modes_and_errors():
    lin = pitch_ratio_transition(1.0, 2.0, 1.0, 1.0, 2)
    assert lin.tolist() == [1.0, 1.5, 2.0]
    cents = pitch_ratio_transition(1.0, 2.0, 1.0, 1.0, 2, "cents-linear")
    assert cents == pytest.approx([1.0, 2**0.5, 2.0])
    assert pitch_ratio_transition(1.5, 1.5, 48000, 44100, 3) == pytest.approx(
        [1.5 * 48000 / 44100] * 4
    )
    for bad in (
        (0.0, 2.0, 1.0, 1.0, 2),
        (1.0, 2.0, 1.0, 0.0, 2),
        (1.0, 2.0, 1.0, 1.0, 0),
    ):
        with pytest.raises(ValueError):
            pitch_ratio_transition(*bad)
    with pytest.raises(ValueError):
        pitch_ratio_transition(1.0, 2.0, 1.0, 1.0, 2, "cubic")


def test_resampler_glide_through_the_audio_input_is_continuous():
    s = np.sin(2 * np.pi * 50 * np.arange(4000) / SR)
    blocks = [Audio(s[i : i + 100], SR, i / SR) for i in range(0, 4000, 100)]
    q = pitch_ratio_transition(1.0, 1.5, SR, SR, len(blocks) - 1, "cents-linear")
    ai = AudioInput(Resampler(q[0]))
    out = []
    for qj, blk in zip(q, blocks):
        ai.primitive.set_ratio(qj)
        out.append(ai.feed(blk))
    y = np.concatenate(out)
    assert len(y) > 2000
    assert np.max(np.abs(np.diff(y))) <= 2 * np.pi * 50 / SR * 1.5 * 1.05


def test_rational_ratio_page_3():
    assert rational_ratio(1.5, 4) == (3, 2)
    d, u = rational_ratio(3333 / 3332, 3332)
    assert (d, u) == (3333, 3332)
    d, u = rational_ratio(0.7071, 1000)
    assert d / u == pytest.approx(0.7071, abs=5e-4)
    for bad in ((0.0, 4), (float("nan"), 4), (1.0, 0), (0.1, 4)):
        with pytest.raises(ValueError):
            rational_ratio(*bad)


@pytest.mark.parametrize("name", ["pitch", "allpole", "resample"])
def test_doctests(name):
    mod = importlib.import_module(f"dense_armor.core.audio.{name}")
    assert doctest.testmod(mod, extraglobs={"np": np}).failed == 0
