"""Compressor, limiter and soft clipper."""

import numpy as np
import pytest

from dense_armor.core.audio.dynamics import (
    Compressor,
    Limiter,
    SoftClip,
    gain_curve,
)

SR = 8000.0


def test_the_static_curve_at_ten_points():
    t, r, w = -20.0, 4.0, 6.0
    x = [-60.0, -40.0, -23.0, -21.5, -20.0, -18.5, -17.0, -10.0, 0.0, 10.0]
    hand = [
        -60.0,
        -40.0,
        -23.0,
        -21.640625,
        -20.5625,
        -19.765625,
        -19.25,
        -17.5,
        -15.0,
        -12.5,
    ]
    assert np.allclose(gain_curve(x, t, r, w), hand, atol=1e-12)


def test_the_static_curve_without_a_knee_and_with_an_infinite_ratio():
    assert gain_curve([-30.0, -20.0, -10.0, 0.0], -20.0, 4.0).tolist() == [
        -30.0,
        -20.0,
        -17.5,
        -15.0,
    ]
    assert gain_curve([-30.0, -20.0, 0.0, 10.0], -20.0, float("inf")).tolist() == [
        -30.0,
        -20.0,
        -20.0,
        -20.0,
    ]


def test_the_curve_and_its_slope_are_continuous_at_both_ends_of_the_knee():
    t, r, w, h = -20.0, 4.0, 6.0, 1e-6
    for edge in (t - w / 2, t + w / 2):
        left, right = gain_curve([edge - h, edge + h], t, r, w)
        assert abs(right - left) < 1e-5
    slopes = []
    for edge, side in ((t - w / 2, -1), (t + w / 2, 1)):
        a, b = gain_curve([edge + side * h, edge + side * 2 * h], t, r, w)
        slopes.append((b - a) / (side * h))
    assert slopes[0] == pytest.approx(1.0, abs=1e-4)
    assert slopes[1] == pytest.approx(1.0 / r, abs=1e-4)


@pytest.mark.parametrize("kw", [{"ratio": 0.5}, {"knee_db": -1.0}])
def test_the_curve_refuses_a_ratio_below_one_and_a_negative_knee(kw):
    with pytest.raises(ValueError, match="ratio must be >= 1"):
        gain_curve([0.0], -20.0, **{"ratio": 4.0, "knee_db": 0.0, **kw})


def test_a_level_12_db_above_the_threshold_gets_9_db_of_gain_reduction():
    c = Compressor(-20.0, 4.0, 0.0, 0.0, 0.0, 0.0, SR)
    level = 10 ** (-8.0 / 20.0)
    y = c.update_many(np.full(50, level))
    assert c.gain_reduction_db == pytest.approx(9.0, abs=1e-9)
    assert 20 * np.log10(y[-1]) == pytest.approx(-20.0 + 3.0, abs=0.5)


def test_below_the_threshold_nothing_changes_and_makeup_adds_a_gain():
    x = np.full(200, 10 ** (-40.0 / 20.0))
    assert np.allclose(Compressor(-20.0, 4.0, sr=SR).update_many(x), x)
    y = Compressor(-20.0, 4.0, makeup_db=6.0, sr=SR).update_many(x)
    assert np.allclose(y / x, 10 ** (6.0 / 20.0))


def test_the_smoothing_is_a_one_pole_filter():
    signal = pytest.importorskip("scipy.signal")
    tau = 0.01
    g = np.where(np.arange(400) >= 100, 9.0, 0.0)
    a = np.exp(-1.0 / (tau * SR))
    ref = signal.lfilter([1.0 - a], [1.0, -a], g)
    c = Compressor(-20.0, 4.0, tau, tau, 0.0, 0.0, SR)
    got = []
    for gi in g:
        level = -20.0 + 12.0 * (gi > 0) - 30.0 * (gi == 0)
        got.append(c.update(10 ** (level / 20.0)).gain_reduction_db)
    assert np.max(np.abs(np.array(got) - ref)) < 1e-9


def test_attack_and_release_follow_their_own_time_constants():
    attack, release, n = 0.05, 0.005, 10
    c = Compressor(-20.0, 4.0, attack, release, 0.0, 0.0, SR)
    c.update_many(np.full(n, 10 ** (-8.0 / 20.0)))
    rise = c.gain_reduction_db
    assert rise == pytest.approx(9.0 * (1 - np.exp(-n / (attack * SR))), rel=1e-9)
    c.update_many(np.full(n, 10 ** (-60.0 / 20.0)))
    drop = rise - c.gain_reduction_db
    assert drop / rise == pytest.approx(1 - np.exp(-n / (release * SR)), rel=1e-9)


def test_a_soft_knee_starts_to_compress_below_the_threshold_and_joins_the_hard_one():
    hard = Compressor(-20.0, 4.0, 0.0, 0.0, 0.0, 0.0, SR)
    soft = Compressor(-20.0, 4.0, 0.0, 0.0, 12.0, 0.0, SR)
    for level, below in ((-23.0, True), (-10.0, False)):
        x = np.full(5, 10 ** (level / 20.0))
        hard.update_many(x)
        soft.update_many(x)
        if below:
            assert hard.gain_reduction_db == 0.0 and soft.gain_reduction_db > 0.0
        else:
            assert soft.gain_reduction_db == pytest.approx(
                hard.gain_reduction_db, abs=1e-9
            )


def test_silence_gives_zero_and_a_negative_input_keeps_its_sign():
    c = Compressor(sr=SR)
    assert c.update_many(np.zeros(5)).tolist() == [0.0] * 5
    c2 = Compressor(-20.0, 4.0, 0.0, 0.0, 0.0, 0.0, SR)
    assert c2.update(-0.5).out < 0.0


def test_the_parameters_are_checked():
    for kw in (
        {"ratio": 0.9},
        {"attack_s": -0.1},
        {"release_s": -0.1},
        {"knee_db": -1.0},
        {"sr": 0.0},
    ):
        with pytest.raises(ValueError, match="ratio must be >= 1"):
            Compressor(**kw)


def test_clone_get_params_and_the_state_survive():
    c = Compressor(-18.0, 3.0, 0.02, 0.2, 4.0, 1.0, SR)
    assert c.clone().get_params() == c.get_params()
    x = np.random.default_rng(0).uniform(-1, 1, 300)
    c.update_many(x[:150])
    d = c.clone()
    d.load_state_dict(c.state_dict())
    assert np.allclose(d.update_many(x[150:]), c.update_many(x[150:]))


def _bursts(n, seed=0):
    r = np.random.default_rng(seed)
    x = r.uniform(-0.3, 0.3, n)
    for i in r.integers(0, n - 50, 12):
        x[i : i + int(r.integers(1, 40))] *= r.uniform(5, 12)
    return x


def test_no_sample_is_above_the_ceiling():
    for ceiling_db, look in ((-6.0, 0.005), (-0.1, 0.002), (-12.0, 0.0001)):
        lim = Limiter(ceiling_db, look, 0.02, SR)
        y = lim.update_many(_bursts(6000, seed=int(-ceiling_db)))
        assert np.abs(y).max() <= 10 ** (ceiling_db / 20.0) * (1 + 1e-12)


def test_a_full_scale_signal_comes_out_at_the_ceiling_within_a_tenth_of_a_db():
    sr = 8000.0
    x = np.sin(2 * np.pi * 440.0 * np.arange(1000) / sr)
    y = Limiter(-6.0, 0.005, 0.05, sr).update_many(x)
    peak_db = 20 * np.log10(np.abs(y).max())
    assert peak_db <= -6.0 + 0.1


def test_the_gain_falls_over_the_lookahead_before_the_loud_sample_comes_out():
    lim = Limiter(-6.0, 0.005, 0.05, SR)
    n_look = lim.latency
    x = np.full(400, 0.1)
    n0 = 150
    x[n0:] = 1.0
    gr = np.array([lim.update(v).gain_reduction_db for v in x])
    start = int(np.argmax(gr > 1e-9))
    full = int(np.argmax(gr >= gr.max() - 1e-9))
    assert start == n0
    assert abs((full - start) - n_look) <= round(0.001 * SR)
    assert gr.max() == pytest.approx(6.0, abs=1e-6)


def test_a_signal_under_the_ceiling_passes_unchanged_and_delayed():
    x = np.random.default_rng(1).uniform(-0.4, 0.4, 500)
    lim = Limiter(-6.0, 0.005, 0.05, SR)
    y = lim.update_many(x)
    n = lim.latency
    assert np.allclose(y[n:], x[:-n]) and (y[:n] == 0.0).all()
    assert lim.gain_reduction_db == pytest.approx(0.0, abs=1e-12)


def test_the_gain_comes_back_after_a_burst_and_not_at_once():
    release = 0.02
    lim = Limiter(-6.0, 0.002, release, SR)
    x = np.concatenate([np.full(100, 0.1), np.full(40, 1.0), np.full(2000, 0.1)])
    gr = np.array([lim.update(v).gain_reduction_db for v in x])
    end = 100 + 40
    peak = gr[end - 1]
    assert gr[end + int(release * SR)] > 0.1 * peak
    assert gr[end + int(10 * release * SR)] < 0.01 * peak
    tail = gr[end + 20 :]
    assert (np.diff(tail) <= 1e-9).all()


def test_the_requested_gain_is_the_hard_limit_of_the_static_curve():
    levels = np.linspace(-30.0, 10.0, 41)
    amp = 10 ** (levels / 20.0)
    ceiling = 10 ** (-6.0 / 20.0)
    asked = np.where(amp > ceiling, ceiling / amp, 1.0)
    curve = gain_curve(levels, -6.0, float("inf"))
    assert np.allclose(asked, 10 ** (-(levels - curve) / 20.0), atol=1e-12)


def test_the_limiter_parameters_are_checked_and_clone_and_state_work():
    for kw in ({"lookahead_s": 0.0}, {"sr": 0.0}, {"release_s": -1.0}):
        with pytest.raises(ValueError, match="lookahead_s and sr"):
            Limiter(**kw)
    lim = Limiter(-6.0, 0.003, 0.03, SR)
    assert lim.clone().get_params() == lim.get_params()
    x = _bursts(400, seed=3)
    lim.update_many(x[:200])
    twin = lim.clone()
    twin.load_state_dict(lim.state_dict())
    assert np.allclose(twin.update_many(x[200:]), lim.update_many(x[200:]))


def _tone_report(y, f0, sr, n_harm):
    spec = np.abs(np.fft.rfft(y)) ** 2
    step = int(round(f0 * len(y) / sr))
    harm = [spec[step * m] for m in range(1, n_harm + 1)]
    thd = np.sqrt(sum(harm[1:])) / np.sqrt(harm[0])
    mask = np.ones(len(spec), bool)
    mask[::step] = False
    mask[0] = False
    return thd, 10 * np.log10(spec[mask].sum() / harm[0])


def _reference_thd(f0, sr, drive_db, n_harm, factor=16):
    n = int(sr)
    t = np.arange(n * factor) / (sr * factor)
    y = np.tanh(10 ** (drive_db / 20.0) * np.sin(2 * np.pi * f0 * t))
    spec = np.abs(np.fft.rfft(y))
    harm = [spec[int(f0 * m)] for m in range(1, n_harm + 1)]
    return np.sqrt(sum(h * h for h in harm[1:])) / harm[0]


CSR = 44100.0
SEG = slice(4410, 4410 + 8820)


@pytest.mark.parametrize(("f0", "n_harm"), [(1000.0, 22), (5000.0, 4)])
def test_the_harmonics_below_nyquist_match_the_reference(f0, n_harm):
    x = np.sin(2 * np.pi * f0 * np.arange(int(CSR)) / CSR)
    y = SoftClip(12.0, 4, CSR).update_many(x)[SEG]
    thd, _ = _tone_report(y, f0, CSR, n_harm)
    assert abs(thd - _reference_thd(f0, CSR, 12.0, n_harm)) < 0.005


def test_oversampling_removes_the_folded_harmonics_of_a_high_tone():
    f0 = 5000.0
    x = np.sin(2 * np.pi * f0 * np.arange(int(CSR)) / CSR)
    _, alias_naive = _tone_report(
        SoftClip(12.0, 1, CSR).update_many(x)[SEG], f0, CSR, 4
    )
    _, alias_4 = _tone_report(SoftClip(12.0, 4, CSR).update_many(x)[SEG], f0, CSR, 4)
    assert alias_naive > -30.0
    assert alias_4 < -40.0 and alias_4 < alias_naive - 40.0


def test_a_1_khz_tone_has_its_aliases_more_than_40_db_below_the_fundamental():
    x = np.sin(2 * np.pi * 1000.0 * np.arange(int(CSR)) / CSR)
    _, alias = _tone_report(SoftClip(12.0, 4, CSR).update_many(x)[SEG], 1000.0, CSR, 22)
    assert alias < -40.0


def test_the_alias_level_agrees_with_scipy_resampling():
    signal = pytest.importorskip("scipy.signal")
    f0 = 5000.0
    x = np.sin(2 * np.pi * f0 * np.arange(int(CSR)) / CSR)
    up = signal.resample_poly(x, 4, 1)
    y = signal.resample_poly(np.tanh(10 ** (12.0 / 20.0) * up), 1, 4)[SEG]
    thd_s, alias_s = _tone_report(y, f0, CSR, 4)
    thd, alias = _tone_report(SoftClip(12.0, 4, CSR).update_many(x)[SEG], f0, CSR, 4)
    assert abs(thd - thd_s) < 0.005
    assert alias < -40.0 and alias_s < -40.0


def test_without_oversampling_it_is_tanh_of_the_driven_signal_and_has_no_latency():
    x = np.random.default_rng(0).uniform(-1, 1, 300)
    sc = SoftClip(6.0, 1, CSR)
    assert sc.latency == 0
    assert np.allclose(sc.update_many(x), np.tanh(10 ** (6.0 / 20.0) * x), atol=1e-15)


def test_a_small_signal_comes_out_amplified_by_the_drive_and_delayed_by_the_latency():
    f0 = 200.0
    x = 0.001 * np.sin(2 * np.pi * f0 * np.arange(4000) / CSR)
    sc = SoftClip(20.0, 4, CSR)
    y = sc.update_many(x)
    n = sc.latency
    assert n == 64
    assert np.allclose(y[n + 300 :], 10.0 * x[300 : len(x) - n], atol=2e-5)


def test_a_signal_in_pieces_gives_the_same_output_as_in_one_block():
    x = np.random.default_rng(1).uniform(-1, 1, 1000)
    whole = SoftClip(12.0, 4, CSR).update_many(x)
    sc = SoftClip(12.0, 4, CSR)
    pieces = np.concatenate([sc.update_many(x[:300]), sc.update_many(x[300:])])
    assert np.allclose(pieces, whole, atol=1e-12)
    one = SoftClip(12.0, 4, CSR)
    for v in x[:10]:
        one.update(v)
    assert one.out == pytest.approx(whole[9], abs=1e-12)


def test_the_output_can_overshoot_one_but_not_the_sum_of_the_taps_and_empty_is_a_noop():
    from dense_armor.core.audio.dynamics import _lowpass_taps

    sc = SoftClip(30.0, 4, CSR)
    y = sc.update_many(np.random.default_rng(2).uniform(-1, 1, 2000))
    peak = np.abs(y).max()
    assert 1.0 < peak <= np.abs(_lowpass_taps(4)).sum()
    last = sc.out
    assert sc.update_many([]).shape == (0,) and sc.out == last


def test_the_soft_clip_parameters_are_checked_and_clone_and_state_work():
    for kw in ({"oversample": 0}, {"oversample": 2.5}, {"sr": 0.0}):
        with pytest.raises(ValueError, match="oversample must be an integer"):
            SoftClip(**kw)
    sc = SoftClip(9.0, 2, CSR)
    assert sc.clone().get_params() == sc.get_params()
    x = np.random.default_rng(3).uniform(-1, 1, 400)
    sc.update_many(x[:200])
    twin = sc.clone()
    twin.load_state_dict(sc.state_dict())
    assert np.allclose(twin.update_many(x[200:]), sc.update_many(x[200:]))
