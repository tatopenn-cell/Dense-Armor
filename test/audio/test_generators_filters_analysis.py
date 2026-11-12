import doctest
import importlib

import numpy as np
import pytest

from dense_armor.utility.audio import (
    ADSR,
    FIR,
    Audio,
    AudioFeatures,
    Biquad,
    DelayLine,
    KarplusStrong,
    Oscillator,
    Resampler,
    StreamingSTFT,
    bandpass,
    cents_to_freq,
    freq_to_cents,
    freq_to_note,
    hard_clip,
    highpass,
    highshelf,
    lowpass,
    lowshelf,
    nmf,
    notch,
    note_to_freq,
    peaking,
    plucked_string,
    resample,
    separate,
    triangle_blamp,
    white_noise,
)

SR = 44100
F0 = 3141.5926


def _fundamental(y, sr):
    n = 1 << 20
    X = np.abs(np.fft.rfft(y * np.hanning(len(y)), n=n))[: n // 2]
    ok = (
        np.where((X[1:-1] > X[:-2]) & (X[1:-1] >= X[2:]) & (X[1:-1] > 0.3 * X.max()))[0]
        + 1
    )
    return ok[0] * sr / n


def _off_harmonic_db(x, f0, sr):
    X = np.abs(np.fft.rfft(x * np.hanning(len(x)))) ** 2
    mask = np.zeros(len(X), bool)
    for h in range(1, int(sr / 2 / f0) + 1):
        c = int(round(h * f0))
        mask[max(0, c - 3) : c + 4] = True
    return 10 * np.log10(X[~mask].sum() / X.sum())


@pytest.mark.parametrize("p", [64, 128, 256])
def test_karplus_strong_period_is_p_plus_half(p):
    y = KarplusStrong(p, 3).update_many(np.zeros(SR // 2))
    f = _fundamental(y, SR)
    assert abs(1200 * np.log2(f / (SR / (p + 0.5)))) < 1.0


def test_karplus_strong_decays_and_block_equals_sample():
    ks = KarplusStrong(100, 1)
    y = ks.update_many(np.zeros(20000))
    n = len(y) // 10
    assert np.sqrt(np.mean(y[-n:] ** 2)) < np.sqrt(np.mean(y[:n] ** 2))
    k2 = KarplusStrong(100, 1)
    z = np.array([k2.update().out for _ in range(300)])
    assert np.array_equal(z, y[:300])
    assert set(np.abs(y[:100])) == {1.0}


def test_plucked_string_pitch_and_errors():
    a = plucked_string(440.0, SR, 0.5, seed=2)
    assert a.sr == SR and a.n == SR // 2
    assert abs(_fundamental(a.x, SR) - 440.0) < 2.0
    with pytest.raises(ValueError):
        KarplusStrong(1)


def test_sine_peak_and_phase_continuity():
    y = Oscillator("sine", 8000).update_many(np.full(8000, 1000.0))
    assert np.abs(np.fft.rfft(y)).argmax() == 1000
    f = np.concatenate([np.full(500, 200.0), np.full(500, 400.0)])
    z = Oscillator("sine", 8000).update_many(f)
    assert np.max(np.abs(np.diff(z))) <= 2 * np.pi * 400 / 8000 * 1.01


def test_additive_harmonic_amplitudes_match_fourier_series():
    sr, f = 4096, 64.0
    saw = Oscillator("sawtooth_additive", sr).update_many(np.full(sr, f))
    sq = Oscillator("square_additive", sr).update_many(np.full(sr, f))
    A = np.abs(np.fft.rfft(saw)) / (sr / 2)
    B = np.abs(np.fft.rfft(sq)) / (sr / 2)
    for k in (1, 2, 3):
        assert A[int(f) * k] == pytest.approx(2 / (np.pi * k), rel=1e-6)
    for k in (1, 3, 5):
        assert B[int(f) * k] == pytest.approx(4 / (np.pi * k), rel=1e-6)
    assert B[int(f) * 2] < 1e-9


def test_polyblep_has_less_aliasing_than_naive():
    n = SR
    naive = 2 * ((np.arange(n) * F0 / SR) % 1) - 1
    pb = Oscillator("sawtooth_polyblep", SR).update_many(np.full(n, F0))
    assert _off_harmonic_db(pb, F0, SR) < _off_harmonic_db(naive, F0, SR) - 10


def test_polyblep_sign_of_source_equation_two_is_worse():
    n = SR
    d = F0 / SR
    ph = (np.arange(n) * d) % 1
    r = lambda t: np.where(  # noqa: E731
        (t >= -1) & (t < 0),
        t * t + 2 * t + 1,
        np.where((t >= 0) & (t < 1), -t * t + 2 * t - 1, 0.0),
    )
    naive = 2 * ph - 1
    printed = naive - r(ph / d) + r((ph - 1) / d)
    assert _off_harmonic_db(printed, F0, SR) > _off_harmonic_db(naive, F0, SR) - 3


def test_oscillator_errors_and_contract():
    with pytest.raises(ValueError):
        Oscillator("triangle", 8000)
    o = Oscillator("square_additive", 8000)
    assert o.update(0.0).out == 0.0
    assert o.clone().get_params() == o.get_params()
    assert o.budget_s > 0 and o.memory_class == "O(1)"


def test_square_and_pulse_polyblep_have_less_aliasing_than_naive():
    n = SR
    ph = (np.arange(n) * F0 / SR) % 1
    for kind, duty in (("square_polyblep", 0.5), ("pulse_polyblep", 0.25)):
        naive = np.where(ph < duty, 1.0, -1.0)
        y = Oscillator(kind, SR, duty).update_many(np.full(n, F0))
        off_y = _off_harmonic_db(y - y.mean(), F0, SR)
        assert off_y < _off_harmonic_db(naive - naive.mean(), F0, SR) - 10
    y = Oscillator("pulse_polyblep", 8000, 0.25).update_many(np.full(8000, 100.0))
    assert y.mean() == pytest.approx(-0.5, abs=0.01)
    assert (
        Oscillator("square_polyblep", 8000, 0.1).update_many(np.full(80, 100.0))[10]
        == 1.0
    )
    for bad in (0.0, 1.0):
        with pytest.raises(ValueError):
            Oscillator("pulse_polyblep", 8000, bad)


def test_triangle_blamp_oscillator_is_the_block_triangle_two_samples_late():
    n = 3000
    st = Oscillator("triangle_blamp", SR).update_many(np.full(n + 2, F0))
    assert st[:2].tolist() == [0.0, 0.0]
    assert np.allclose(st[2:n], triangle_blamp(F0, SR, n)[: n - 2], atol=1e-9)
    with pytest.raises(ValueError):
        Oscillator("triangle_blamp", SR).update(SR / 4)


def test_white_noise_moments():
    x = white_noise(100000, 4)
    assert abs(x.mean()) < 0.02 and abs(x.var() - 1) < 0.02
    assert np.array_equal(white_noise(5, 1), white_noise(5, 1))


def _gain(b, a, f, sr):
    z = np.exp(-1j * 2 * np.pi * f / sr)
    return abs(np.polyval(b[::-1], z) / np.polyval(a[::-1], z))


def test_biquad_design_gains():
    sr, f0 = 48000, 1000.0
    b, a = lowpass(f0, sr)
    assert _gain(b, a, f0, sr) == pytest.approx(1 / np.sqrt(2), rel=1e-6)
    assert _gain(b, a, 1.0, sr) == pytest.approx(1.0, rel=1e-3)
    b, a = highpass(f0, sr)
    assert _gain(b, a, f0, sr) == pytest.approx(1 / np.sqrt(2), rel=1e-6)
    b, a = bandpass(f0, sr, 2.0)
    assert _gain(b, a, f0, sr) == pytest.approx(1.0, rel=1e-9)
    b, a = bandpass(f0, sr, 2.0, constant_peak=False)
    assert _gain(b, a, f0, sr) == pytest.approx(2.0, rel=1e-9)
    b, a = notch(f0, sr)
    assert _gain(b, a, f0, sr) < 1e-9
    b, a = peaking(f0, sr, 6.0)
    assert _gain(b, a, f0, sr) == pytest.approx(10 ** (6.0 / 20), rel=1e-9)
    b, a = lowshelf(f0, sr, 6.0)
    assert _gain(b, a, 1.0, sr) == pytest.approx(10 ** (6.0 / 20), rel=1e-3)
    assert _gain(b, a, 20000.0, sr) == pytest.approx(1.0, rel=1e-2)
    b, a = highshelf(f0, sr, 6.0)
    assert _gain(b, a, 1.0, sr) == pytest.approx(1.0, rel=1e-3)
    assert _gain(b, a, 20000.0, sr) == pytest.approx(10 ** (6.0 / 20), rel=1e-2)


def test_biquad_design_errors():
    with pytest.raises(ValueError):
        lowpass(24000.0, 48000.0)
    with pytest.raises(ValueError):
        lowpass(0.0, 48000.0)
    with pytest.raises(ValueError):
        peaking(1000.0, 48000.0, 3.0, q=0.0)


def test_biquad_filter_properties():
    b, a = lowpass(1000.0, 8000.0)
    x = np.random.default_rng(5).standard_normal(500)
    whole = Biquad(b, a).update_many(x)
    f = Biquad(b, a)
    parts = np.concatenate([f.update_many(x[:200]), f.update_many(x[200:])])
    assert np.array_equal(whole, parts)
    g = Biquad(b, a)
    assert np.array_equal(np.array([g.update(v).out for v in x]), whole)
    imp = Biquad(b, a).update_many(np.r_[1.0, np.zeros(9)])
    h = np.zeros(10)
    for n in range(10):
        h[n] = (b[n] if n < 3 else 0.0) - sum(
            a[k] * h[n - k] for k in (1, 2) if n - k >= 0
        )
    assert np.allclose(imp, h)
    assert Biquad(b, a).clone().get_params()["b"].shape == (3,)


def test_biquad_errors():
    with pytest.raises(ValueError):
        Biquad([1, 0], [1, 0, 0])
    with pytest.raises(ValueError):
        Biquad([1, 0, 0], [0, 0, 0])
    with pytest.raises(ValueError):
        Biquad([1, 0, 0], [1, -2.5, 1])


def test_hard_clip():
    x = np.array([-2.0, -0.3, 0.0, 0.7, 2.0])
    assert hard_clip(x, 0.5).tolist() == [-0.5, -0.3, 0.0, 0.5, 0.5]
    for bad in (0.0, 1.5, -1.0):
        with pytest.raises(ValueError):
            hard_clip(x, bad)


def test_resampler_hand_case_and_length():
    assert Resampler(2.0).update_many([0.0, 1.0, 2.0, 3.0, 4.0]).tolist() == [0.0, 2.0]
    assert Resampler(0.5).update_many([0.0, 2.0, 4.0]).tolist() == [0.0, 1.0, 2.0, 3.0]
    x = np.sin(2 * np.pi * 1000 * np.arange(4410) / 44100)
    y = resample(Audio(x, 44100), 2.0)
    assert y.sr == 22050 and abs(y.n - len(x) / 2) <= 1
    assert abs(np.abs(np.fft.rfft(y.x)).argmax() * y.sr / y.n - 1000) < 30


def test_resampler_block_equals_whole_and_ratio_change_continuous():
    x = np.random.default_rng(6).standard_normal(1000)
    whole = Resampler(1.37).update_many(x)
    r = Resampler(1.37)
    parts = np.concatenate([r.update_many(x[i : i + 97]) for i in range(0, 1000, 97)])
    assert np.allclose(whole, parts[: len(whole)])
    s = np.sin(2 * np.pi * 50 * np.arange(2000) / 8000)
    rr = Resampler(1.0)
    a = rr.update_many(s[:1000])
    rr.set_ratio(1.5)
    b = rr.update_many(s[1000:])
    y = np.concatenate([a, b])
    assert np.max(np.abs(np.diff(y))) <= 2 * np.pi * 50 / 8000 * 1.5 * 1.05
    assert np.allclose(Resampler(1.0).update_many(s)[:100], s[:100])
    with pytest.raises(ValueError):
        Resampler(0.0)


def _euclid_update(V, W, H):
    n, r = W.shape
    m = H.shape[1]
    WtV = np.zeros((r, m))
    WtWH = np.zeros((r, m))
    WtW = np.zeros((r, r))
    for a in range(r):
        for b in range(r):
            WtW[a, b] = sum(W[i, a] * W[i, b] for i in range(n))
    for a in range(r):
        for u in range(m):
            WtV[a, u] = sum(W[i, a] * V[i, u] for i in range(n))
            WtWH[a, u] = sum(WtW[a, b] * H[b, u] for b in range(r))
    return H * WtV / WtWH


def test_nmf_euclidean_update_equals_loop_reference():
    rng = np.random.default_rng(0)
    V = rng.random((2, 2)) + 0.1
    g = np.random.default_rng(0)
    W0 = g.random((2, 2)) + 1e-12
    H0 = g.random((2, 2)) + 1e-12
    W, H, _ = nmf(V, 2, 1, "euclidean", seed=0)
    H_ref = _euclid_update(V, W0, H0)
    W_ref = W0 * (V @ H_ref.T) / (W0 @ H_ref @ H_ref.T)
    assert np.allclose(H, H_ref) and np.allclose(W, W_ref)


@pytest.mark.parametrize("cost", ["euclidean", "divergence"])
def test_nmf_cost_non_increasing_and_exact_factorisation(cost):
    rng = np.random.default_rng(1)
    V = rng.random((12, 3)) @ rng.random((3, 20))
    W, H, hist = nmf(V, 3, 300, cost, seed=1)
    assert all(hist[i + 1] <= hist[i] + 1e-9 for i in range(len(hist) - 1))
    assert np.linalg.norm(V - W @ H) / np.linalg.norm(V) < 0.05
    assert W.min() >= 0 and H.min() >= 0


def test_nmf_errors():
    with pytest.raises(ValueError):
        nmf(np.array([[1.0, -1.0]]), 1)
    with pytest.raises(ValueError):
        nmf(np.ones((2, 2)), 1, cost="kl")


def test_separate_two_tones_in_time():
    sr = 8000
    t = np.arange(sr * 2) / sr
    x = np.where(t < 1, np.sin(2 * np.pi * 440 * t), np.sin(2 * np.pi * 1000 * t))
    parts = separate(Audio(x, sr), 2, 512, 256, 200)
    peaks = sorted(np.abs(np.fft.rfft(c)).argmax() * sr / len(c) for c in parts)
    assert abs(peaks[0] - 440) < 20 and abs(peaks[1] - 1000) < 20
    with pytest.raises(ValueError):
        separate(Audio(np.zeros((100, 2)), sr), 2)


def test_pitch_units():
    assert freq_to_cents(880.0, 440.0) == pytest.approx(1200.0)
    assert freq_to_cents(440.0, 440.0) == 0.0
    f = np.array([261.63, 440.0, 1234.5])
    assert np.allclose(cents_to_freq(freq_to_cents(f, 440.0), 440.0), f, rtol=1e-12)
    for k in range(12):
        j, o, d = freq_to_note(440.0 * 2 ** (k / 12), 440.0)
        assert (j, o) == (k, 0) and abs(d) < 1e-9
    j, o, d = freq_to_note(440.0 * 2 ** (-3 / 12 + 0.2 / 12), 440.0)
    assert (j, o) == (9, -1) and d == pytest.approx(20.0)
    assert note_to_freq(9, -1, 20.0, 440.0) == pytest.approx(
        440.0 * 2 ** (-3 / 12 + 0.2 / 12)
    )
    names = tuple("A A# B C C# D D# E F F# G G#".split())
    assert freq_to_note(523.2511306011972, 440.0, names)[3] == "C"
    with pytest.raises(ValueError):
        freq_to_cents(-1.0, 440.0)
    with pytest.raises(ValueError):
        note_to_freq(12, 0, 0.0, 440.0)
    with pytest.raises(ValueError):
        freq_to_note(440.0, 440.0, ("A",))


@pytest.mark.parametrize(
    "name",
    [
        "generators",
        "biquad",
        "clipping",
        "resample",
        "nmf",
        "pitch_units",
        "streams",
        "features",
    ],
)
def test_doctests(name):
    mod = importlib.import_module(f"dense_armor.utility.audio.{name}")
    assert doctest.testmod(mod, extraglobs={"np": np, "Audio": Audio}).failed == 0


_B, _A = lowpass(1000.0, 8000.0)
_STATEFUL = {
    "sine": (lambda: Oscillator("sine", 8000), True),
    "triangle_blamp": (lambda: Oscillator("triangle_blamp", 8000), True),
    "pulse": (lambda: Oscillator("pulse_polyblep", 8000, 0.3), True),
    "biquad": (lambda: Biquad(_B, _A), False),
    "fir": (lambda: FIR([0.5, 0.25, 0.25]), False),
    "delay": (lambda: DelayLine(5), False),
    "karplus": (lambda: KarplusStrong(20, 1), False),
    "stft": (lambda: StreamingSTFT(16, 8), False),
    "resampler": (lambda: Resampler(1.5), False),
    "adsr": (lambda: ADSR(0.1, 0.1, 0.5, 0.2, 0.5, 100), False),
}


@pytest.mark.parametrize("name", list(_STATEFUL))
def test_state_dict_restores_behaviour(name):
    make, is_freq = _STATEFUL[name]
    rng = np.random.default_rng(7)
    first, second = (
        (rng.uniform(100, 900, n) if is_freq else rng.standard_normal(n))
        for n in (37, 60)
    )
    o = make()
    o.update_many(first)
    c = make()
    c.load_state_dict(o.state_dict())
    got = np.asarray(c.update_many(second))
    assert got.size > 0 and np.allclose(got, np.asarray(o.update_many(second)))
    fresh = np.asarray(make().update_many(second))
    assert got.shape != fresh.shape or not np.allclose(got, fresh)


def test_root_contract_clone_and_memory_class():
    for make, _ in _STATEFUL.values():
        o = make()
        assert type(o.clone()) is type(o)
        assert o.budget_s > 0 and o.memory_class in ("O(1)", "O(window)")
    assert AudioFeatures(4).memory_class == "O(1)"
