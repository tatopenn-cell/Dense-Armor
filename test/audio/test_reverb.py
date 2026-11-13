"""The comb and all-pass filters of Schroeder and the reverberator."""

import numpy as np
import pytest

from dense_armor.core.audio.reverb import (
    ALLPASS_DELAYS_S,
    ALLPASS_GAIN,
    COMB_DELAYS_S,
    AllpassFilter,
    CombFilter,
    SchroederReverb,
    comb_gain,
    reverberation_time,
)

SR = 8000.0


def test_the_comb_gain_gives_the_reverberation_time_of_the_paper():
    g = comb_gain(0.1, 2.0)
    assert g == pytest.approx(0.7079, abs=1e-4)
    assert 3 * 0.1 / np.log10(1 / g) == pytest.approx(2.0, abs=1e-12)


def test_the_comb_gain_refuses_a_time_that_is_not_positive():
    for args in ((0.0, 1.0), (0.1, 0.0), (-0.1, 1.0)):
        with pytest.raises(ValueError, match="must be positive"):
            comb_gain(*args)


def test_the_comb_impulse_response_is_an_echo_train():
    c = CombFilter(5, 0.6)
    h = c.update_many(np.r_[1.0, np.zeros(24)])
    expected = np.zeros(25)
    for k in range(1, 5):
        expected[5 * k] = 0.6 ** (k - 1)
    assert np.allclose(h, expected, atol=1e-15)


def test_the_comb_matches_scipy():
    signal = pytest.importorskip("scipy.signal")
    d, g = 37, 0.83
    x = np.random.default_rng(0).standard_normal(500)
    b = np.r_[np.zeros(d), 1.0]
    a = np.r_[1.0, np.zeros(d - 1), -g]
    ref = signal.lfilter(b, a, x)
    assert np.max(np.abs(CombFilter(d, g).update_many(x) - ref)) < 1e-12


def test_the_comb_with_damping_matches_its_transfer_function():
    signal = pytest.importorskip("scipy.signal")
    d, g, damp = 7, 0.8, 0.4
    x = np.random.default_rng(1).standard_normal(400)
    b = np.r_[np.zeros(d), 1.0, -damp]
    a = np.r_[1.0, -damp, np.zeros(d - 2), -g * (1 - damp)]
    ref = signal.lfilter(b, a, x)
    assert np.max(np.abs(CombFilter(d, g, damp).update_many(x) - ref)) < 1e-12


def test_the_allpass_impulse_response_is_the_one_of_the_paper():
    ap = AllpassFilter(4, 0.7)
    h = ap.update_many(np.r_[1.0, np.zeros(15)])
    expected = np.zeros(16)
    expected[0] = -0.7
    for k in range(1, 4):
        expected[4 * k] = (1 - 0.7**2) * 0.7 ** (k - 1)
    assert np.allclose(h, expected, atol=1e-15)


def test_the_allpass_matches_scipy():
    signal = pytest.importorskip("scipy.signal")
    d, g = 29, 0.7
    x = np.random.default_rng(2).standard_normal(500)
    b = np.r_[-g, np.zeros(d - 1), 1.0]
    a = np.r_[1.0, np.zeros(d - 1), -g]
    assert (
        np.max(np.abs(AllpassFilter(d, g).update_many(x) - signal.lfilter(b, a, x)))
        < 1e-12
    )


def test_the_allpass_has_unit_gain_at_every_frequency():
    signal = pytest.importorskip("scipy.signal")
    d, g = 13, 0.7
    h = AllpassFilter(d, g).update_many(np.r_[1.0, np.zeros(4000)])
    _, resp = signal.freqz(h, worN=512)
    assert np.max(np.abs(np.abs(resp) - 1.0)) < 1e-9


def test_the_structure_is_the_one_of_the_paper():
    ms = np.array(COMB_DELAYS_S) * 1000.0
    assert len(ms) == 4 and ms.min() >= 30.0 and ms.max() <= 45.0
    assert ms.max() / ms.min() <= 1.5
    assert ALLPASS_DELAYS_S == (0.005, 0.0017) and ALLPASS_GAIN == 0.7
    r = SchroederReverb(sr=SR)
    assert len(r._combs) == 4 and len(r._allpasses) == 2


def _t60(room, damping=0.0, sr=SR):
    r = SchroederReverb(room, damping, 1.0, 0.0, sr)
    n = int(sr * (1.6 * r.t60_s + 0.2))
    h = r.update_many(np.r_[1.0, np.zeros(n - 1)])
    return r.t60_s, reverberation_time(h, sr)


@pytest.mark.parametrize("room", [0.2, 0.5, 0.9])
def test_the_reverberation_time_is_the_one_asked(room):
    asked, measured = _t60(room)
    assert measured == pytest.approx(asked, rel=0.10)


def test_damping_shortens_the_reverberation():
    _, flat = _t60(0.5, 0.0)
    _, damped = _t60(0.5, 0.8)
    assert damped < 0.9 * flat


def test_the_reverberation_time_of_a_known_exponential():
    sr = 1000.0
    h = np.exp(-np.arange(4000) / sr * 3.0 * np.log(10) / 1.0)
    assert reverberation_time(h, sr) == pytest.approx(1.0, rel=0.02)
    with pytest.raises(ValueError, match="35 dB"):
        reverberation_time(h[:50], sr)


def test_the_response_to_a_unit_impulse_is_finite_and_bounded():
    r = SchroederReverb(sr=SR)
    h = r.update_many(np.r_[1.0, np.zeros(20000)])
    assert np.isfinite(h).all() and np.abs(h).max() < 1.5
    long_noise = SchroederReverb(1.0, 0.0, 1.0, 1.0, SR).update_many(
        np.random.default_rng(0).uniform(-1, 1, 20000)
    )
    assert np.isfinite(long_noise).all() and np.abs(long_noise).max() < 40.0


def test_wet_and_dry_mix_the_direct_and_the_reverberated_sound():
    x = np.random.default_rng(3).standard_normal(300)
    assert np.allclose(SchroederReverb(0.5, 0.5, 0.0, 0.7, SR).update_many(x), 0.7 * x)
    h = SchroederReverb(0.5, 0.0, 1.0, 0.0, SR).update_many(np.r_[1.0, np.zeros(999)])
    assert h[0] == 0.0
    assert int(np.argmax(h != 0.0)) == round(COMB_DELAYS_S[0] * SR)


def test_the_parameters_are_checked_and_clone_and_state_work():
    for kw in ({"room_size": 1.5}, {"damping": 1.0}, {"wet": -1.0}, {"sr": 0.0}):
        with pytest.raises(ValueError, match="must be"):
            SchroederReverb(**kw)
    for bad in (
        lambda: CombFilter(0),
        lambda: CombFilter(5, 1.0),
        lambda: CombFilter(5, 0.5, 1.0),
        lambda: AllpassFilter(0),
        lambda: AllpassFilter(5, 1.0),
    ):
        with pytest.raises(ValueError, match="must be"):
            bad()
    r = SchroederReverb(0.3, 0.2, 0.4, 0.6, SR)
    assert r.clone().get_params() == r.get_params()
    x = np.random.default_rng(4).standard_normal(600)
    r.update_many(x[:300])
    twin = r.clone()
    twin.load_state_dict(r.state_dict())
    assert np.allclose(twin.update_many(x[300:]), r.update_many(x[300:]))
