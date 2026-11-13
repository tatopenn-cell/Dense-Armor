"""Compressor, limiter and soft clipper."""

import numpy as np
import pytest

from dense_armor.core.audio.dynamics import Compressor, gain_curve

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
