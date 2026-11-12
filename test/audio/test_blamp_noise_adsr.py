import doctest
import importlib

import numpy as np
import pytest

from dense_armor.utility.signals.audio import (
    ADSR,
    blamp_residual,
    hard_clip,
    hard_clip_blamp,
    pink_noise,
    triangle_blamp,
)

SR = 44100
F0 = 3141.5926


def _off_harmonic_db(x, f0, sr):
    X = np.abs(np.fft.rfft(x * np.hanning(len(x)))) ** 2
    mask = np.zeros(len(X), bool)
    for h in range(1, int(sr / 2 / f0) + 1):
        c = int(round(h * f0))
        mask[max(0, c - 3) : c + 4] = True
    return 10 * np.log10(X[~mask].sum() / X.sum())


def test_residual_peak_support_and_smoothness():
    assert float(blamp_residual(0.0)) == pytest.approx(7 / 30)
    assert float(blamp_residual(2.0)) == 0.0 and float(blamp_residual(-2.5)) == 0.0
    assert float(blamp_residual(1.0)) == pytest.approx(1 / 120)
    t = np.linspace(-1.99, 1.99, 801)
    r = blamp_residual(t)
    assert np.allclose(r, r[::-1], atol=1e-12)
    assert np.max(np.abs(np.diff(r))) < 0.01


def test_residual_is_second_integral_of_bspline_minus_ramp():
    t = np.linspace(-3, 3, 24001)
    h = t[1] - t[0]
    b = np.where(
        np.abs(t) < 1,
        2 / 3 - t**2 + np.abs(t) ** 3 / 2,
        np.where(np.abs(t) < 2, (2 - np.abs(t)) ** 3 / 6, 0.0),
    )
    step = np.cumsum(b) * h
    ramp = np.cumsum(step) * h
    ref = ramp - np.maximum(t, 0.0)
    ref -= ref[0]
    got = blamp_residual(t)
    assert np.max(np.abs(got - ref)) < 2e-3


def test_triangle_blamp_aliasing_and_shape():
    n = SR
    p = (np.arange(n) * F0 / SR) % 1
    naive = 1 - 4 * np.abs(p - 0.5)
    tb = triangle_blamp(F0, SR, n)
    assert _off_harmonic_db(tb, F0, SR) < _off_harmonic_db(naive, F0, SR) - 5
    assert np.max(np.abs(tb)) <= 1.0 + 1e-9
    assert np.max(np.abs(tb - naive)) < 4 * 4 * F0 / SR * 7 / 30
    wrong = naive.copy()
    assert _off_harmonic_db(wrong, F0, SR) > _off_harmonic_db(tb, F0, SR)


def test_triangle_blamp_corners_are_rounded_and_errors():
    tb = triangle_blamp(1000.0, 44100, 441)
    assert tb[0] > -1.0 and tb.min() > -1.0
    with pytest.raises(ValueError):
        triangle_blamp(0.0, 44100, 10)
    with pytest.raises(ValueError):
        triangle_blamp(20000.0, 44100, 10)


def test_hard_clip_blamp_aliasing_and_levels():
    t = np.arange(SR) / SR
    s = 2 * np.sin(2 * np.pi * F0 * t)
    for level in (1.0, 0.7):
        plain = hard_clip(s, level)
        fixed = hard_clip_blamp(s, level)
        assert _off_harmonic_db(fixed, F0, SR) < _off_harmonic_db(plain, F0, SR) - 5
    assert np.array_equal(
        hard_clip_blamp(np.array([0.1, 0.2, 0.1, 0.0, -0.1, 0.1]), 1.0),
        [0.1, 0.2, 0.1, 0.0, -0.1, 0.1],
    )
    for bad in (0.0, 1.5):
        with pytest.raises(ValueError):
            hard_clip_blamp(s, bad)


def test_pink_noise_slopes_and_errors():
    for beta in (1.0, 2.0):
        x = pink_noise(1 << 16, 3, beta)
        X = np.abs(np.fft.rfft(x)) ** 2
        k = np.arange(1, len(X))
        slope = np.polyfit(np.log(k), np.log(X[1:]), 1)[0]
        assert abs(slope + beta) < 0.05
    x = pink_noise(1 << 12, 1)
    assert abs(x.mean()) < 1e-9 and x.std() == pytest.approx(1.0)
    assert pink_noise(7, 2).shape == (7,)
    assert np.array_equal(pink_noise(100, 5), pink_noise(100, 5))
    with pytest.raises(ValueError):
        pink_noise(1)


def test_adsr_segments_and_errors():
    e = ADSR(0.1, 0.2, 0.5, 0.3, 1.0, 1000)
    pts = [
        (0.0, 0.0),
        (0.05, 0.5),
        (0.1, 1.0),
        (0.2, 0.75),
        (0.3, 0.5),
        (0.7, 0.5),
        (1.0, 0.5),
    ]
    for t, v in pts:
        assert e.value(t) == pytest.approx(v)
    assert e.value(1.15) == pytest.approx(0.25)
    assert e.value(1.3) == pytest.approx(0.0) and e.value(5.0) == pytest.approx(0.0)
    assert e.value(-0.2) == 0.0
    y = ADSR(0.1, 0.2, 0.5, 0.3, 1.0, 1000).update_many(np.zeros(1500))
    g = ADSR(0.1, 0.2, 0.5, 0.3, 1.0, 1000)
    assert np.array_equal(np.array([g.update().out for _ in range(1500)]), y)
    assert y.max() == pytest.approx(1.0) and y[-1] == 0.0
    assert e.clone().get_params()["t_atk"] == 0.1
    for kw in ({"t_atk": 0.0}, {"t_dec": -1.0}, {"v_sus": 2.0}, {"t_off": 0.1}):
        with pytest.raises(ValueError):
            ADSR(
                **{
                    **dict(
                        t_atk=0.1, t_dec=0.2, v_sus=0.5, t_rel=0.3, t_off=1.0, sr=1000
                    ),
                    **kw,
                }
            )


@pytest.mark.parametrize("name", ["blamp", "generators"])
def test_doctests(name):
    mod = importlib.import_module(f"dense_armor.utility.signals.audio.{name}")
    assert doctest.testmod(mod, extraglobs={"np": np}).failed == 0
