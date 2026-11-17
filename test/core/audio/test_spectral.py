"""Spectral centroid, rolloff, flatness and flux."""

import numpy as np
import pytest

from dense_armor.core.audio.fourier import stft
from dense_armor.core.audio.spectral import (
    spectral_centroid,
    spectral_flatness,
    spectral_flux,
    spectral_rolloff,
)

SR = 8000.0


def _tone(f, n=8000):
    return np.sin(2 * np.pi * f * np.arange(n) / SR)


def test_the_centroid_and_the_rolloff_of_a_tone_are_at_its_frequency():
    for f in (500.0, 1500.0, 3000.0):
        c = spectral_centroid(_tone(f), SR, 1024, 256)
        r = spectral_rolloff(_tone(f), SR, 0.85, 1024, 256)
        assert abs(c.mean() - f) < 15.0
        assert abs(r.mean() - f) < 40.0


def test_the_centroid_is_the_weighted_mean_of_the_bin_frequencies():
    x = np.random.default_rng(0).standard_normal(3000)
    mag = np.abs(stft(x, 512, 128, "hann"))
    freq = np.arange(257) * SR / 512
    ref = (mag * freq).sum(axis=1) / mag.sum(axis=1)
    assert np.allclose(spectral_centroid(x, SR, 512, 128), ref, rtol=1e-12)


def test_the_rolloff_is_the_first_bin_that_reaches_the_fraction_of_the_total():
    x = np.random.default_rng(1).standard_normal(3000)
    mag = np.abs(stft(x, 512, 128, "hann"))
    cum = np.cumsum(mag, axis=1)
    got = spectral_rolloff(x, SR, 0.5, 512, 128)
    for m in range(len(mag)):
        k = int(round(got[m] * 512 / SR))
        assert cum[m, k] >= 0.5 * cum[m, -1]
        assert k == 0 or cum[m, k - 1] < 0.5 * cum[m, -1]


def test_silence_gives_zero_centroid_and_rolloff_and_flatness_one():
    z = np.zeros(2048)
    assert (spectral_centroid(z, SR, 512, 256) == 0.0).all()
    assert (spectral_rolloff(z, SR, 0.85, 512, 256) == 0.0).all()
    assert np.allclose(spectral_flatness(z, 512, 256), 1.0)


def test_flatness_is_low_for_a_tone_and_high_for_noise():
    noise = np.random.default_rng(2).standard_normal(8000)
    assert spectral_flatness(_tone(1000.0), 1024, 256).max() < 0.05
    assert spectral_flatness(noise, 1024, 256).mean() > 0.3
    f = spectral_flatness(noise, 1024, 256)
    assert ((f > 0.0) & (f <= 1.0)).all()


def test_flux_is_the_rectified_difference():
    x = np.random.default_rng(3).standard_normal(3000)
    mag = np.abs(stft(x, 512, 128, "hann"))
    ref = np.zeros(len(mag))
    for m in range(1, len(mag)):
        ref[m] = np.maximum(mag[m] - mag[m - 1], 0.0).sum()
    assert np.allclose(spectral_flux(x, 512, 128), ref, rtol=1e-12)


def test_flux_is_zero_for_a_steady_tone_and_large_at_an_onset():
    steady = spectral_flux(_tone(1000.0, 16000), 512, 512)
    x = np.concatenate([np.zeros(4096), _tone(1000.0, 4096)])
    f = spectral_flux(x, 512, 512)
    assert steady[1:].max() < 0.02 * f.max()
    assert f[:7].max() == 0.0 and int(f.argmax()) in (8, 9)


def test_a_short_signal_gives_no_frames_and_flux_of_one_frame_is_zero():
    short = np.zeros(100)
    for out in (
        spectral_centroid(short, SR, 512, 128),
        spectral_rolloff(short, SR, 0.85, 512, 128),
        spectral_flatness(short, 512, 128),
        spectral_flux(short, 512, 128),
    ):
        assert out.shape == (0,)
    assert spectral_flux(np.ones(512), 512, 128).tolist() == [0.0]


@pytest.mark.parametrize("bad", [0.0, 1.0, -0.1, 1.5])
def test_rolloff_refuses_a_fraction_outside_the_open_interval(bad):
    with pytest.raises(ValueError, match="roll_percent"):
        spectral_rolloff(np.zeros(2048), SR, bad)


@pytest.mark.parametrize(
    "name", ["spectral_centroid", "spectral_rolloff", "spectral_flatness"]
)
def test_the_features_match_librosa(name):
    librosa = pytest.importorskip("librosa")
    x = np.random.default_rng(4).standard_normal(8000)
    ref = getattr(librosa.feature, name)(
        y=x,
        **({"sr": 8000} if name != "spectral_flatness" else {}),
        n_fft=2048,
        hop_length=512,
        center=False,
    )[0]
    ours = {
        "spectral_centroid": lambda: spectral_centroid(x, SR),
        "spectral_rolloff": lambda: spectral_rolloff(x, SR),
        "spectral_flatness": lambda: spectral_flatness(x),
    }[name]()
    assert ours.shape == ref.shape
    assert np.max(np.abs(ours - ref)) < 1e-6
