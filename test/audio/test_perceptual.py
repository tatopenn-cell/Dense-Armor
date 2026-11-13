"""The mel scale, the mel filterbank, the mel spectrogram and the MFCC."""

import numpy as np
import pytest

from dense_armor.core.audio.perceptual import (
    hz_to_mel,
    mel_filterbank,
    mel_to_hz,
)


def test_the_slaney_log_step_is_the_factor_of_the_auditory_toolbox():
    ratio = mel_to_hz(30.0) / mel_to_hz(29.0)
    assert ratio == pytest.approx(1.0711703, abs=1e-7)
    assert ratio == pytest.approx(6.4 ** (1 / 27), abs=1e-12)


def test_the_scale_is_linear_below_one_khz_and_logarithmic_above():
    assert hz_to_mel(1000.0) == pytest.approx(15.0, abs=1e-12)
    assert hz_to_mel(200.0) == pytest.approx(3.0, abs=1e-12)
    assert hz_to_mel(6400.0) == pytest.approx(15.0 + 27.0, abs=1e-9)
    assert mel_to_hz(15.0 + 27.0) == pytest.approx(6400.0, abs=1e-6)


def test_hz_to_mel_and_back_on_an_array_and_on_a_number():
    f = np.array([0.0, 100.0, 999.0, 1000.0, 1001.0, 4000.0, 8000.0])
    assert np.allclose(mel_to_hz(hz_to_mel(f)), f, atol=1e-9)
    assert isinstance(hz_to_mel(440.0), float) and isinstance(mel_to_hz(7.0), float)
    assert hz_to_mel(f).shape == f.shape


def test_the_filterbank_shape_sign_and_unit_area_of_every_filter():
    fb = mel_filterbank(40, 1024, 16000.0)
    assert fb.shape == (40, 513)
    assert (fb >= 0.0).all()
    area = fb.sum(axis=1) * (16000.0 / 1024.0)
    assert np.allclose(area, 1.0, atol=0.03)


def test_the_filters_are_triangles_that_overlap_with_their_neighbours():
    fb = mel_filterbank(10, 512, 8000.0)
    peak = fb.argmax(axis=1)
    assert (np.diff(peak) > 0).all()
    assert ((fb[:-1] > 0) & (fb[1:] > 0)).any(axis=1).all()


def test_fmin_and_fmax_limit_the_band():
    fb = mel_filterbank(8, 512, 8000.0, fmin=1000.0, fmax=3000.0)
    cols = np.nonzero(fb.sum(axis=0))[0]
    bins = np.linspace(0.0, 4000.0, 257)
    assert (
        bins[cols.min()] >= 1000.0 - 8000.0 / 512
        and bins[cols.max()] <= 3000.0 + 8000.0 / 512
    )


@pytest.mark.parametrize("args", [(0, 512, 8000.0), (8, 0, 8000.0)])
def test_the_filterbank_refuses_a_size_below_one(args):
    with pytest.raises(ValueError, match=">= 1"):
        mel_filterbank(*args)


def test_the_filterbank_refuses_an_empty_band():
    with pytest.raises(ValueError, match="above fmin"):
        mel_filterbank(8, 512, 8000.0, fmin=3000.0, fmax=3000.0)


def test_hz_to_mel_matches_librosa():
    librosa = pytest.importorskip("librosa")
    f = np.linspace(0.0, 8000.0, 2001)
    assert np.max(np.abs(hz_to_mel(f) - librosa.hz_to_mel(f))) < 1e-9
    m = np.linspace(0.0, 45.0, 901)
    assert np.max(np.abs(mel_to_hz(m) - librosa.mel_to_hz(m))) < 1e-9


@pytest.mark.parametrize(
    ("n_mels", "n_fft", "sr", "fmin", "fmax"),
    [
        (128, 2048, 8000, 0.0, None),
        (40, 1024, 16000, 133.33, 6855.5),
        (13, 512, 22050, 50.0, 8000.0),
    ],
)
def test_mel_filterbank_matches_librosa(n_mels, n_fft, sr, fmin, fmax):
    librosa = pytest.importorskip("librosa")
    ref = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax)
    ours = mel_filterbank(n_mels, n_fft, sr, fmin, fmax)
    assert ours.shape == ref.shape
    assert np.max(np.abs(ours - ref)) < 1e-6
