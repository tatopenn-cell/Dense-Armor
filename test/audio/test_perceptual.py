"""The mel scale, the mel filterbank, the mel spectrogram and the MFCC."""

import numpy as np
import pytest

from dense_armor.core.audio.perceptual import (
    _dct_ii_matrix,
    hz_to_mel,
    mel_filterbank,
    mel_spectrogram,
    mel_to_hz,
    mfcc,
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


def _tone(f=440.0, sr=8000, n=8000):
    return np.sin(2 * np.pi * f * np.arange(n) / sr)


def test_the_mel_spectrogram_shape_and_that_a_tone_peaks_in_its_filter():
    sr = 8000.0
    mel = mel_spectrogram(_tone(440.0), sr, n_mels=40, n_fft=1024, hop=256)
    assert mel.shape == (1 + (8000 - 1024) // 256, 40)
    assert (mel >= 0.0).all()
    fb = mel_filterbank(40, 1024, sr)
    centre = fb.argmax(axis=1) * sr / 1024
    best = int(mel.mean(axis=0).argmax())
    assert abs(centre[best] - 440.0) < 60.0


def test_the_mel_spectrogram_is_the_filterbank_applied_to_the_power_of_the_stft():
    from dense_armor.core.audio.fourier import stft

    x = np.random.default_rng(0).standard_normal(3000)
    ref = (np.abs(stft(x, 512, 128, "hann")) ** 2) @ mel_filterbank(20, 512, 8000.0).T
    assert np.allclose(mel_spectrogram(x, 8000.0, 20, 512, 128), ref, rtol=1e-12)


def test_a_signal_shorter_than_one_frame_gives_zero_frames():
    assert mel_spectrogram(np.zeros(100), 8000.0, 20, 512, 128).shape == (0, 20)


def test_mel_spectrogram_matches_librosa():
    librosa = pytest.importorskip("librosa")
    x = np.random.default_rng(1).standard_normal(8000)
    ref = librosa.feature.melspectrogram(
        y=x, sr=8000, n_fft=2048, hop_length=512, n_mels=128, center=False
    )
    ours = mel_spectrogram(x, 8000.0)
    assert ours.shape == ref.T.shape
    assert np.max(np.abs(ours - ref.T)) / np.max(np.abs(ref)) < 1e-5


def test_the_dct_matrix_is_orthonormal_and_a_constant_has_only_the_first_coefficient():
    mat = _dct_ii_matrix(16, 16)
    assert np.allclose(mat @ mat.T, np.eye(16), atol=1e-12)
    out = _dct_ii_matrix(5, 16) @ np.full(16, 3.0)
    assert out[0] == pytest.approx(3.0 * np.sqrt(16.0), abs=1e-12)
    assert np.allclose(out[1:], 0.0, atol=1e-12)


def test_the_dct_matrix_is_the_orthonormal_dct_ii():
    fft = pytest.importorskip("scipy.fft")
    v = np.random.default_rng(2).standard_normal(40)
    ours = _dct_ii_matrix(13, 40) @ v
    assert np.max(np.abs(ours - fft.dct(v, type=2, norm="ortho")[:13])) < 1e-12


def test_mfcc_shape_and_the_floor_of_the_log():
    x = _tone(440.0)
    out = mfcc(x, 8000.0, n_mfcc=13, n_mels=40, n_fft=1024, hop=256)
    assert out.shape == (1 + (8000 - 1024) // 256, 13)
    mel = mel_spectrogram(x, 8000.0, 40, 1024, 256)
    log_mel = 10.0 * np.log10(np.maximum(mel, 1e-10))
    floored = np.maximum(log_mel, log_mel.max() - 80.0)
    assert np.allclose(out, floored @ _dct_ii_matrix(13, 40).T, atol=1e-9)
    no_floor = mfcc(x, 8000.0, 13, 40, 1024, 256, top_db=None)
    assert np.allclose(no_floor, log_mel @ _dct_ii_matrix(13, 40).T, atol=1e-9)


def test_mfcc_of_a_short_signal_has_no_frames_and_n_mfcc_is_checked():
    assert mfcc(np.zeros(100), 8000.0, 13, 40, 512, 128).shape == (0, 13)
    for bad in (0, 41):
        with pytest.raises(ValueError, match="between 1 and n_mels"):
            mfcc(_tone(), 8000.0, bad, 40, 512, 128)


def test_mfcc_matches_librosa():
    librosa = pytest.importorskip("librosa")
    x = np.random.default_rng(3).standard_normal(8000)
    ref = librosa.feature.mfcc(
        y=x, sr=8000, n_mfcc=13, n_mels=128, n_fft=2048, hop_length=512, center=False
    )
    ours = mfcc(x, 8000.0)
    assert ours.shape == ref.T.shape
    assert np.max(np.abs(ours - ref.T)) < 1e-4
