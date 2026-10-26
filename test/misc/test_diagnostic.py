"""Tests for the differential analysis of two audio signals."""

import numpy as np
import pytest
from scipy.io import wavfile

from dense_armor.utility.misc.diagnostic import diag


def _write_wav(path, data, sr=44100):
    wavfile.write(path, sr, data.astype(np.float32))


def test_identical_arrays_maximum_fidelity():
    signal = np.random.default_rng(0).normal(size=2000).astype(np.float32)
    result = diag(signal.copy(), signal.copy())
    assert result is not None
    assert result["fidelity"] == 100.0
    assert result["removed_energy"] == 0.0
    assert result["peak_distortion_db"] == -99.0
    assert result["verdict"] == "micro_edits"


def test_noise_added_high_but_not_perfect_fidelity():
    rng = np.random.default_rng(1)
    original = rng.normal(size=5000).astype(np.float32)
    noisy = original + rng.normal(scale=0.01, size=5000).astype(np.float32)

    result = diag(original, noisy)

    assert result is not None
    assert 90.0 < result["fidelity"] < 100.0
    assert result["removed_energy"] > 0.0
    assert 0.0 <= result["modulation_rate"] <= 100.0


def test_truncation_to_shortest_length():
    a = np.ones(100, dtype=np.float32)
    b = np.ones(50, dtype=np.float32)
    result = diag(a, b)
    assert result is not None


def test_stereo_to_mono_by_channel_mean():
    stereo = np.tile(np.array([[0.5, 0.5]], dtype=np.float32), (1000, 1))
    result = diag(stereo, stereo.copy())
    assert result is not None
    assert result["fidelity"] == 100.0


def test_real_wav_files(tmp_path):
    rng = np.random.default_rng(2)
    original = rng.normal(size=4000).astype(np.float32)
    path_a = str(tmp_path / "original.wav")
    path_b = str(tmp_path / "filtered.wav")
    _write_wav(path_a, original)
    _write_wav(path_b, original * 0.9)

    result = diag(path_a, path_b)

    assert result is not None
    assert 0.0 <= result["modulation_rate"] <= 100.0
    assert 95.0 < result["fidelity"] < 100.0


def test_missing_file_returns_none():
    result = diag("missing_a.wav", "missing_b.wav")
    assert result is None


def test_strong_alteration_aggressive_verdict():
    rng = np.random.default_rng(3)
    original = rng.normal(size=2000).astype(np.float32)
    altered = rng.normal(size=2000).astype(np.float32)

    result = diag(original, altered)

    assert result is not None
    assert result["fidelity"] < 95.0
    assert result["verdict"] == "aggressive"


def test_mixed_arguments_raises():
    arr = np.zeros(100, dtype=np.float32)
    with pytest.raises(TypeError):
        diag("a.wav", arr)
    with pytest.raises(TypeError):
        diag(arr, "a.wav")


def test_different_sample_rates_raise(tmp_path):
    sig = (np.sin(np.linspace(0, 20, 400)) * 100 + 128).astype(np.uint8)
    a = tmp_path / "a.wav"
    b = tmp_path / "b.wav"
    wavfile.write(a, 8000, sig)
    wavfile.write(b, 16000, sig)
    with pytest.raises(ValueError):
        diag(str(a), str(b))
