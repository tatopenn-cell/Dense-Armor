# -*- coding: utf-8 -*-
import numpy as np
from scipy.io import wavfile

from dense_armor.utility.diagnostic import diag


def _write_wav(path, data, sr=44100):
    wavfile.write(path, sr, data.astype(np.float32))


def test_diag_su_array_identici_fedelta_massima():
    signal = np.random.default_rng(0).normal(size=2000).astype(np.float32)
    result = diag(signal.copy(), signal.copy())
    assert result["fedelta"] == 100.0
    assert result["energia_rimossa"] == 0.0


def test_diag_su_array_con_rumore_aggiunto_fedelta_alta_ma_non_perfetta():
    rng = np.random.default_rng(1)
    original = rng.normal(size=5000).astype(np.float32)
    noisy = original + rng.normal(scale=0.01, size=5000).astype(np.float32)

    result = diag(original, noisy)

    assert 90.0 < result["fedelta"] < 100.0
    assert result["energia_rimossa"] > 0.0


def test_diag_troncamento_alla_lunghezza_minima():
    a = np.ones(100, dtype=np.float32)
    b = np.ones(50, dtype=np.float32)
    result = diag(a, b)
    assert result is not None  # non deve sollevare per lunghezze diverse


def test_diag_converte_stereo_in_mono_facendo_la_media_dei_canali():
    stereo = np.tile(np.array([[0.5, 0.5]], dtype=np.float32), (1000, 1))
    result = diag(stereo, stereo.copy())
    assert result["fedelta"] == 100.0


def test_diag_su_file_wav_reali(tmp_path):
    rng = np.random.default_rng(2)
    original = rng.normal(size=4000).astype(np.float32)
    path_a = str(tmp_path / "originale.wav")
    path_b = str(tmp_path / "filtrato.wav")
    _write_wav(path_a, original)
    _write_wav(path_b, original * 0.9)

    result = diag(path_a, path_b)

    assert result is not None
    assert 0.0 <= result["tasso_modulazione"] <= 100.0


def test_diag_file_assente_ritorna_none(capsys):
    result = diag("assente_a.wav", "assente_b.wav")
    assert result is None
    assert "[ERR]" in capsys.readouterr().out


def test_diag_forte_alterazione_segnala_mutazione_aggressiva(capsys):
    rng = np.random.default_rng(3)
    original = rng.normal(size=2000).astype(np.float32)
    altered = rng.normal(size=2000).astype(np.float32)  # segnale scorrelato

    result = diag(original, altered)

    assert result["fedelta"] < 95.0
    assert "MUTAZIONE AGGRESSIVA" in capsys.readouterr().out
