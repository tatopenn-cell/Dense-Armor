# -*- coding: utf-8 -*-
import numpy as np
from scipy.io import wavfile

from dense_armor.utility.anwav import anwav


def _write_wav(path, amplitude=0.5, sr=44100, seconds=1, dtype=np.float32):
    t = np.linspace(0, seconds, int(sr * seconds), endpoint=False)
    data = (amplitude * np.sin(2 * np.pi * 440 * t)).astype(dtype)
    wavfile.write(path, sr, data)


def test_anwav_file_assente_stampa_errore_e_non_solleva(capsys):
    anwav("questo_file_non_esiste.wav")
    out = capsys.readouterr().out
    assert "[ERR]" in out


def test_anwav_wav_float_conforme_stampa_verdetto_conforme(tmp_path, capsys):
    path = str(tmp_path / "conforme.wav")
    _write_wav(path, amplitude=0.3)

    anwav(path)

    out = capsys.readouterr().out
    assert "Picco Massimo" in out
    assert "CONFORME (Peak)" in out


def test_anwav_wav_int16_gestisce_la_scala_intera(tmp_path, capsys):
    path = str(tmp_path / "int16.wav")
    _write_wav(path, amplitude=0.9, dtype=np.int16)
    _, raw = wavfile.read(path)
    assert raw.dtype == np.int16

    anwav(path)

    out = capsys.readouterr().out
    assert "dBFS" in out


def test_anwav_traccia_troppo_piatta_segnala_dinamica_schiacciata(tmp_path, capsys):
    path = str(tmp_path / "piatta.wav")
    sr = 44100
    data = np.full(sr, 0.5, dtype=np.float32)
    wavfile.write(path, sr, data)

    anwav(path)

    out = capsys.readouterr().out
    assert "Traccia troppo schiacciata" in out


def test_anwav_wav_int32_gestisce_la_scala_intera(tmp_path, capsys):
    path = str(tmp_path / "int32.wav")
    _write_wav(path, amplitude=0.9, dtype=np.int32)
    _, raw = wavfile.read(path)
    assert raw.dtype == np.int32

    anwav(path)

    assert "dBFS" in capsys.readouterr().out


def test_anwav_wav_uint8_usa_la_scala_generica_via_iinfo(tmp_path, capsys):
    path = str(tmp_path / "uint8.wav")
    sr = 8000
    t = np.linspace(0, 1, sr, endpoint=False)
    data = (128 + 100 * np.sin(2 * np.pi * 440 * t)).astype(np.uint8)
    wavfile.write(path, sr, data)

    anwav(path)

    assert "dBFS" in capsys.readouterr().out


def test_anwav_traccia_troppo_silenziosa(tmp_path, capsys):
    path = str(tmp_path / "silenziosa.wav")
    sr = 44100
    rng = np.random.default_rng(0)
    data = np.zeros(sr, dtype=np.float32)
    spike_idx = np.arange(0, sr, 1000)
    data[spike_idx] = 0.05  # picchi radi, RMS complessivo molto basso
    wavfile.write(path, sr, data)

    anwav(path)

    assert "Traccia troppo silenziosa" in capsys.readouterr().out


def test_anwav_traccia_troppo_spinta(tmp_path, capsys):
    path = str(tmp_path / "spinta.wav")
    sr = 44100
    data = np.full(sr, 0.95, dtype=np.float32)
    wavfile.write(path, sr, data)

    anwav(path)

    assert "Volume molto spinto" in capsys.readouterr().out


def test_anwav_dinamica_conforme_con_picchi_radi(tmp_path, capsys):
    path = str(tmp_path / "dinamica_ok.wav")
    sr = 44100
    data = np.zeros(sr, dtype=np.float32)
    spike_idx = np.arange(0, sr, 1000)
    data[spike_idx] = 0.9  # picco alto, RMS complessivo basso -> crest alto
    wavfile.write(path, sr, data)

    anwav(path)

    assert "CONFORME (Dinamica)" in capsys.readouterr().out
