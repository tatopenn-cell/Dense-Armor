import os

import numpy as np
from scipy.io import wavfile


def anwav(fpath: str) -> None:
    """Analizza il file wav verificando i parametri di picco e dinamica.

    Il loudness e' una stima: il livello RMS in dBFS piu' 3 dB. Non e' il
    LUFS di ITU-R BS.1770, che usa un filtro di ponderazione e un gating.
    """
    if not os.path.exists(fpath):
        print(f"[ERR] File {fpath} non trovato!")
        return

    srate, data = wavfile.read(fpath)
    if np.issubdtype(data.dtype, np.floating):
        scala = 1.0
    elif data.dtype == np.int16:
        scala = 32768.0
    elif data.dtype == np.int32:
        scala = 2147483648.0
    else:
        scala = float(np.iinfo(data.dtype).max) + 1.0
    audio = data.astype(np.float32) / scala

    # Calcolo parametri essenziali
    mxval = np.max(np.abs(audio))
    p_db = 20 * np.log10(mxval) if mxval > 0 else -99.0
    rms = np.sqrt(np.mean(audio**2))
    r_db = 20 * np.log10(rms) if rms > 0 else -99.0
    lufs = r_db + 3.0
    crest = p_db - r_db

    print(f" -> File                      : {fpath}")
    print(f" -> Picco Massimo             : {p_db:.2f} dBFS")
    print(f" -> Volume Medio RMS          : {r_db:.2f} dBFS")
    print(f" -> Loudness (stima RMS + 3 dB): {lufs:.1f} dB")
    print(f" -> Fattore Cresta (Dinamica) : {crest:.2f} dB")
    print("-" * 85)

    print("[VERDETTO STANDARD]:")
    if p_db > -1.0:
        print("   AVVISO: Il picco supera i -1.0 dB. Rischio distorsione.")
    else:
        print("   CONFORME (Peak): Picco in sicurezza sotto i -1.0 dB.")

    if lufs > -7.0:
        print("   AVVISO: Volume molto spinto da Club.")
    elif lufs < -16.0:
        print("   AVVISO: Traccia troppo silenziosa.")
    else:
        print("   CONFORME (Loudness): Rispetta i target standard.")

    if crest < 6.0:
        print("   AVVISO: Traccia troppo schiacciata. Manca impatto.")
    else:
        print("   CONFORME (Dinamica): Mantiene l'impatto analogico.")
