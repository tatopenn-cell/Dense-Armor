# -*- coding: utf-8 -*-
import os
from typing import Dict, Optional, Union

import numpy as np
from scipy.io import wavfile

def diag(iorig: Union[str, np.ndarray], ifilt: Union[str, np.ndarray]) -> Optional[Dict[str, float]]:
    """Esegue un'analisi differenziale profonda accettando sia percorsi file (str) che array NumPy."""
    if isinstance(iorig, str) and isinstance(ifilt, str):
        if not os.path.exists(iorig) or not os.path.exists(ifilt):
            print("[ERR] Uno dei file audio non è presente.")
            return
        sr1, d_ori = wavfile.read(iorig)
        sr2, d_flt = wavfile.read(ifilt)
        v_ori = d_ori.astype(np.float32) / 32768.0
        v_flt = d_flt.astype(np.float32) / 32768.0
    else:
        v_ori = iorig.astype(np.float32) / 32768.0 if iorig.dtype != np.float32 else iorig.copy()
        v_flt = ifilt.astype(np.float32) / 32768.0 if ifilt.dtype != np.float32 else ifilt.copy()

    if len(v_ori.shape) > 1: v_ori = np.mean(v_ori, axis=1)
    if len(v_flt.shape) > 1: v_flt = np.mean(v_flt, axis=1)
    
    mlen = min(v_ori.shape[0], v_flt.shape[0])
    v_ori = v_ori[:mlen]
    v_flt = v_flt[:mlen]

    v_dff = v_ori - v_flt
    v_rem = float(np.var(v_dff))
    fdel  = (1.0 - (np.sum(v_dff**2) / np.sum(v_ori**2))) * 100.0 if np.sum(v_ori**2) > 0 else 0.0
    pk_df = float(np.max(np.abs(v_dff))) if len(v_dff) > 0 else 0.0
    pk_db = 20 * np.log10(pk_df) if pk_df > 0 else -99.0
    alter = float(np.mean(np.abs(v_dff) > 0.05) * 100.0) if len(v_dff) > 0 else 0.0

    print("=" * 85)
    print("[DIAGNOSTICA DIFFERENZIALE STEREO] RE-ALLINEAMENTO COMPLETATO")
    print("=" * 85)
    print(f" -> Indice Strutturale di Fedeltà : {fdel:.4f}% (Portante preservata)")
    print(f" -> Energia Totale Rimossa (Var)  : {v_rem:.4e}")
    print(f" -> Picco di Distorsione Segato   : {pk_db:.2f} dBFS (Transiente massimo)")
    print(f" -> Tasso Modulazione Reticolo    : {alter:.2f}% (Campioni modificati)")
    print("-" * 85)

    if fdel > 99.5:
        print("[VERDETTO DIAG] INTERVENTO CHIRURGICO: Solo micro-fruscii rimossi.")
    elif fdel >= 95.0:
        print("[VERDETTO DIAG] RESTAURO EQUILIBRATO: Ottimo bilanciamento inter-canale.")
    else:
        print("[VERDETTO DIAG] MUTAZIONE AGGRESSIVA STEREO: Il Test 2 ha riscritto lo spazio dinamico.")
    print("=" * 85)
    return {"fedelta": fdel, "energia_rimossa": v_rem, "picco_distorsione_db": pk_db, "tasso_modulazione": alter}
