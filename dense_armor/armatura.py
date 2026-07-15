# -*- coding: utf-8 -*-
"""
armatura.py — L'ARMATURA: istanza indossabile dello scudo Sentinel.
Versione standalone: usa i core/ e utility/ di QUESTA cartella (shield_),
nessuna dipendenza da BERT o da altri progetti.

Uso rapido (PowerShell, da questa cartella):
    python armatura.py 1.2 1.3 9999 1.25 nan 1.3
    python armatura.py file_con_numeri.txt

Uso da codice (qualsiasi IA):
    from armatura import Armatura
    a = Armatura(livello_ia=0.0)        # 0=IA neonata (FILTRA) ... 1=matura (solo MARCA)
    pulito, K, anomalie = a.analizza(serie)
    pulito, K, anomalie = a.analizza(serie_oggi, riferimento=baseline_di_ieri)  # anti-deriva
"""
import os
import sys


import numpy as np
import jax.numpy as jnp

from .core.engine import AdaptiveSignalStabilizer   # scudo originale, intatto
from .utility.collatz import ABCollatz               # scudo originale, intatto
from .utility.metro import Metro                     # scaler anti-underflow, intatto


class Armatura:
    """Scudo indossabile con CLIP DINAMICO universale.

    livello_ia (0..1): quanto e' matura l'IA che lo indossa.
      0.0 -> clip pieno: lo scudo FILTRA il dato (corazza per IA neonata)
      1.0 -> clip zero:  lo scudo NON tocca il dato, solo MARCA (lente per IA matura)
    PAVIMENTO: la marcatura delle anomalie non viene MAI clippata — a nessun
    livello lo scudo tace (l'anomalia puo' ESSERE la risposta: mai cancellarla
    in silenzio). Il livello puo' essere misurato, non dichiarato:
    Armatura.livello_da_entropia(entropia, vocab_size).

    MEMORIA: lo scudo e' una funzione PURA, senza stato (leggero per design).
    La memoria appartiene all'IA che lo indossa: per vedere la DERIVA LENTA
    passa la tua baseline storica come `riferimento`.
    """

    def __init__(self, static_threshold=0.15, initial_damping=0.85, alpha=0.05,
                 soglia_anomalia=None, livello_ia=1.0):
        self.stab = AdaptiveSignalStabilizer(static_threshold=static_threshold,
                                             initial_damping=initial_damping, alpha=alpha)
        self.shield = ABCollatz(epsilon_target=1.0)
        self.soglia = soglia_anomalia
        self.livello_ia = float(min(1.0, max(0.0, livello_ia)))

    def set_livello(self, livello):
        """Aggiorna il livello dell'IA (0=debole -> filtra; 1=matura -> marca)."""
        self.livello_ia = float(min(1.0, max(0.0, livello)))

    @staticmethod
    def livello_da_entropia(entropia, vocab_size):
        """Livello MISURATO: 1 - entropia/log(V) (stessa formula dello sfogo dinamico)."""
        import math
        return max(0.0, min(1.0, 1.0 - entropia / (math.log(max(2, vocab_size)) + 1e-12)))

    def analizza(self, serie, riferimento=None):
        """serie: 1D (lista/array). Per tensori N-D: passare tensore.ravel()
        e rifare reshape dopo (lo scudo e' agnostico: la trasposizione basta).
        Ritorna (pulito, K, indici_anomalie)."""
        s = np.asarray(serie, dtype=np.float64).reshape(1, -1)
        s_jnp = jnp.array(s)

        f1 = self.stab.filter_batch_scenarios(s_jnp)          # Stadio 1
        f1 = jnp.where(jnp.isnan(f1), jnp.nan_to_num(s_jnp), f1)
        if riferimento is not None:
            rif = jnp.array(np.asarray(riferimento, dtype=np.float64).reshape(1, -1))
        else:
            # riferimento cieco = smoothing pesante (pattern SentinelCV2D):
            # mediana mobile immune agli spike, bersaglio per lo Stadio 2
            v = np.array(f1).ravel()
            n = v.size
            rif_np = np.empty(n)
            for _i in range(n):
                a, b = max(0, _i - 3), min(n, _i + 4)
                rif_np[_i] = np.median(v[a:b])
            rif = jnp.array(rif_np.reshape(1, -1))
        gt = self.shield.compute_damping_gating(f1, rif)      # Stadio 2
        gt = jnp.where(jnp.isnan(gt), 0.85, gt)

        # PERCEZIONE a K pieno (pavimento: mai clippata)
        K = np.array(gt).ravel()
        pulito_pieno = np.array(f1 - gt * (f1 - rif)).ravel()
        # INTERVENTO clippato dal livello dell'IA
        clip = 1.0 - self.livello_ia
        pulito = np.array(f1 - (gt * clip) * (f1 - rif)).ravel()

        grezza = s.ravel()
        dev = np.abs(np.nan_to_num(grezza) - pulito_pieno)
        soglia = self.soglia if self.soglia is not None else (dev.mean() + 2.0 * dev.std())
        anomalie = set(int(i) for i in np.where(dev > max(soglia, 1e-12))[0])

        # rilevatore robusto (mediana/MAD) per gli spike che lo Stadio 1 insegue
        finiti = grezza[np.isfinite(grezza)]
        if finiti.size >= 4:
            med = float(np.median(finiti))
            mad = float(np.median(np.abs(finiti - med))) or 1e-12
            z_rob = np.abs((np.nan_to_num(grezza, nan=med) - med) / (1.4826 * mad))
            anomalie |= set(int(i) for i in np.where(z_rob > 6.0)[0])

        anomalie |= {int(i) for i in np.where(~np.isfinite(grezza))[0]}
        return pulito, K, sorted(anomalie)

    def deriva(self, serie):
        """RILEVATORE DI DERIVA LENTA (usa METRO, come suggerito da Salvatore).

        La rana bollita: incrementi infinitesimi (es. +0.004/passo) invisibili
        allo scudo punto-per-punto. Qui si guarda la TENDENZA degli incrementi:
        se la loro media e' sistematica rispetto al rumore, c'e' deriva — e
        METRO (scaler logaritmico) ne risolve la scala anche quando e' sotto
        la precisione visibile. Funziona SENZA baseline storica (ma la baseline
        via `riferimento` in analizza() resta il metodo piu' forte).

        Ritorna (tasso_per_passo, significativa: bool, esponente_metro).
        """
        v = np.asarray(serie, dtype=np.float64).ravel()
        v = v[np.isfinite(v)]
        if v.size < 9:
            return 0.0, False, 0.0
        # la firma della deriva e' lo SPOSTAMENTO CUMULATIVO: mediana del primo
        # terzo vs mediana dell'ultimo terzo (robusto agli spike), confrontato
        # col rumore della serie. Gli incrementi singoli restano nel rumore —
        # e' la somma che tradisce la rana.
        terzo = v.size // 3
        med_inizio = float(np.median(v[:terzo]))
        med_fine = float(np.median(v[-terzo:]))
        spostamento = med_fine - med_inizio
        tasso = spostamento / max(1, v.size - terzo)
        mad_serie = float(np.median(np.abs(v - np.median(v)))) or 1e-12
        rumore_mediane = 1.4826 * mad_serie / np.sqrt(terzo)
        significativa = abs(spostamento) > 3.0 * rumore_mediane
        # METRO: risolve la scala dell'infinitesimo (leggibile anche a 1e-12)
        _v_new, fact = Metro().enc(abs(tasso))
        esponente = float(np.log10(fact)) - (-4.0) if fact > 0 else 0.0  # = -log10(|tasso|)
        return tasso, bool(significativa), esponente

    def referto_json(self, serie, nome="serie", riferimento=None):
        """Referto MACCHINA-LEGGIBILE (dict pronto per json.dumps).
        Pensato per IA e altri programmi: niente prosa, solo dati.
        I valori non finiti diventano stringhe ("NaN") per dare JSON valido."""
        pulito, K, anomalie = self.analizza(serie, riferimento)
        s = np.asarray(serie, dtype=np.float64).ravel()
        tasso, sig, esp = self.deriva(serie)
        def _num(v):
            return float(v) if np.isfinite(v) else "NaN"
        return {
            "nome": nome,
            "punti": int(s.size),
            "livello_ia": self.livello_ia,
            "anomalie": [{"indice": int(i), "grezzo": _num(s[i]),
                          "pulito": _num(pulito[i]), "K": _num(K[i])} for i in anomalie],
            "K_medio": _num(np.mean(K)),
            "deriva": {"tasso_per_passo": _num(tasso), "significativa": bool(sig),
                       "esponente_metro": _num(esp)},
        }

    def referto(self, serie, nome="serie", riferimento=None):
        """Analisi + referto leggibile."""
        pulito, K, anomalie = self.analizza(serie, riferimento)
        s = np.asarray(serie, dtype=np.float64).ravel()
        print(f"[ARMATURA] {nome}: {len(s)} punti | livello IA {self.livello_ia:.2f} | anomalie: {len(anomalie)}")
        for i in anomalie[:10]:
            print(f"   -> indice {i}: valore {s[i]!r} (pulito: {pulito[i]:.4f}, K={K[i]:.3f})")
        if len(anomalie) > 10:
            print(f"   ... e altre {len(anomalie) - 10}")
        tasso, sig, esp = self.deriva(serie)
        if sig:
            print(f"   ⚠ DERIVA LENTA rilevata (Metro): {tasso:+.2e} per passo (scala 10^-{esp:.1f})")
        return anomalie


def main():
    import re as _re
    argv = sys.argv[1:]
    come_json = "--json" in argv
    argv = [a for a in argv if a != "--json"]
    if not argv:
        print("USO: python armatura.py [--json] <file_con_numeri>  oppure  python armatura.py [--json] n1 n2 n3 ...")
        sys.exit(0)
    if len(argv) == 1 and os.path.exists(argv[0]):
        testo = open(argv[0], encoding="utf-8", errors="ignore").read()
        numeri = [float(x) if x.lower() != "nan" else float("nan")
                  for x in _re.findall(r"[-+]?(?:\d+\.?\d*(?:[eE][-+]?\d+)?|nan|NaN)", testo)]
        nome = os.path.basename(argv[0])
    else:
        numeri = [float(x) if x.lower() != "nan" else float("nan") for x in argv]
        nome = "serie da riga di comando"
    if come_json:
        import json as _json
        print(_json.dumps(Armatura().referto_json(numeri, nome), ensure_ascii=False))
    else:
        Armatura().referto(numeri, nome)


if __name__ == "__main__":
    main()
