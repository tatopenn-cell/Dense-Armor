"""
Confronto sperimentale: damping_essenziale, abcollatz_essenziale e la loro
combinazione (test/algoritmi_essenziali.py, versioni "all'osso" senza JAX)
contro Dense-Armor (Orca, pipeline reale) e Kalman (filterpy), sugli stessi
7 scenari di testKalman.py.

Domanda a cui risponde: i due meccanismi essenziali funzionano meglio da
soli o combinati? Non deciso a tavolino -- si guardano i numeri.

Uso:
    python testAlgoritmiEssenziali.py
"""
import logging
import time

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)

from dense_armor.utility.orca import Orca

from testKalman import genera_scenari, kalman_1d, modello_a_valle, X_CLEAN, N
from algoritmi_essenziali import damping_essenziale, abcollatz_essenziale, combinato_essenziale


def rmse(out: np.ndarray, target: np.ndarray, mask: np.ndarray) -> float:
    return float(np.sqrt(np.mean((out[mask] - target[mask]) ** 2)))


def max_leak(out: np.ndarray, target: np.ndarray, mask: np.ndarray) -> float:
    return float(np.max(np.abs(out[mask] - target[mask])))


def main() -> None:
    logging.getLogger("dense_armor").setLevel(logging.CRITICAL)
    scenari = genera_scenari(X_CLEAN, N)

    orca = Orca()
    _ = orca.protect_and_forward(modello_a_valle, np.full(N, X_CLEAN), x_reference=None)  # warmup

    algoritmi = {
        "Damping essenziale": lambda s: damping_essenziale(s),
        "ABCollatz essenziale": lambda s: abcollatz_essenziale(s),
        "Combinato essenziale": lambda s: combinato_essenziale(s),
        "Kalman (filterpy)": lambda s: kalman_1d(s, x0=X_CLEAN, q=0.01, r=1.0),
        "Dense-Armor (Orca)": lambda s: np.array(orca.protect_and_forward(modello_a_valle, s, x_reference=None)),
    }

    print("=" * 100)
    print("  CONFRONTO: meccanismi essenziali (senza JAX) vs Dense-Armor vs Kalman")
    print("=" * 100)

    punteggi = {nome: 0 for nome in algoritmi}

    for key, (descr, dati, target) in scenari.items():
        mask = ~np.isnan(dati)
        print(f"\n[{key}] {descr}")

        risultati_scenario = {}
        for nome, fn in algoritmi.items():
            t0 = time.perf_counter()
            out = fn(dati)
            t_ms = (time.perf_counter() - t0) * 1000
            r = rmse(out, target, mask)
            l = max_leak(out, target, mask)
            risultati_scenario[nome] = r
            print(f"    {nome:22s}  RMSE={r:9.4f}   max_leak={l:9.4f}   tempo={t_ms:7.3f}ms")

        vincitore = min(risultati_scenario, key=risultati_scenario.get)
        punteggi[vincitore] += 1
        print(f"    -> migliore: {vincitore}")

    print("\n" + "=" * 100)
    print("  RIEPILOGO VITTORIE (RMSE piu' basso per scenario)")
    for nome, v in sorted(punteggi.items(), key=lambda kv: -kv[1]):
        print(f"    {nome:22s}  {v}/{len(scenari)}")
    print("=" * 100)


if __name__ == "__main__":
    main()
