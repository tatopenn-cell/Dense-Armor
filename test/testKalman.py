"""
Dense-Armor (Orca, pipeline reale) vs Kalman Filter -- benchmark professionale.

Chiama DAVVERO Orca.protect_and_forward() del pacchetto pubblicato su PyPI
(dense-armor>=1.0.9), non una reimplementazione approssimata, e lo confronta
con un Kalman filter vero (filterpy), calibrato onestamente sullo stesso
processo generatore -- stesso spirito del confronto gia' documentato nel
README (~0.12 MSE Kalman calibrato vs ~0.23 MSE Dense-Armor su random walk),
esteso a scenari pensati per trovare dove ciascuno dei due cede, non solo
dove Dense-Armor vince.

Uso:
    python dense_armor_vs_kalman.py             # report + salva grafico/CSV
    python dense_armor_vs_kalman.py --verbose    # mostra anche i log interni di Orca
"""
import argparse
import csv
import logging
import time
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)

from dense_armor.utility.orca import Orca
from filterpy.kalman import KalmanFilter

np.random.seed(42)
X_CLEAN = 120.0
N = 100
OUT_DIR = Path(__file__).parent / "risultati"


def modello_a_valle(x: np.ndarray) -> np.ndarray:
    """Modello IA fittizio a valle dello scudo (identita': isola l'effetto del filtro)."""
    return x


def genera_scenari(x_clean: float, n: int) -> Dict[str, Tuple[str, np.ndarray, np.ndarray]]:
    """Costruisce scenari di difficolta' crescente, dai casi favorevoli a
    Dense-Armor (outlier isolati) fino a quelli strutturalmente favorevoli
    a un Kalman ben calibrato (rumore gaussiano diffuso, dropout esteso)."""
    scenari = {}

    s = np.full(n, x_clean)
    s[30:33] = [x_clean + 500.0, x_clean - 500.0, x_clean + 500.0]
    scenari["A"] = ("Impulsi alternati asimmetrici (+/-500)", s, np.full(n, x_clean))

    s = np.full(n, x_clean)
    s[50:53] = x_clean + 0.049
    scenari["B"] = ("Stealth sub-soglia (0.049)", s, np.full(n, x_clean))

    s = np.full(n, x_clean)
    s[70:73] = 0.0
    scenari["C"] = ("Collasso totale sensore (zero injection)", s, np.full(n, x_clean))

    t = np.arange(n)
    s = x_clean + 50.0 * np.sin(2 * np.pi * 3 * t / n)
    scenari["D"] = ("Iniezione sinusoidale strutturata", s, np.full(n, x_clean))

    # Rumore a code pesanti (non-gaussiano): sfida l'assunzione gaussiana di Kalman
    s = x_clean + np.random.standard_cauchy(n) * 3.0
    scenari["E"] = ("Rumore a code pesanti (Cauchy, non-gaussiano)", s, np.full(n, x_clean))

    # Rottura strutturale: salto di livello PERMANENTE, non un impulso.
    # Caso ambiguo per costruzione: e' un'anomalia da respingere o un nuovo
    # regime da seguire? Nessun filtro robusto lo risolve gratis.
    s = np.full(n, x_clean)
    s[60:] = x_clean + 80.0
    target_f = np.concatenate([np.full(60, x_clean), np.full(n - 60, x_clean + 80.0)])
    scenari["F"] = ("Rottura strutturale (salto di livello permanente)", s, target_f)

    # Buco esteso di dati: 8 NaN consecutivi su un fondo con rumore leggero
    # (non piatto: altrimenti il test e' banale). Terreno di casa per Kalman,
    # che gestisce i dropout nativamente col solo passo di predizione.
    rumore_fondo = np.random.normal(0, 0.3, n)
    s = x_clean + rumore_fondo
    s[40:48] = np.nan
    scenari["G"] = ("Buco esteso di dati (8 NaN consecutivi, fondo rumoroso)", s, np.full(n, x_clean))

    return scenari


def kalman_1d(misure: np.ndarray, x0: float, q: float, r: float) -> np.ndarray:
    """Kalman filter 1D a posizione costante, calibrato onestamente (q, r noti a priori)."""
    kf = KalmanFilter(dim_x=1, dim_z=1)
    kf.x = np.array([[x0]])
    kf.F = np.array([[1.0]])
    kf.H = np.array([[1.0]])
    kf.P *= 10.0
    kf.R = np.array([[r]])
    kf.Q = np.array([[q]])

    out = np.zeros_like(misure, dtype=np.float64)
    for i, z in enumerate(misure):
        kf.predict()
        if not np.isnan(z):
            kf.update(np.array([[z]]))
        out[i] = kf.x[0, 0]
    return out


def esegui_benchmark(verbose: bool = False) -> list:
    if not verbose:
        # Silenzia i log interni di Orca (livello di default: INFO su ogni
        # fase dello scudo) -- meccanismo standard di logging, non un
        # print() da sopprimere a mano.
        logging.getLogger("dense_armor").setLevel(logging.CRITICAL)

    scenari = genera_scenari(X_CLEAN, N)
    righe_report = []

    print("=" * 92)
    print("  DENSE-ARMOR (Orca, pipeline REALE) vs KALMAN FILTER (filterpy) -- benchmark")
    print("=" * 92)

    # Istanza Orca unica, pre-scaldata una sola volta: il tempo misurato per
    # scenario e' il costo reale "a regime", non la compilazione JIT (che
    # altrimenti domina la misura e falsa il confronto).
    orca = Orca()
    _ = orca.protect_and_forward(modello_a_valle, np.full(N, X_CLEAN), x_reference=None)

    for key, (descr, dati, target) in scenari.items():
        t0 = time.perf_counter()
        out_armor = np.array(orca.protect_and_forward(modello_a_valle, dati, x_reference=None))
        t_armor = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        out_kalman = kalman_1d(dati, x0=X_CLEAN, q=0.01, r=1.0)
        t_kalman = (time.perf_counter() - t0) * 1000

        mask = ~np.isnan(dati)
        rmse_armor = float(np.sqrt(np.mean((out_armor[mask] - target[mask]) ** 2)))
        rmse_kalman = float(np.sqrt(np.mean((out_kalman[mask] - target[mask]) ** 2)))
        max_leak_armor = float(np.max(np.abs(out_armor[mask] - target[mask])))
        max_leak_kalman = float(np.max(np.abs(out_kalman[mask] - target[mask])))

        if rmse_armor < 1e-9 and rmse_kalman < 1e-9:
            vincitore, margine = "Pareggio", 0.0
        else:
            vincitore = "Dense-Armor" if rmse_armor < rmse_kalman else "Kalman"
            margine = abs(rmse_armor - rmse_kalman) / max(rmse_armor, rmse_kalman, 1e-9) * 100

        print(f"\n[{key}] {descr}")
        print(f"    RMSE      Dense-Armor: {rmse_armor:8.4f}   |   Kalman: {rmse_kalman:8.4f}")
        print(f"    Max leak  Dense-Armor: {max_leak_armor:8.4f}   |   Kalman: {max_leak_kalman:8.4f}")
        print(f"    Tempo(ms) Dense-Armor: {t_armor:8.2f}   |   Kalman: {t_kalman:8.3f}")
        print(f"    -> vince {vincitore} (margine {margine:.1f}%)")

        righe_report.append({
            "scenario": key, "descrizione": descr,
            "rmse_armor": rmse_armor, "rmse_kalman": rmse_kalman,
            "max_leak_armor": max_leak_armor, "max_leak_kalman": max_leak_kalman,
            "t_armor_ms": t_armor, "t_kalman_ms": t_kalman,
            "vincitore": vincitore,
            "dati": dati, "target": target,
            "out_armor": out_armor, "out_kalman": out_kalman,
        })

    vittorie_armor = sum(1 for r in righe_report if r["vincitore"] == "Dense-Armor")
    vittorie_kalman = sum(1 for r in righe_report if r["vincitore"] == "Kalman")
    pareggi = sum(1 for r in righe_report if r["vincitore"] == "Pareggio")
    print("\n" + "=" * 92)
    print(f"  RIEPILOGO: Dense-Armor migliore in {vittorie_armor}/{len(righe_report)} scenari, "
          f"Kalman migliore in {vittorie_kalman}/{len(righe_report)}"
          + (f", pareggio in {pareggi}" if pareggi else ""))
    print("=" * 92)

    return righe_report


def salva_risultati(righe_report: list) -> None:
    OUT_DIR.mkdir(exist_ok=True)

    csv_path = OUT_DIR / "risultati_benchmark.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        campi = ["scenario", "descrizione", "rmse_armor", "rmse_kalman",
                 "max_leak_armor", "max_leak_kalman", "t_armor_ms", "t_kalman_ms", "vincitore"]
        writer = csv.DictWriter(f, fieldnames=campi)
        writer.writeheader()
        for r in righe_report:
            writer.writerow({k: r[k] for k in campi})
    print(f"\nRisultati numerici salvati in: {csv_path}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        n_scen = len(righe_report)
        fig, axes = plt.subplots(n_scen, 1, figsize=(12, 3.2 * n_scen), dpi=110)
        for ax, r in zip(axes, righe_report):
            ax.plot(r["dati"], color="#e74c3c", alpha=0.35, linestyle="--",
                    linewidth=1.3, label="Input corrotto")
            ax.plot(r["target"], color="#2c3e50", linestyle=":", alpha=0.6, label="Target vero")
            ax.plot(r["out_armor"], color="#2ecc71", linewidth=1.8, label="Dense-Armor (reale)")
            ax.plot(r["out_kalman"], color="#3498db", linewidth=1.5, label="Kalman")
            ax.set_title(f"[{r['scenario']}] {r['descrizione']}  ->  vince {r['vincitore']}",
                         fontsize=10, fontweight="bold")
            ax.legend(loc="upper right", fontsize=7, framealpha=0.9)
            ax.tick_params(labelsize=8)

        plt.tight_layout()
        png_path = OUT_DIR / "confronto_scenari.png"
        plt.savefig(png_path)
        plt.close(fig)
        print(f"Grafico salvato in: {png_path}")
    except ImportError:
        print("matplotlib non disponibile: grafico saltato, solo CSV salvato.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verbose", action="store_true",
                         help="Mostra anche i log interni [ORCA] di ogni chiamata")
    args = parser.parse_args()

    risultati = esegui_benchmark(verbose=args.verbose)
    salva_risultati(risultati)