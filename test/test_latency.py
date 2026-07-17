"""
Test di regressione per il bug di ricompilazione JIT ad ogni chiamata.

filter_data_stream costruiva un jax.jit(lambda...) NUOVO ad ogni chiamata
(oggetto Python diverso -> cache-miss garantito in JAX -> XLA ricompila
tutto da zero ogni singola volta, anche a "regime": misurato ~80-108ms
fissi per chiamata su un array di 8 elementi, MAI in calo).

Fix: kernel 1D costruito una sola volta in __init__
(_compiled_single_stream_filter), con thr/dmp/alp/noise_scalar passati
come argomenti jit invece che chiusi su self.* -- stesso pattern gia'
corretto di _compiled_batch_filter.

Questo test non fissa una soglia assoluta in millisecondi (dipende
dalla macchina/CI), ma verifica il RAPPORTO tra la prima chiamata
(compilazione) e le successive (cache calda): se qualcuno reintroduce
un jax.jit ricreato ad ogni chiamata, le chiamate successive alla prima
torneranno a costare quanto la prima, e il rapporto crollera' vicino a 1.
"""
import time
from unittest.mock import MagicMock, patch

import jax
jax.config.update("jax_enable_x64", True)

import numpy as np
import pytest

from dense_armor.core.engine import AdaptiveSignalStabilizer
from dense_armor.utility.orca import Orca


def _ram_abbondante() -> MagicMock:
    """psutil.virtual_memory() finto con RAM abbondante libera (90%).

    Orca._gc_se_ram_bassa() chiama jax.clear_caches() sotto una soglia di
    RAM libera -- svuota la cache di compilazione JIT di TUTTO il processo,
    non solo di questo test. Se la macchina/CI reale scende sotto quella
    soglia mentre gira questo test (dipende dal carico del momento, non dal
    codice), il test diventa flaky: sembra un fallimento della cache JIT
    quando e' solo pressione di RAM del momento. Fissare qui un valore
    abbondante rende il test deterministico, indipendente dallo stato
    reale della macchina.
    """
    finto = MagicMock()
    finto.total = 16_000_000_000
    finto.available = int(finto.total * 0.90)
    return finto


def test_filter_data_stream_usa_la_cache_jit_non_ricompila_ogni_volta():
    stab = AdaptiveSignalStabilizer()
    serie = np.array([1.2, 1.3, 1.28, 1.31, 9999.0, 1.29, 1.30, 1.27])

    t0 = time.perf_counter()
    stab.filter_data_stream(serie)
    t_prima_chiamata = time.perf_counter() - t0

    tempi_successivi = []
    for _ in range(5):
        t0 = time.perf_counter()
        stab.filter_data_stream(serie)
        tempi_successivi.append(time.perf_counter() - t0)

    t_medio_a_regime = sum(tempi_successivi) / len(tempi_successivi)

    # A regime deve costare una FRAZIONE piccola della prima chiamata
    # (compilazione). Soglia larga (5x piu' rapido) per non essere
    # fragile su CI/hardware lenti, ma abbastanza stretta da beccare
    # una regressione totale (ricompilazione ad ogni chiamata, dove il
    # rapporto tornerebbe vicino a 1x).
    assert t_medio_a_regime < (t_prima_chiamata / 5.0), (
        f"la cache JIT non sembra funzionare: prima chiamata "
        f"{t_prima_chiamata*1000:.1f}ms, media successive "
        f"{t_medio_a_regime*1000:.1f}ms -- possibile regressione "
        f"del bug di ricompilazione ad ogni chiamata"
    )


@patch("dense_armor.utility.orca.psutil.virtual_memory", side_effect=_ram_abbondante)
def test_orca_protect_and_forward_usa_la_cache_jit_non_ricompila_ogni_volta(_mock_ram):
    def modello_finto(x):
        return x * 2 + 1

    serie = np.array([1.2, 1.3, 1.28, 1.31, 9999.0, 1.29, 1.30, 1.27])
    orca = Orca()

    t0 = time.perf_counter()
    orca.protect_and_forward(modello_finto, serie, x_reference=None)
    t_prima_chiamata = time.perf_counter() - t0

    tempi_successivi = []
    for _ in range(3):
        t0 = time.perf_counter()
        orca.protect_and_forward(modello_finto, serie, x_reference=None)
        tempi_successivi.append(time.perf_counter() - t0)

    t_medio_a_regime = sum(tempi_successivi) / len(tempi_successivi)

    assert t_medio_a_regime < (t_prima_chiamata / 5.0), (
        f"la cache JIT non sembra funzionare in Orca: prima chiamata "
        f"{t_prima_chiamata*1000:.1f}ms, media successive "
        f"{t_medio_a_regime*1000:.1f}ms"
    )