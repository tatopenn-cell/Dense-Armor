"""
Test di regressione per due bug trovati in Orca/AdaptiveSignalStabilizer
su input 1D (il caso d'uso principale documentato nel README: una singola
serie da sensore/pipeline).

Bug 1 (crash): protect_and_forward interpretava un array (N,) come un
batch di N istanze scalari indipendenti (B=N, slice_shape=()). Questo
faceva collassare calibrate_macro_context su shape (1,1), dove
np.diff(...) produce un array vuoto e np.max/np.mean esplodono.

Bug 2 (silenzioso, piu' grave): anche dove non crashava, trattare ogni
elemento come istanza isolata azzerava il contesto temporale su cui si
basa il rilevamento outlier in modalita' cieca (senza x_reference) --
un outlier enorme passava intonso, senza nessun errore o avviso.

Fix: promuovere un input 1D a (1, N) prima di calcolare B/slice_shape,
processarlo come UNA istanza con N campioni correlati, e riportare
output/margini alla forma 1D originale prima del return.
"""
import jax
jax.config.update("jax_enable_x64", True)

import numpy as np
import pytest

from dense_armor.utility.orca import Orca


def _modello_lineare(x):
    return x * 2 + 1


@pytest.mark.parametrize("n", [1, 2, 3, 4, 5, 8, 10, 16, 32, 50, 100])
def test_orca_1d_non_crasha_con_riferimento(n):
    """Bug 1: prima crashava per QUALSIASI n con ValueError su np.max di array vuoto."""
    dato_corrotto = np.ones(n)
    dato_corrotto[min(2, n - 1)] = 9999.0
    dato_pulito_rif = np.ones(n)

    orca = Orca()
    out = orca.protect_and_forward(
        _modello_lineare, dato_corrotto, x_reference=dato_pulito_rif
    )

    assert out.shape == (n,), "l'output deve restare 1D come l'input, non (1, n)"
    assert np.all(np.isfinite(out))


def test_orca_1d_singolo_elemento_non_crasha():
    """Caso limite del bug 1: array di un solo campione."""
    orca = Orca()
    out = orca.protect_and_forward(
        _modello_lineare, np.array([5.0]), x_reference=np.array([5.0])
    )
    assert out.shape == (1,)


def test_orca_1d_modalita_cieca_rileva_outlier():
    """
    Bug 2 (il piu' importante): senza x_reference, un array 1D deve
    ancora rilevare un outlier isolato usando il contesto dei campioni
    vicini -- non deve lasciarlo passare intonso solo perche' e' 1D.
    """
    serie = np.array([1.2, 1.3, 1.28, 1.31, 9999.0, 1.29, 1.30, 1.27])
    idx_outlier = 4

    orca = Orca()
    out = orca.protect_and_forward(_modello_lineare, serie, x_reference=None)

    assert out.shape == serie.shape
    # l'output al netto del modello (2x+1) non deve piu' contenere
    # il valore grezzo dell'outlier (19999.0)
    assert out[idx_outlier] < 100.0, (
        "l'outlier e' passato intonso: il contesto temporale non e' stato "
        "usato per rilevarlo (regressione del bug silenzioso)"
    )
    # il margine d'errore in ingresso deve segnalare chiaramente
    # quel punto come fortemente corretto rispetto agli altri
    assert orca.margine_ingresso[idx_outlier] > 1000.0
    assert np.all(np.array(orca.margine_ingresso)[
        [i for i in range(len(serie)) if i != idx_outlier]
    ] < 10.0)


def test_orca_2d_esplicito_resta_invariato():
    """Un batch esplicito (B, N) non deve cambiare comportamento con la patch."""
    dato_corrotto = np.array([[1.2, 1.3, 9999.0, 1.31]])
    orca = Orca()
    out = orca.protect_and_forward(_modello_lineare, dato_corrotto, x_reference=None)
    assert out.shape == (1, 4)