# -*- coding: utf-8 -*-
"""Test di regressione per la vettorizzazione dello scudo entrata di Orca
(_execute_4_phase_input_shield_batch): B righe indipendenti processate in
UNA chiamata (jax.vmap, calibrazione per-riga) invece di B dispatch JAX
separati (era il collo di bottiglia di latenza documentato nell'esperimento
APT29 + AI-shield, ~87% dispatch secondo lo stesso profiling che ha motivato
la vettorizzazione di Armatura in core/hybrid_engine.py).

Il rischio esplicito da coprire (non ipotetico): filter_batch_scenarios
esistente condivide UNA calibrazione (calibrate_macro_context) su TUTTO il
batch passato -- se _execute_4_phase_input_shield_batch chiamasse quella
invece della nuova variante indipendente, righe con scala/rumore diversi si
contaminerebbero a vicenda in modo silenzioso (nessun errore, solo numeri
diversi dal path a loop). Questi test usano righe con scala ESPLICITAMENTE
diversa (fattore 0.1x-20x) apposta per far emergere quella contaminazione se
mai tornasse."""
import numpy as np

from dense_armor.utility.orca import Orca


def _batch_scale_diversa(seed: int, B: int, F: int):
    rng = np.random.default_rng(seed)
    scale = rng.uniform(0.1, 20.0, size=(B, 1))
    x_clean = rng.uniform(1.0, 50.0, size=(B, F)) * scale
    x_corrupted = x_clean.copy()
    for b in range(B):
        x_corrupted[b, rng.integers(0, F)] = np.nan
        x_corrupted[b, rng.integers(0, F)] *= 50.0
    return x_clean, x_corrupted


def test_execute_4_phase_input_shield_batch_combacia_col_loop_riga_per_riga():
    """Il claim centrale: la versione batch deve produrre esattamente lo
    stesso output del vecchio path (chiamare _execute_4_phase_input_shield
    una volta per riga), su scale/rumore diversi per riga."""
    B, F = 10, 12
    x_clean, x_corrupted = _batch_scale_diversa(seed=11, B=B, F=F)

    orca_loop = Orca()
    out_loop = np.zeros((B, F))
    marg_loop = np.zeros((B, F))
    for b in range(B):
        out_loop[b], marg_loop[b] = orca_loop._execute_4_phase_input_shield(x_clean[b], x_corrupted[b])

    orca_batch = Orca()
    out_batch, marg_batch = orca_batch._execute_4_phase_input_shield_batch(x_clean, x_corrupted)

    np.testing.assert_allclose(out_batch, out_loop, rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(marg_batch, marg_loop, rtol=1e-9, atol=1e-12)


def test_execute_4_phase_input_shield_batch_singola_riga_combacia():
    """Caso limite B=1: il path batch deve ridursi esattamente al path
    originale a riga singola (nessuna regressione sul caso gia' testato
    altrove in test_orca.py/test_orca2.py)."""
    x_clean, x_corrupted = _batch_scale_diversa(seed=3, B=1, F=20)

    orca = Orca()
    out_ref, marg_ref = orca._execute_4_phase_input_shield(x_clean[0], x_corrupted[0])
    out_batch, marg_batch = orca._execute_4_phase_input_shield_batch(x_clean, x_corrupted)

    np.testing.assert_allclose(out_batch[0], out_ref, rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(marg_batch[0], marg_ref, rtol=1e-9, atol=1e-12)


def test_calibrazione_indipendente_non_si_contamina_tra_righe():
    """Verifica diretta del rischio descritto nel docstring del modulo:
    una riga con uno spike enorme non deve cambiare la calibrazione (e
    quindi l'output) di un'ALTRA riga pulita nello stesso batch."""
    F = 15
    rng = np.random.default_rng(9)
    riga_pulita = 1.0 + rng.normal(0, 0.02, F)
    riga_con_shock = 1.0 + rng.normal(0, 0.02, F)
    riga_con_shock_corrotta = riga_con_shock.copy()
    riga_con_shock_corrotta[7] = 9999.0

    orca = Orca()
    x_clean = np.stack([riga_pulita, riga_con_shock])
    x_corrupted = np.stack([riga_pulita, riga_con_shock_corrotta])
    out_batch, _ = orca._execute_4_phase_input_shield_batch(x_clean, x_corrupted)

    out_riga_pulita_da_sola, _ = orca._execute_4_phase_input_shield(riga_pulita, riga_pulita)

    np.testing.assert_allclose(out_batch[0], out_riga_pulita_da_sola, rtol=1e-9, atol=1e-12)


def test_protect_and_forward_batch_multi_riga_indipendente_end_to_end():
    """Wiring end-to-end (protect_and_forward, non il metodo interno): un
    batch B=25 con x_reference nota deve restare finito, forma corretta, e
    il margine d'errore deve essere popolato -- prima verifica mai fatta di
    protect_and_forward con B>1 su scenari davvero indipendenti (i test
    esistenti in test_orca.py/test_orca2.py usano tutti B=1)."""
    B, F = 25, 10
    x_clean, x_corrupted = _batch_scale_diversa(seed=42, B=B, F=F)

    orca = Orca()
    out = orca.protect_and_forward(
        None, x_corrupted, x_reference=x_clean,
        use_model_injection=False, use_output_shield=False,
    )

    assert out.shape == (B, F)
    assert np.all(np.isfinite(np.array(out)))
    assert orca.margine_ingresso.shape == (B, F)
    assert np.isfinite(orca.margine_ingresso_medio)
