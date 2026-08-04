# -*- coding: utf-8 -*-
"""Test di regressione per i bug trovati e corretti in questa sessione:
1. Valori negativi azzerati dal clamp nella compressione logaritmica.
2. Scudo di uscita confrontato con lo spazio sbagliato (input invece di output).
3. Modalita' cieca (senza x_reference) che collassava tutto a zero.
"""
import types
from unittest.mock import patch

import numpy as np
import pytest
import jax
import jax.numpy as jnp

import dense_armor.core.memory as memory_module
import dense_armor.utility.orca as orca_module
from dense_armor.core.memory import MemoryPressureError
from dense_armor.utility.orca import Orca


def test_segno_preservato_con_riferimento():
    """Regressione bug #1: valori negativi puliti non devono uscire a 0.0."""
    np.random.seed(0)
    x_clean = jnp.array(np.random.randn(1, 4, 4))
    x_corrupted = x_clean.at[0, 0, 0].set(float("nan")).at[0, 1, 1].set(1e6)

    orca = Orca()
    out = orca.protect_and_forward(None, x_corrupted, x_reference=x_clean,
                                    use_model_injection=False, use_output_shield=False)

    neg_idx = np.where(np.array(x_clean).ravel() < 0)[0]
    out_at_neg = np.array(out).ravel()[neg_idx]
    assert not np.all(out_at_neg == 0.0), "i valori negativi non devono essere azzerati"


def test_modalita_cieca_non_collassa_a_zero():
    """Regressione bug #3: senza x_reference, l'output non deve essere tutto zero."""
    np.random.seed(1)
    x_corrupted = jnp.array(np.random.randn(1, 50) * 2.0)
    x_corrupted = x_corrupted.at[0, 10].set(float("nan"))
    x_corrupted = x_corrupted.at[0, 20].set(5e5)

    orca = Orca()
    out = orca.protect_and_forward(None, x_corrupted,
                                    use_model_injection=False, use_output_shield=False)

    assert not bool(jnp.all(out == 0.0))
    assert not bool(jnp.any(jnp.isnan(out)))


def test_scudo_uscita_spazio_corretto_con_modello_trasformativo():
    """Regressione bug #2: con un modello non-identita' (tanh), lo scudo con
    riferimento deve battere nettamente il non fare nulla, non peggiorarlo."""
    key = jax.random.PRNGKey(42)
    W1 = jax.random.normal(key, (32, 32)) * 0.3

    def ai_model(x):
        flat = x.reshape(x.shape[0], -1)
        return jnp.tanh(flat @ W1).reshape(x.shape[0], 4, 4, 2)

    x_clean = jax.random.normal(jax.random.PRNGKey(7), (1, 4, 4, 2)) * 2.0
    mask_nan = jax.random.bernoulli(jax.random.PRNGKey(1), p=0.15, shape=x_clean.shape)
    x_corrupted = jnp.where(mask_nan, jnp.nan, x_clean)
    x_corrupted = x_corrupted.at[0, 1, 1, 0].set(5e5)

    out_none = jnp.nan_to_num(ai_model(x_corrupted), nan=0.0)
    out_truth = ai_model(x_clean)

    orca = Orca()
    out_protected = orca.protect_and_forward(ai_model, x_corrupted, x_reference=x_clean)

    mse_none = float(jnp.mean((out_none - out_truth) ** 2))
    mse_protected = float(jnp.mean((out_protected - out_truth) ** 2))
    assert mse_protected < mse_none


def test_margine_errore_popolato():
    x_clean = jnp.ones((1, 10))
    x_corrupted = x_clean.at[0, 3].set(float("nan"))

    orca = Orca()
    orca.protect_and_forward(None, x_corrupted, x_reference=x_clean,
                              use_model_injection=False, use_output_shield=False)

    assert orca.margine_ingresso is not None
    assert np.isfinite(orca.margine_ingresso_medio)
    assert np.isfinite(orca.margine_ingresso_max)


def _fake_virtual_memory(total, available):
    return types.SimpleNamespace(total=total, available=available)


def test_gc_se_ram_bassa_delega_a_memory_guard_e_solleva_sotto_soglia(monkeypatch):
    """Integrazione UniversalMemoryGuard: sotto la soglia dura di RAM libera,
    _gc_se_ram_bassa deve sollevare MemoryPressureError invece di continuare
    in silenzio come faceva il vecchio check psutil ad-hoc."""
    monkeypatch.setattr(
        memory_module.psutil, "virtual_memory",
        lambda: _fake_virtual_memory(total=100, available=1),
    )
    orca = Orca(min_free_ram_percentage=0.15)
    with pytest.raises(MemoryPressureError):
        orca._gc_se_ram_bassa()


def test_gc_se_ram_bassa_non_solleva_con_ram_abbondante(monkeypatch):
    monkeypatch.setattr(
        memory_module.psutil, "virtual_memory",
        lambda: _fake_virtual_memory(total=100, available=90),
    )
    orca = Orca(min_free_ram_percentage=0.15)
    orca._gc_se_ram_bassa()  # non deve sollevare


def test_recall_reference_none_se_banca_vuota():
    """Prima banca mai popolata (modalita' cieca al primissimo utilizzo):
    nessun richiamo possibile, comportamento identico a prima dell'integrazione."""
    orca = Orca()
    row = np.random.default_rng(0).standard_normal(20)
    assert orca._recall_reference(row, slice_shape=(20,)) is None


def test_remember_e_recall_reference_richiama_il_piu_simile():
    orca = Orca()
    t = np.linspace(0, 4 * np.pi, 50)
    rng = np.random.default_rng(3)

    clean_A = np.sin(t)
    clean_B = rng.standard_normal(50)  # riferimento scorrelato in banca

    orca._remember_reference(np.stack([clean_A, clean_B]), slice_shape=(50,))

    query_corrotta = clean_A + rng.standard_normal(50) * 0.1  # corrotta ma correlata ad A
    recalled = orca._recall_reference(query_corrotta, slice_shape=(50,))

    assert recalled is not None
    np.testing.assert_allclose(recalled, clean_A)


def test_recall_reference_none_se_nessun_candidato_abbastanza_simile():
    orca = Orca(reference_recall_min_score=0.90)
    rng = np.random.default_rng(4)
    clean_scorrelato = rng.standard_normal(50)

    orca._remember_reference(clean_scorrelato.reshape(1, -1), slice_shape=(50,))

    query_non_correlata = rng.standard_normal(50)
    assert orca._recall_reference(query_non_correlata, slice_shape=(50,)) is None


def test_protect_and_forward_ricorda_e_richiama_riferimento_end_to_end():
    """Wiring end-to-end: una chiamata con x_reference nota popola la banca;
    una chiamata successiva IN MODALITA' CIECA con un input corrotto simile
    deve produrre un risultato quantomeno buono quanto la stima cieca pura
    (il richiamo del vero riferimento non deve mai peggiorare la ricostruzione).

    Segnale spostato lontano dallo zero (+3.0): un riferimento che attraversa
    lo zero era, all'epoca in cui questo test e' stato scritto, un bug
    preesistente non ancora corretto (vedi test_riferimento_che_attraversa_lo_
    zero_non_esplode_piu, aggiunto in una sessione successiva per quel caso
    specifico). Offset mantenuto qui: questo test vuole verificare la memoria
    di riferimenti, non la robustezza alla compressione log10 vicino allo
    zero, gia' coperta altrove.
    """
    t = np.linspace(0, 4 * np.pi, 60)
    rng = np.random.default_rng(5)
    clean = (np.sin(t) + 3.0).reshape(1, -1)

    orca_con_memoria = Orca()
    corrupted_1 = clean + rng.standard_normal(clean.shape) * 0.05
    orca_con_memoria.protect_and_forward(
        None, corrupted_1, x_reference=clean,
        use_model_injection=False, use_output_shield=False,
    )

    corrupted_2 = clean + rng.standard_normal(clean.shape) * 0.05
    out_con_memoria = orca_con_memoria.protect_and_forward(
        None, corrupted_2, x_reference=None,
        use_model_injection=False, use_output_shield=False,
    )

    orca_senza_memoria = Orca()
    out_senza_memoria = orca_senza_memoria.protect_and_forward(
        None, corrupted_2, x_reference=None,
        use_model_injection=False, use_output_shield=False,
    )

    mse_con_memoria = float(np.mean((np.array(out_con_memoria) - clean) ** 2))
    mse_senza_memoria = float(np.mean((np.array(out_senza_memoria) - clean) ** 2))
    assert mse_con_memoria <= mse_senza_memoria


def _fake_profiler(max_tensor_dim):
    fake = types.SimpleNamespace(
        max_tensor_dim=max_tensor_dim,
        get_profile_summary=lambda: f"fake profiler max_tensor_dim={max_tensor_dim}",
    )
    return fake


def test_chunk_threshold_auto_su_host_baseline_ripete_il_vecchio_default(monkeypatch):
    """Host 'base' (RAM<12GB, nessuna GPU/TPU): max_tensor_dim=2048, la stessa
    baseline della classe -- chunk_threshold auto-derivato deve ricadere
    esattamente sul vecchio valore fisso (nessuna regressione silenziosa sul
    caso comune)."""
    monkeypatch.setattr(orca_module, "AIHardwareProfiler", lambda: _fake_profiler(2048))
    orca = Orca()
    assert orca.chunk_threshold == Orca._CHUNK_THRESHOLD_BASELINE


def test_chunk_threshold_auto_scala_con_hardware_migliore(monkeypatch):
    """Host con piu' RAM/GPU (max_tensor_dim=8192, 4x la baseline 2048):
    chunk_threshold deve scalare proporzionalmente (4x)."""
    monkeypatch.setattr(orca_module, "AIHardwareProfiler", lambda: _fake_profiler(8192))
    orca = Orca()
    assert orca.chunk_threshold == Orca._CHUNK_THRESHOLD_BASELINE * 4


def test_chunk_threshold_esplicito_non_viene_sovrascritto(monkeypatch):
    """Se il chiamante passa un chunk_threshold esplicito, l'auto-tuning non
    deve nemmeno istanziare AIHardwareProfiler."""
    def _boom():
        raise AssertionError("AIHardwareProfiler non doveva essere istanziato")
    monkeypatch.setattr(orca_module, "AIHardwareProfiler", _boom)
    orca = Orca(chunk_threshold=42)
    assert orca.chunk_threshold == 42


def test_riferimento_che_attraversa_lo_zero_non_esplode_piu():
    """Regressione: un riferimento pulito che attraversa lo zero (es. un seno
    centrato in 0) faceva esplodere numericamente la compressione log10 dello
    scudo entrata -- un singolo punto quasi-zero (anche solo un residuo di
    floating point, mai esattamente 0.0) produceva un fact_shared enorme,
    applicato anche al corrotto (che NON e' vicino a zero), avvelenando la
    calibrazione causale di AdaptiveSignalStabilizer per l'intero batch.
    Prima del fix: MSE ~9 (peggio della modalita' cieca, MSE ~0.1, sullo
    stesso identico segnale). Dopo il fix (floor su exp10_cl + clip su
    co_chunk): deve restare in un ordine di grandezza compatibile col rumore
    iniettato (std=0.05 -> varianza attesa ~0.0025), non esplodere."""
    t = np.linspace(0, 4 * np.pi, 60)
    rng = np.random.default_rng(5)
    clean = np.sin(t).reshape(1, -1)  # attraversa lo zero piu' volte
    corrupted = clean + rng.standard_normal(clean.shape) * 0.05

    orca = Orca()
    out = np.array(orca.protect_and_forward(
        None, corrupted, x_reference=clean,
        use_model_injection=False, use_output_shield=False,
    ))
    mse = float(np.mean((out - clean) ** 2))
    assert mse < 0.5, f"MSE {mse:.4f} suggerisce che l'esplosione numerica e' tornata"
