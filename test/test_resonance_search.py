# -*- coding: utf-8 -*-
import dense_armor.utility.resonance_search as resonance_search
from dense_armor.utility.resonance_search import smoke_test, apply_fast_resonance
import numpy as np


def test_smoke_test_passa():
    assert smoke_test() is True


def test_smoke_test_ritorna_false_se_apply_fast_resonance_fallisce(monkeypatch):
    def _boom(*args, **kwargs):
        raise RuntimeError("guasto simulato")

    monkeypatch.setattr(resonance_search, "apply_fast_resonance", _boom)
    assert smoke_test() is False


def test_apply_fast_resonance_shape():
    rng = np.random.default_rng(0)
    m = rng.standard_normal((5, 8)).astype(np.float32)
    q = rng.standard_normal(8).astype(np.float32)
    scores = apply_fast_resonance(m, q)
    assert scores.shape == (5,)
    assert not np.any(np.isnan(scores))


def test_apply_fast_resonance_matrice_o_query_none_ritorna_array_vuoto():
    rng = np.random.default_rng(0)
    m = rng.standard_normal((5, 8)).astype(np.float32)
    q = rng.standard_normal(8).astype(np.float32)
    assert apply_fast_resonance(None, q).shape == (0,)
    assert apply_fast_resonance(m, None).shape == (0,)


def test_apply_fast_resonance_matrice_o_query_vuote_ritorna_zeri():
    rng = np.random.default_rng(0)
    m = rng.standard_normal((5, 8)).astype(np.float32)
    q = rng.standard_normal(8).astype(np.float32)
    empty_m = np.empty((0, 8), dtype=np.float32)
    empty_q = np.empty((0,), dtype=np.float32)

    out_a = apply_fast_resonance(empty_m, q)
    assert out_a.shape == (0,)

    out_b = apply_fast_resonance(m, empty_q)
    np.testing.assert_array_equal(out_b, np.zeros(5, dtype=np.float32))


def test_apply_fast_resonance_query_scalare_ritorna_zeri():
    rng = np.random.default_rng(0)
    m = rng.standard_normal((5, 8)).astype(np.float32)
    out = apply_fast_resonance(m, np.array(3.0, dtype=np.float32))
    np.testing.assert_array_equal(out, np.zeros(5, dtype=np.float32))


def test_apply_fast_resonance_query_multidimensionale_viene_appiattita():
    rng = np.random.default_rng(0)
    m = rng.standard_normal((5, 8)).astype(np.float32)
    q_flat = rng.standard_normal(8).astype(np.float32)
    q_2d = q_flat.reshape(2, 4)

    out_flat = apply_fast_resonance(m, q_flat)
    out_2d = apply_fast_resonance(m, q_2d)

    np.testing.assert_allclose(out_flat, out_2d, atol=1e-5)


def test_apply_fast_resonance_query_a_norma_zero_ritorna_zeri():
    rng = np.random.default_rng(0)
    m = rng.standard_normal((5, 8)).astype(np.float32)
    out = apply_fast_resonance(m, np.zeros(8, dtype=np.float32))
    np.testing.assert_array_equal(out, np.zeros(5, dtype=np.float32))


def test_apply_fast_resonance_kappa_modula_davvero_il_punteggio_non_solo_cosine():
    """L'utilita' dichiarata (vedi README/docs) e' che il punteggio sia
    modulato da apply_damping_blend -- lo stesso operatore di Orca -- non
    una cosine similarity pura travestita. Se kappa (peso della componente
    smorzata) non cambiasse il punteggio, la modulazione sarebbe solo
    decorativa: qui si verifica che kappa=0 e kappa=1 diano punteggi
    misurabilmente diversi sulla stessa matrice/query."""
    rng = np.random.default_rng(3)
    db = rng.standard_normal((4, 16)).astype(np.float32)
    query = rng.standard_normal(16).astype(np.float32)

    basso = apply_fast_resonance(db, query, kappa=0.0)
    alto = apply_fast_resonance(db, query, kappa=1.0)

    assert not np.allclose(basso, alto, atol=1e-3), (
        "kappa non ha alcun effetto misurabile sul punteggio: la modulazione "
        "via apply_damping_blend sarebbe puramente decorativa"
    )


def test_apply_fast_resonance_ranking_e_indistinguibile_da_cosine_puro():
    """Il test precedente mostra che kappa/delta_eff/stress_segnale spostano
    i VALORI del punteggio -- ma per un motore di retrieval cio' che conta
    davvero e' l'ORDINE (ranking), non il valore assoluto. Benchmark reale
    su quantumrag (collection quantum_info, 1855 chunk, 12 query corte
    etichettate con il documento corretto atteso, Mean Reciprocal Rank come
    metrica) trovato 2026-09-01: cosine puro da' MRR=0.8125; apply_fast_
    resonance con le sue costanti reali (kappa=0.86210, delta_eff=0.04341,
    stress_segnale=9.42e-04) da' lo STESSO identico MRR=0.8125; 30 trial
    indipendenti con kappa/delta_eff/stress_segnale COMPLETAMENTE
    randomizzati (range ampi, incluso fuori scala) danno anch'essi
    MRR=0.8125 in ognuno dei 30 trial, deviazione standard 0.0000 --
    z-score del valore "reale" rispetto alla distribuzione randomizzata:
    0.00. Causa root, verificata numericamente: apply_fast_resonance e
    cosine puro correlano a 0.999996 sugli stessi dati (quasi-collineari)
    -- il blend/K_ABC perturba i valori assoluti ma quella perturbazione e'
    dominata cosi' fortemente dal termine cosine di base che non altera mai
    l'ordine relativo dei risultati, ne' con le costanti "vere" ne' con
    costanti arbitrarie.

    Stesso schema di confondimento gia' trovato in Dense-Evolution-Discovery
    Experiment 35 (zne_healing_sigma_provenance.py): una perturbazione
    reale nei numeri che pero' non dipende in modo significativo da COSA la
    guida, verificato con lo stesso disegno a controllo negativo (permutare/
    randomizzare l'input "significativo" e verificare che il risultato non
    cambi). Qui replicato su dati sintetici (non richiede quantumrag) per
    tenere il test portabile e veloce."""
    rng = np.random.default_rng(7)
    n_docs, dim = 40, 24
    db = rng.standard_normal((n_docs, dim)).astype(np.float32)

    # Query costruita come un documento reale + rumore, cosi' il "corretto"
    # top-1 e' noto per costruzione (indice 5), non arbitrario.
    target_idx = 5
    query = db[target_idx] + 0.15 * rng.standard_normal(dim).astype(np.float32)

    def top1(scores):
        return int(np.argmax(scores))

    cosine_scores = db @ query / (
        np.linalg.norm(db, axis=1) * np.linalg.norm(query) + 1e-8
    )
    assert top1(cosine_scores) == target_idx, "il benchmark sintetico stesso non e' valido: cosine non trova il target"

    real_scores = apply_fast_resonance(db, query)
    assert top1(real_scores) == target_idx

    mismatches = 0
    n_trials = 30
    for trial in range(n_trials):
        kappa = float(rng.uniform(0.01, 2.0))
        delta_eff = float(rng.uniform(-1.0, 1.0))
        stress = float(rng.uniform(1e-6, 1e-1))
        random_scores = apply_fast_resonance(db, query, kappa=kappa, delta_eff=delta_eff, stress_segnale=stress)
        if top1(random_scores) != target_idx:
            mismatches += 1

    assert mismatches == 0, (
        f"{mismatches}/{n_trials} trial con parametri randomizzati hanno cambiato il ranking rispetto "
        f"a cosine puro -- se questo fallisce in futuro, la modulazione avrebbe smesso di essere "
        f"un confondimento puro e andrebbe rivalutata (non solo re-skippata)"
    )
