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
