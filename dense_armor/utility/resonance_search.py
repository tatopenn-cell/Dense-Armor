# -*- coding: utf-8 -*-
import jax
import jax.numpy as jnp
import numpy as np
from ..core.damping_operator import apply_damping_blend

@jax.jit
def _resonance_scores(database_matrix: jnp.ndarray, query_vector: jnp.ndarray, kappa: float, delta_eff: float, stress_segnale: float) -> jnp.ndarray:
    """Punteggio di risonanza per riga di database_matrix rispetto a query_vector."""
    eps = 1e-8
    norms_M = jnp.linalg.norm(database_matrix, axis=1, keepdims=True)
    M_normed = database_matrix / jnp.maximum(norms_M, eps)
    cos_base = M_normed @ query_vector
    
    q_exp = jnp.expand_dims(query_vector, axis=0)
    M_phi = apply_damping_blend(database_matrix, q_exp)
    
    M_phi_stable = M_phi * (1.0 + jnp.abs(delta_eff))
    norms_phi = jnp.linalg.norm(M_phi_stable, axis=1, keepdims=True)
    M_phi_normed = M_phi_stable / jnp.maximum(norms_phi, eps)
    cos_phi = M_phi_normed @ query_vector
    
    w_phi = jnp.clip(0.60 + (kappa * 0.25), 0.10, 0.85)
    w_base = 1.0 - w_phi
    score_coerente = w_base * cos_base + w_phi * cos_phi
    
    n_discreto = jnp.round(jnp.abs(stress_segnale) * 100.0) + 3.0
    cond_pari = (n_discreto % 2.0 == 0.0)
    n_next = jnp.where(cond_pari, n_discreto / 2.0, 3.0 * n_discreto + 1.0)
    epsilon_ABC = jnp.abs(jnp.log(jnp.maximum(n_next, eps)) - 1.0)
    K_ABC = 0.10 + jax.nn.sigmoid(epsilon_ABC) * (0.85 - 0.10)
    
    return score_coerente * (1.0 + (K_ABC * jnp.abs(stress_segnale)))

def apply_fast_resonance(
    matrix_np: np.ndarray, 
    query_np: np.ndarray, 
    kappa: float = 0.86210, 
    delta_eff: float = 0.043410,
    stress_segnale: float = 9.42194e-04
) -> np.ndarray:
    """Punteggio di risonanza tra ogni riga di matrix_np e query_np, gestendo input vuoti/degeneri."""
    if matrix_np is None or query_np is None:
        return np.array([], dtype=np.float32)
    if matrix_np.size == 0 or query_np.size == 0:
        return np.zeros(len(matrix_np), dtype=np.float32)
        
    q = np.asarray(query_np, dtype=np.float32).squeeze()
    if q.ndim == 0 or q.size == 0:
        return np.zeros(len(matrix_np), dtype=np.float32)
    if q.ndim > 1:
        q = q.flatten()
        
    qn = np.linalg.norm(q)
    if qn < 1e-8:
        return np.zeros(len(matrix_np), dtype=np.float32)
    q = q / qn
    
    j_matrix = jnp.array(matrix_np, dtype=jnp.float32)
    j_query = jnp.array(q, dtype=jnp.float32)
    scores = _resonance_scores(j_matrix, j_query, float(kappa), float(delta_eff), float(stress_segnale))
    return np.array(scores, dtype=np.float32)

def smoke_test() -> bool:
    """Auto-test rapido: True se apply_fast_resonance produce un risultato sensato su dati sintetici."""
    try:
        N, D = 4, 8
        rng = np.random.default_rng(42)
        m = rng.standard_normal((N, D)).astype(np.float32)
        q = rng.standard_normal(D).astype(np.float32)
        s = apply_fast_resonance(m, q)
        assert s.shape == (N,)
        assert not np.all(np.isnan(s))
        assert not np.all(s == 0.0)
        return True
    except Exception:
        # broad by design: uno smoke test deve catturare QUALUNQUE
        # fallimento (import, shape, NaN, assert) e ridurlo a True/False --
        # non sta nascondendo un bug, e' la sua funzione.
        return False
