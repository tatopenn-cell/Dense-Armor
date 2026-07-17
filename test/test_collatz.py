# -*- coding: utf-8 -*-
import numpy as np
import jax
import jax.numpy as jnp
from dense_armor.utility.collatz import ABCollatz


def test_gating_range_limitato():
    shield = ABCollatz(epsilon_target=1.0)
    x_clean = jnp.array([[1.0, 2.0, 3.0, 4.0, 5.0]])
    x_corrupted = x_clean + jnp.array([[0.0, 15.0, -20.0, 100.0, -1000.0]])
    gt = np.array(shield.compute_damping_gating(x_corrupted, x_clean))
    assert np.all(gt >= 0.10 - 1e-6)
    assert np.all(gt <= 0.85 + 1e-6)


def test_nan_forza_gating_massimo():
    shield = ABCollatz(epsilon_target=1.0)
    x_clean = jnp.array([[1.0, 2.0, 3.0]])
    x_corrupted = jnp.array([[1.0, float("nan"), 3.0]])
    gt = np.array(shield.compute_damping_gating(x_corrupted, x_clean))
    assert gt[0, 1] == 0.85


def test_radicale_dipende_dal_valore_collatz():
    """Regressione: calculate_jax_rad va calcolato su b (l'intero generato
    da Collatz), non sul prodotto a*b*c -- un prodotto di tre float generici
    non e' quasi mai divisibile esattamente per un intero piccolo, quindi il
    radicale collassava sempre a 1.0 indipendentemente da b, rendendo l'intera
    traiettoria di Collatz inerte sul gating finale."""
    shield = ABCollatz(epsilon_target=1.0)
    a = jnp.array(1.234)
    c = jnp.array(56.789)
    d1 = shield.evaluate_abc_discrepancy(a, jnp.array(7.0), c)
    d2 = shield.evaluate_abc_discrepancy(a, jnp.array(999999.0), c)
    assert d1 != d2


def test_mappa_smooth_coincide_su_interi():
    """La mappa continua deve coincidere esattamente con quella discreta
    sugli interi (dove cos^2/sin^2 valgono 0 o 1)."""
    shield = ABCollatz(epsilon_target=1.0)
    n = jnp.array([4.0, 7.0, 10.0, 13.0])
    discrete = np.array(jax.vmap(shield.execute_collatz_step)(n))
    smooth = np.array(jax.vmap(shield.execute_collatz_step_smooth)(n))
    assert np.allclose(discrete, smooth, atol=1e-6)
