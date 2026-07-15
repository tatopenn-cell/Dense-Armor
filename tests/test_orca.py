# -*- coding: utf-8 -*-
"""Test di regressione per i bug trovati e corretti in questa sessione:
1. Valori negativi azzerati dal clamp nella compressione logaritmica.
2. Scudo di uscita confrontato con lo spazio sbagliato (input invece di output).
3. Modalita' cieca (senza x_reference) che collassava tutto a zero.
"""
import numpy as np
import jax
import jax.numpy as jnp
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
