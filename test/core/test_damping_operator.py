# -*- coding: utf-8 -*-
import numpy as np
import jax.numpy as jnp
from dense_armor.core.damping_operator import apply_damping_blend


def test_nessun_nan_in_uscita_anche_con_nan_in_ingresso():
    grezzo = jnp.array([1.0, float("nan"), 0.3, -2.0])
    riferimento = jnp.array([1.0, 0.4, float("nan"), -1.9])
    out = apply_damping_blend(grezzo, riferimento)
    assert not bool(jnp.any(jnp.isnan(out)))


def test_blend_range_limitato():
    """Il coefficiente K non deve mai far collassare l'output esattamente
    sull'uno o sull'altro segnale (mai 0% o 100%)."""
    grezzo = jnp.array([1.0, 50.0, 0.3, -2.0])
    riferimento = jnp.array([1.0, 0.4, 0.35, -1.9])
    out = np.array(apply_damping_blend(grezzo, riferimento))
    for g, r, o in zip(np.array(grezzo), np.array(riferimento), out):
        if g == r:
            continue
        assert min(g, r) - 1e-6 <= o <= max(g, r) + 1e-6
