# -*- coding: utf-8 -*-
import numpy as np
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
