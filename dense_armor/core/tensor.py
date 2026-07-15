# -*- coding: utf-8 -*-
"""
core/tensor.py
====================
TensorVault — custodisce matrici di trasformazione statiche e parametriche.
Rileva automaticamente il backend (JAX/NumPy) e la precisione (float32/float64).
"""

import numpy as np

try:
    import jax
    import jax.numpy as jnp
    HAS_JAX = True
except ImportError:
    HAS_JAX = False


class TensorVault:
    """Vault ottimizzato e compatto per matrici statiche e parametriche."""

    def __init__(self):
        self.xp    = jnp if HAS_JAX else np
        # Lettura sicura di jax_enable_x64 senza usare il vecchio metodo .get()
        is_x64     = getattr(jax.config, "jax_enable_x64", False) if HAS_JAX else False
        self.dtype = self.xp.float64 if (is_x64 or not HAS_JAX) else self.xp.float32

    # ── Static transforms ─────────────────────────────────────────────────────

    def get_static_transform(self, name: str) -> np.ndarray:
        """Sintetizza e restituisce la matrice statica richiesta."""
        xp, dt = self.xp, self.dtype
        transforms = {
            'invert':        lambda: xp.array([[0., 1.], [1., 0.]], dtype=dt),
            'identity':      lambda: xp.eye(2, dtype=dt),
            'edge_detector': lambda: xp.array([-1., 2., -1.], dtype=dt),
            'blend':         lambda: xp.array([[0.5, 0.5], [0.5, 0.5]], dtype=dt),
        }
        key = name.lower()
        if key not in transforms:
            raise KeyError(f"Trasformazione statica '{name}' non disponibile. "
                           f"Disponibili: {list(transforms.keys())}")
        return transforms[key]()

    # ── Parametric transforms ─────────────────────────────────────────────────

    def get_parametric_transform(self, name: str, p: float) -> np.ndarray:
        """Sintetizza dinamicamente la matrice parametrizzata richiesta."""
        xp, dt = self.xp, self.dtype
        transforms = {
            'scale_project': lambda: xp.array(
                [[xp.cos(p), -xp.sin(p)], [xp.sin(p), xp.cos(p)]], dtype=dt),
            'amplify':       lambda: xp.array([[p, 0.], [0., p]], dtype=dt),
            'bias_shift':    lambda: xp.array([p, -p], dtype=dt),
        }
        key = name.lower()
        if key not in transforms:
            raise KeyError(f"Trasformazione parametrica '{name}' non disponibile. "
                           f"Disponibili: {list(transforms.keys())}")
        return transforms[key]()

    # ── Info ──────────────────────────────────────────────────────────────────

    def get_backend_info(self) -> str:
        precision = "64-bit" if self.dtype in (
            np.float64, getattr(jnp, "float64", None)
        ) else "32-bit"
        backend = "JAX / GPU Accelerato" if HAS_JAX else "NumPy / CPU Standard"
        return f"{backend} ({precision})"