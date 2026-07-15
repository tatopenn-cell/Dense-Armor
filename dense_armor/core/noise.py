# -*- coding: utf-8 -*-
"""
core/noise.py
================================
AIHardwareProfiler  — profila l'architettura host per soglie di carico ottimali.
StochasticAdversarialNoise — inietta perturbazioni avversariali su CPU e GPU.
"""

import platform
import psutil
import numpy as np

try:
    import jax
    import jax.numpy as jnp
    HAS_JAX = True
except ImportError:
    HAS_JAX = False


# =========================================================================
# AIHARDWAREPROFILER (FIXED CLASS DEFINITION)
# =========================================================================
class AIHardwareProfiler:
    """
    Profila l'architettura host (CPU/GPU/JAX) per impostare le soglie di
    carico ottimali.
    """

    def __init__(self):
        self.processor       = platform.processor()
        self.ram_total_gb    = psutil.virtual_memory().total / (1024 ** 3)
        self.has_jax         = HAS_JAX
        self.backend_device  = self._detect_active_backend()
        self.max_tensor_dim  = self._get_safe_tensor_limit()

    def _detect_active_backend(self) -> str:
        if not HAS_JAX:
            return "CPU (NumPy Standard)"
        try:
            backend = jax.default_backend()
            return f"{backend.upper()} (JAX Accelerato)"
        except Exception:
            return "CPU (JAX Fallback)"

    def _get_safe_tensor_limit(self) -> int:
        multiplier = 2 if any(x in self.backend_device for x in ("GPU", "TPU")) else 1
        if self.ram_total_gb >= 32:
            return 8192 * multiplier
        if self.ram_total_gb >= 12:
            return 4096 * multiplier
        return 2048 * multiplier

    def get_profile_summary(self) -> str:
        return (
            f"Processor: {self.processor} | "
            f"RAM: {self.ram_total_gb:.1f} GB | "
            f"Engine: {self.backend_device} | "
            f"SafeMaxDim: {self.max_tensor_dim}"
        )


# =========================================================================
# STOCHASTICADVERSARIALNOISE
# =========================================================================
class StochasticAdversarialNoise:
    """
    Riadattamento industriale di NoiseModel.
    Inietta rumore probabilistico o perturbazioni avversariali nei tensori IA,
    con dispatch automatico CPU (NumPy) / GPU (JAX).
    """

    SUPPORTED = {"bitflip", "dropout_noise", "gaussian_blur", "clean"}

    @staticmethod
    def inject_noise(
        data_vector: np.ndarray,
        noise_type:  str,
        intensity:   float,
        seed:        int = 42,
    ) -> np.ndarray:
        """Applica alterazioni probabilistiche preservando la norma del tensore."""
        noise_type = noise_type.lower().strip()

        if intensity <= 0.0 or noise_type == "clean":
            return data_vector

        # ── JAX path 
        if HAS_JAX and isinstance(data_vector, (jnp.ndarray, jax.Array)):
            key          = jax.random.PRNGKey(seed)
            key, subkey  = jax.random.split(key)
            trigger_mask = jax.random.uniform(subkey, shape=data_vector.shape) < intensity

            if noise_type == "bitflip":
                output = jnp.where(trigger_mask, -data_vector, data_vector)
            elif noise_type == "dropout_noise":
                output = jnp.where(trigger_mask, 0.0, data_vector)
            elif noise_type == "gaussian_blur":
                key, subkey2 = jax.random.split(key)
                noise  = jax.random.normal(subkey2, shape=data_vector.shape) * intensity
                output = data_vector + noise
            else:
                output = data_vector

            norm = jnp.linalg.norm(output)
            return jnp.where(norm > 0, output / (norm + 1e-15), output)

        # ── NumPy fallback 
        output       = np.array(data_vector, copy=True)
        rng          = np.random.default_rng(seed)
        trigger_mask = rng.random(size=output.shape) < intensity

        if noise_type == "bitflip":
            output = np.where(trigger_mask, -output, output)
        elif noise_type == "dropout_noise":
            output = np.where(trigger_mask, 0.0, output)
        elif noise_type == "gaussian_blur":
            noise  = rng.normal(0.0, intensity, size=output.shape)
            output = output + noise

        norm = np.linalg.norm(output)
        return output / (norm + 1e-15) if norm > 0 else output
