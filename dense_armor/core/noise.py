"""Host hardware profiler and stochastic adversarial noise injection.

:class:`AIHardwareProfiler` reads the CPU, the total RAM and the active JAX
backend to size the safe tensor limit. :class:`StochasticAdversarialNoise`
injects probabilistic perturbations into a tensor, dispatching between the
NumPy path and the JAX path based on the input type.
"""

import platform
from typing import ClassVar

import numpy as np
import psutil

try:
    import jax
    import jax.numpy as jnp

    HAS_JAX = True
except ImportError:
    HAS_JAX = False


class AIHardwareProfiler:
    """Profile the host (CPU, RAM, JAX backend) to size load thresholds."""

    def __init__(self) -> None:
        """Read CPU, total RAM and the active backend on the current host."""
        self.processor = platform.processor()
        self.ram_total_gb = psutil.virtual_memory().total / (1024**3)
        self.has_jax = HAS_JAX
        self.backend_device = self._detect_active_backend()
        self.max_tensor_dim = self._get_safe_tensor_limit()

    def _detect_active_backend(self) -> str:
        """Return a readable name of the active compute backend.

        Returns ``"CPU (NumPy Standard)"`` when JAX is absent, the JAX
        backend name when available, or ``"CPU (JAX Fallback)"`` when JAX
        is installed but no backend can be initialized on the host.
        """
        if not HAS_JAX:
            return "CPU (NumPy Standard)"
        try:
            backend = jax.default_backend()
            return f"{backend.upper()} (JAX Accelerated)"
        except RuntimeError:
            return "CPU (JAX Fallback)"

    def _get_safe_tensor_limit(self) -> int:
        """Maximum safe size for a single tensor, scaled on RAM and backend.

        GPU and TPU backends double the limit; RAM thresholds are 32 GB,
        12 GB and below.
        """
        multiplier = 2 if any(x in self.backend_device for x in ("GPU", "TPU")) else 1
        if self.ram_total_gb >= 32:
            return 8192 * multiplier
        if self.ram_total_gb >= 12:
            return 4096 * multiplier
        return 2048 * multiplier

    def get_profile_summary(self) -> str:
        """Return a one-line readable summary of the detected hardware."""
        return (
            f"Processor: {self.processor} | "
            f"RAM: {self.ram_total_gb:.1f} GB | "
            f"Engine: {self.backend_device} | "
            f"SafeMaxDim: {self.max_tensor_dim}"
        )


class StochasticAdversarialNoise:
    """Inject probabilistic or adversarial perturbations into AI tensors.

    Dispatches between the NumPy path (CPU) and the JAX path (GPU) based on
    the type of the input array.
    """

    SUPPORTED: ClassVar[set[str]] = {
        "bitflip",
        "dropout_noise",
        "gaussian_blur",
        "clean",
    }

    @staticmethod
    def inject_noise(
        data_vector: np.ndarray,
        noise_type: str,
        intensity: float,
        seed: int = 42,
    ) -> np.ndarray:
        """Apply a probabilistic alteration preserving the tensor norm.

        Args:
            data_vector: the input tensor.
            noise_type: one of ``bitflip``, ``dropout_noise``,
                ``gaussian_blur``, ``clean``.
            intensity: probability of triggering the alteration per element
                (or standard deviation for ``gaussian_blur``).
            seed: seed for the random generator.

        Returns:
            The altered tensor, renormalized to the original L2 norm.
        """
        noise_type = noise_type.lower().strip()

        if intensity <= 0.0 or noise_type == "clean":
            return data_vector

        if HAS_JAX and isinstance(data_vector, (jnp.ndarray, jax.Array)):
            key = jax.random.PRNGKey(seed)
            key, subkey = jax.random.split(key)
            trigger_mask = (
                jax.random.uniform(subkey, shape=data_vector.shape) < intensity
            )

            if noise_type == "bitflip":
                output = jnp.where(trigger_mask, -data_vector, data_vector)
            elif noise_type == "dropout_noise":
                output = jnp.where(trigger_mask, 0.0, data_vector)
            elif noise_type == "gaussian_blur":
                key, subkey2 = jax.random.split(key)
                noise = jax.random.normal(subkey2, shape=data_vector.shape) * intensity
                output = data_vector + noise
            else:
                output = data_vector

            norm = jnp.linalg.norm(output)
            return jnp.where(norm > 0, output / (norm + 1e-15), output)

        output = np.array(data_vector, copy=True)
        rng = np.random.default_rng(seed)
        trigger_mask = rng.random(size=output.shape) < intensity

        if noise_type == "bitflip":
            output = np.where(trigger_mask, -output, output)
        elif noise_type == "dropout_noise":
            output = np.where(trigger_mask, 0.0, output)
        elif noise_type == "gaussian_blur":
            noise = rng.normal(0.0, intensity, size=output.shape)
            output = output + noise

        norm = np.linalg.norm(output)
        return output / (norm + 1e-15) if norm > 0 else output
