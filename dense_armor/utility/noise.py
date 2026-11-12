"""Host hardware profiler.

:class:`AIHardwareProfiler` reads the CPU, the total RAM and the active JAX
backend to size the safe tensor limit.
"""

import platform

import psutil

try:
    import jax

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
