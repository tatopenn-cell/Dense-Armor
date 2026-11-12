"""RAM and VRAM guard for heavy allocations.

:class:`UniversalMemoryGuard` checks free RAM and VRAM before an allocation
and computes the chunk split that avoids an out-of-memory error.
:class:`MemoryPressureError` is raised when the available memory is below
the configured threshold.
"""

import gc
import logging
import math
import subprocess

import psutil

try:
    import jax

    HAS_JAX = True
except ImportError:
    HAS_JAX = False

logger = logging.getLogger(__name__)


class MemoryPressureError(Exception):
    """Raised when system RAM or VRAM is below the configured threshold."""


class UniversalMemoryGuard:
    """Check free RAM and VRAM before a heavy allocation.

    Args:
        min_free_ram_percentage: minimum free RAM fraction required before
            an allocation is allowed.
        force_gc: when true, attempt a soft garbage collection before
            raising.
    """

    def __init__(
        self,
        min_free_ram_percentage: float = 0.15,
        force_gc: bool = True,
    ) -> None:
        """Store the RAM threshold and the soft-GC flag."""
        self.min_free_ram = min_free_ram_percentage
        self.force_gc = force_gc

    def _get_gpu_free_memory_nvidia(self) -> float:
        """Return the minimum free VRAM fraction over all NVIDIA GPUs.

        Handles the multi-GPU case by taking the minimum ratio across every
        device in the ``nvidia-smi`` output. When ``nvidia-smi`` is missing,
        fails, times out, or prints an unexpected format, the method returns
        ``1.0`` (assume free VRAM): no NVIDIA GPU is readable, so no block.

        Returns:
            A fraction in ``[0.0, 1.0]``.
        """
        try:
            cmd = ["nvidia-smi", "--query-gpu=memory.total,memory.free", "--format=csv,nounits,noheader"]
            output = subprocess.check_output(cmd, timeout=5).decode().strip()

            lines = [line.strip() for line in output.split("\n") if line.strip()]
            min_free_ratio = 1.0

            for line in lines:
                total, free = map(float, line.split(","))
                if total > 0:
                    min_free_ratio = min(min_free_ratio, free / total)
            return min_free_ratio
        except (
            subprocess.CalledProcessError,
            subprocess.TimeoutExpired,
            FileNotFoundError,
            OSError,
            ValueError,
        ):
            return 1.0

    def check_memory_safety(self) -> None:
        """Raise if free RAM or free VRAM are below the thresholds.

        When free RAM is within 10% of the threshold, a soft GC and a JAX
        cache clear run before the check is repeated. On a JAX host with a
        GPU device, free VRAM must stay above 5%; a driver or device query
        error is ignored (the check proceeds on the readable data).

        Raises:
            MemoryPressureError: when free RAM or VRAM are below the
                thresholds.
        """
        vm = psutil.virtual_memory()
        free_pct = vm.available / vm.total

        if self.force_gc and free_pct < (self.min_free_ram + 0.10):
            gc.collect()
            if HAS_JAX:
                try:
                    jax.clear_caches()
                except Exception:
                    logger.debug(
                        "jax.clear_caches() failed during soft GC", exc_info=True
                    )
            vm = psutil.virtual_memory()
            free_pct = vm.available / vm.total

        if free_pct < self.min_free_ram:
            raise MemoryPressureError(
                f"insufficient RAM: {free_pct:.1%} available, "
                f"minimum required: {self.min_free_ram:.1%}"
            )

        if HAS_JAX:
            try:
                for dev in jax.devices():
                    if dev.platform == "gpu":
                        vram_free = self._get_gpu_free_memory_nvidia()
                        if vram_free < 0.05:
                            raise MemoryPressureError(
                                f"VRAM exhausted on {dev.device_kind}: "
                                f"{vram_free:.1%} free."
                            )
            except MemoryPressureError:
                raise
            except (RuntimeError, AttributeError):
                pass

    def calculate_optimal_chunks(self, total_items: int, item_size_bytes: int) -> int:
        """Return the chunk split that fits the current free RAM.

        The allocatable window is set to 40% of free RAM to compensate for
        the temporary buffers produced during static graph tracing. A single
        chunk is returned when the whole payload fits inside that window.

        Args:
            total_items: number of items in the payload.
            item_size_bytes: size of one item, in bytes.

        Returns:
            The number of chunks, at least 1.
        """
        self.check_memory_safety()
        vm = psutil.virtual_memory()
        safe_allocatable_bytes = int(vm.available * 0.40)
        total_size_bytes = total_items * item_size_bytes

        if total_size_bytes <= safe_allocatable_bytes:
            return 1

        return max(math.ceil(total_size_bytes / safe_allocatable_bytes), 1)
