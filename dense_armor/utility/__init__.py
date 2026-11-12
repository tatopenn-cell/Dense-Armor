"""Public surface of the :mod:`dense_armor.utility` package."""

from .. import __version__
from .damping_operator import apply_damping_blend
from .engine import AdaptiveSignalStabilizer
from .memory import MemoryPressureError, UniversalMemoryGuard
from .noise import AIHardwareProfiler

__all__ = [
    "AIHardwareProfiler",
    "AdaptiveSignalStabilizer",
    "MemoryPressureError",
    "UniversalMemoryGuard",
    "apply_damping_blend",
]
