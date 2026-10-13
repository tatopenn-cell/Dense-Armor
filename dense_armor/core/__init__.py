"""Public surface of the :mod:`dense_armor.core` package."""

from .compiler import CMD_MAP, DynamicAICodegen
from .damping_operator import apply_damping_blend
from .engine import AdaptiveSignalStabilizer
from .memory import MemoryPressureError, UniversalMemoryGuard
from .noise import AIHardwareProfiler, StochasticAdversarialNoise
from .profiler import PipelineProfiler
from .tensor import TensorVault
from .vector import BitwisePermutationEngine, ParametricScenarioSimulator
from .visualizer import AIEngineVisualizer

__version__ = "1.0.11"
__all__ = [
    "CMD_MAP",
    "AIEngineVisualizer",
    "AIHardwareProfiler",
    "AdaptiveSignalStabilizer",
    "BitwisePermutationEngine",
    "DynamicAICodegen",
    "MemoryPressureError",
    "ParametricScenarioSimulator",
    "PipelineProfiler",
    "StochasticAdversarialNoise",
    "TensorVault",
    "UniversalMemoryGuard",
    "apply_damping_blend",
]
