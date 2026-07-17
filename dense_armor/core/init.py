# -*- coding: utf-8 -*-
"""
core/__init__.py
================
TensorFlowEngine (Sentinel) — Core Package Production Map.
Autore del Framework: Salvatore Pennacchio (Napoli, 2026)
"""

# Importazioni allineate rigorosamente al file system reale della cartella core/
from .noise     import AIHardwareProfiler, StochasticAdversarialNoise
from .memory    import UniversalMemoryGuard, MemoryPressureError
from .vector    import ParametricScenarioSimulator, BitwisePermutationEngine
from .engine    import AdaptiveSignalStabilizer
from .compiler  import DynamicAICodegen, CMD_MAP
from .tensor    import TensorVault
from .visualizer import AIEngineVisualizer
from .profiler  import PipelineProfiler
from .damping_operator import apply_damping_blend

__version__ = "1.0.4"
__all__ = [
    "AIHardwareProfiler", 
    "StochasticAdversarialNoise",
    "UniversalMemoryGuard", 
    "MemoryPressureError",
    "ParametricScenarioSimulator", 
    "BitwisePermutationEngine",
    "AdaptiveSignalStabilizer",
    "DynamicAICodegen", 
    "CMD_MAP",
    "TensorVault",
    "AIEngineVisualizer",
    "PipelineProfiler",
    "apply_damping_blend"
]
