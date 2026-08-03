# -*- coding: utf-8 -*-
import numpy as np

from dense_armor.core.compiler import DynamicAICodegen
from dense_armor.core.engine import AdaptiveSignalStabilizer
from dense_armor.core.profiler import PipelineProfiler


def test_measure_microseconds_ritorna_le_chiavi_attese():
    codegen = DynamicAICodegen()
    ops = codegen.compile_pipeline(["relu", "tanh"])
    stats = PipelineProfiler.measure_microseconds(codegen, np.ones(16), ops, repetitions=2)

    assert stats["repetitions"] == 2
    assert stats["warmup_compilation_us"] > 0
    assert stats["mean_execution_us"] >= 0
    assert stats["min_execution_us"] >= 0


def test_measure_stabilizer_microseconds_ritorna_le_chiavi_attese():
    stabilizer = AdaptiveSignalStabilizer()
    raw_batch = np.random.default_rng(0).normal(size=(2, 8))
    stats = PipelineProfiler.measure_stabilizer_microseconds(stabilizer, raw_batch, repetitions=2)

    assert stats["repetitions"] == 2
    assert stats["warmup_compilation_us"] > 0
    assert stats["mean_execution_us"] >= 0
