# -*- coding: utf-8 -*-
"""
core/profiler.py
===========================
PipelineProfiler — misura latenze JIT in microsecondi con warm-up XLA separato.
"""

import time

import jax
import numpy as np


class PipelineProfiler:
    """Profila le prestazioni della pipeline e del filtro in microsecondi."""

    @staticmethod
    def measure_microseconds(
        codegen_instance,
        input_data:   np.ndarray,
        compiled_ops: np.ndarray,
        repetitions:  int = 100,
    ) -> dict:
        """
        Misura latenza JIT della pipeline DynamicAICodegen.

        Returns
        -------
        dict con chiavi:
            warmup_compilation_us  — prima esecuzione (compilazione XLA)
            mean_execution_us      — media a regime
            repetitions            — numero di run
        """
        start_warmup = time.perf_counter_ns()
        warmup_res   = codegen_instance.run_dynamic_pipeline(input_data, compiled_ops)
        _            = jax.block_until_ready(warmup_res)
        warmup_us    = (time.perf_counter_ns() - start_warmup) / 1_000.0

        latencies = []
        for _ in range(repetitions):
            t0  = time.perf_counter_ns()
            res = codegen_instance.run_dynamic_pipeline(input_data, compiled_ops)
            _   = jax.block_until_ready(res)
            latencies.append((time.perf_counter_ns() - t0) / 1_000.0)

        return {
            "warmup_compilation_us": warmup_us,
            "mean_execution_us":     float(np.mean(latencies)),
            "std_execution_us":      float(np.std(latencies)),
            "min_execution_us":      float(np.min(latencies)),
            "repetitions":           repetitions,
        }

    @staticmethod
    def measure_stabilizer_microseconds(
        stabilizer_instance,
        raw_batch:   np.ndarray,
        repetitions: int = 100,
    ) -> dict:
        """
        Misura latenza vmap del filtro AdaptiveSignalStabilizer.

        Returns
        -------
        dict con chiavi:
            warmup_compilation_us  — prima esecuzione (compilazione vmap)
            mean_execution_us      — media a regime
            repetitions            — numero di run
        """
        start_warmup = time.perf_counter_ns()
        warmup_res   = stabilizer_instance.filter_batch_scenarios(raw_batch)
        _            = jax.block_until_ready(warmup_res)
        warmup_us    = (time.perf_counter_ns() - start_warmup) / 1_000.0

        latencies = []
        for _ in range(repetitions):
            t0  = time.perf_counter_ns()
            res = stabilizer_instance.filter_batch_scenarios(raw_batch)
            _   = jax.block_until_ready(res)
            latencies.append((time.perf_counter_ns() - t0) / 1_000.0)

        return {
            "warmup_compilation_us": warmup_us,
            "mean_execution_us":     float(np.mean(latencies)),
            "std_execution_us":      float(np.std(latencies)),
            "min_execution_us":      float(np.min(latencies)),
            "repetitions":           repetitions,
        }