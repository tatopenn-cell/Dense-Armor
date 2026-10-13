"""JIT latency profiler for the codegen pipeline and the signal filter.

Both measurements separate the first call (XLA or vmap compilation) from the
steady-state runs, so the compilation cost does not pollute the per-call
latency.
"""

from typing import Any

import time

import jax
import numpy as np


class PipelineProfiler:
    """Measure pipeline and filter latency in microseconds."""

    @staticmethod
    def measure_microseconds(
        codegen_instance: Any,
        input_data: np.ndarray,
        compiled_ops: np.ndarray,
        repetitions: int = 100,
    ) -> dict:
        """Measure JIT latency of the ``DynamicAICodegen`` pipeline.

        Args:
            codegen_instance: an object exposing ``run_dynamic_pipeline``.
            input_data: the input array.
            compiled_ops: the compiled operation list.
            repetitions: number of steady-state runs.

        Returns:
            A dict with keys ``warmup_compilation_us`` (first run, XLA
            compilation), ``mean_execution_us``, ``std_execution_us``,
            ``min_execution_us``, ``repetitions``.
        """
        start_warmup = time.perf_counter_ns()
        warmup_res = codegen_instance.run_dynamic_pipeline(input_data, compiled_ops)
        _ = jax.block_until_ready(warmup_res)
        warmup_us = (time.perf_counter_ns() - start_warmup) / 1_000.0

        latencies = []
        for _ in range(repetitions):
            t0 = time.perf_counter_ns()
            res = codegen_instance.run_dynamic_pipeline(input_data, compiled_ops)
            _ = jax.block_until_ready(res)
            latencies.append((time.perf_counter_ns() - t0) / 1_000.0)

        return {
            "warmup_compilation_us": warmup_us,
            "mean_execution_us": float(np.mean(latencies)),
            "std_execution_us": float(np.std(latencies)),
            "min_execution_us": float(np.min(latencies)),
            "repetitions": repetitions,
        }

    @staticmethod
    def measure_stabilizer_microseconds(
        stabilizer_instance: Any,
        raw_batch: np.ndarray,
        repetitions: int = 100,
    ) -> dict:
        """Measure vmap latency of the ``AdaptiveSignalStabilizer`` filter.

        Args:
            stabilizer_instance: an object exposing ``filter_batch_scenarios``.
            raw_batch: the batch of raw signals.
            repetitions: number of steady-state runs.

        Returns:
            A dict with keys ``warmup_compilation_us`` (first run, vmap
            compilation), ``mean_execution_us``, ``std_execution_us``,
            ``min_execution_us``, ``repetitions``.
        """
        start_warmup = time.perf_counter_ns()
        warmup_res = stabilizer_instance.filter_batch_scenarios(raw_batch)
        _ = jax.block_until_ready(warmup_res)
        warmup_us = (time.perf_counter_ns() - start_warmup) / 1_000.0

        latencies = []
        for _ in range(repetitions):
            t0 = time.perf_counter_ns()
            res = stabilizer_instance.filter_batch_scenarios(raw_batch)
            _ = jax.block_until_ready(res)
            latencies.append((time.perf_counter_ns() - t0) / 1_000.0)

        return {
            "warmup_compilation_us": warmup_us,
            "mean_execution_us": float(np.mean(latencies)),
            "std_execution_us": float(np.std(latencies)),
            "min_execution_us": float(np.min(latencies)),
            "repetitions": repetitions,
        }
