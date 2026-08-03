# Toolkit (standalone utilities)

A second part of the package, under `core/`/`utility/`, independent of [`Armatura`](armatura.md)
and [`Orca`](orca.md) — none of it participates in the anomaly shield. Generic tools for JAX/NumPy
pipelines: a small op-compiler, a memory guard, hardware profiling, logging, provenance export,
and a couple of audio/data I/O helpers. Each is tested on its own (`test/test_chunk.py`,
`test_compiler.py`, `test_memory.py`, `test_preset.py`, `test_tensor.py`, `test_noise.py`,
`test_vector.py`, `test_profiler.py`, `test_visualizer.py`, `test_logger.py`, `test_anwav.py`,
`test_diagnostic.py`, `test_iodat.py`, `test_resonance_search.py`).

## Pipeline & chunking

Compiles a list of operation names into a JIT-compiled JAX pipeline, and splits large batches
or instruction lists into fixed-size blocks.

::: dense_armor.core.compiler

::: dense_armor.core.chunk

## Memory guard

::: dense_armor.core.memory

## Hardware & profiling

Detects the host's CPU/RAM/backend to pick safe tensor size limits, injects synthetic
adversarial noise for testing a detector, and measures JIT latency in microseconds.

::: dense_armor.core.noise

::: dense_armor.core.profiler

## Tensors & configuration

::: dense_armor.core.tensor

::: dense_armor.core.vector

::: dense_armor.core.preset

## Logging & provenance

Two `logging.Formatter` subclasses, plus SHA-256-signed provenance export and text trend
reports.

::: dense_armor.core.logger

::: dense_armor.core.visualizer

## Audio & data I/O

::: dense_armor.utility.anwav

::: dense_armor.utility.diagnostic

::: dense_armor.utility.iodat

## Similarity search

Cosine-similarity resonance score between a query vector and the rows of a matrix, modulated
by `apply_damping_blend` (the same operator Orca's Stage 1/2 gating uses).

::: dense_armor.utility.resonance_search
