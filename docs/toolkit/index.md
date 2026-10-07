# Toolkit

Dense-Armor has two parts. The first is the shields and detectors — everything that
touches a signal and cleans it. The second is the toolkit: generic tools for JAX and
NumPy pipelines that the shields use internally and that work on their own.

None of the toolkit modules participates in the anomaly shield. Nothing here changes
the values of a signal, and nothing here detects anything. They are helpers:
compile a pipeline of operations into a single JIT function, guard a large allocation
against out-of-memory, profile a JIT pipeline, log a run, read a WAV or an HDF5.

## Pages

- **[Toolkit](toolkit.md)** — the full list with a runnable example for each module:
  op-compiler and chunker, memory guard, hardware profiler and noise injector, tensor
  catalog, logging and provenance, audio and data I/O, and similarity search.
