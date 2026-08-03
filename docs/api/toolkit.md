# Toolkit (standalone utilities)

A second part of the package, under `core/`/`utility/`, independent of [`Armatura`](armatura.md)
and [`Orca`](orca.md) — none of it participates in the anomaly shield. Generic tools for JAX/NumPy
pipelines. Each is tested on its own (`test/test_chunk.py`, `test_compiler.py`, `test_memory.py`,
`test_preset.py`, `test_tensor.py`, `test_noise.py`, `test_vector.py`, `test_profiler.py`,
`test_visualizer.py`, `test_logger.py`, `test_anwav.py`, `test_diagnostic.py`, `test_iodat.py`,
`test_resonance_search.py`); every example below was run for real before being written down.

## Pipeline & chunking

**`DynamicAICodegen`** compiles a list of operation names (`relu`, `sigmoid`, `tanh`, `scale`,
`dropout`, `clip`, `l2_normalize`, `identity`) into a single JIT-compiled JAX pipeline. Useful
when you want to describe a small transformation pipeline declaratively (as data, not as
hand-written JAX code) and still get one compiled kernel plus a gradient for free.

```python
from dense_armor.core import DynamicAICodegen

codegen = DynamicAICodegen()
ops = codegen.compile_pipeline(["relu", "l2_normalize"])
out = codegen.run_dynamic_pipeline([-2.0, 3.0, -1.0, 4.0], ops)
# out -> [0. 0.6 0. 0.8]
```

::: dense_armor.core.compiler

---

**`ImageChunker`** splits a large batch (or a long list of compiled operations) into fixed-size
blocks, and merges the results back. Useful when a batch doesn't fit in memory in one shot, or
when a long instruction list would otherwise force XLA to recompile every time its length
changes.

```python
from dense_armor.core.chunk import ImageChunker
import numpy as np

chunker = ImageChunker(chunk_size=2)
chunks = chunker.split_array(np.arange(5))
# chunks -> [array([0, 1]), array([2, 3]), array([4])]
merged = chunker.merge_chunks(chunks)
# merged -> array([0, 1, 2, 3, 4])
```

::: dense_armor.core.chunk

## Memory guard

**`UniversalMemoryGuard`** checks free RAM (and VRAM, if an NVIDIA GPU is present) before a
heavy allocation, and computes how many chunks a batch needs to fit safely. Useful as a guard
rail right before a large `jax`/`numpy` allocation you don't want to OOM on.

```python
from dense_armor.core import UniversalMemoryGuard

guard = UniversalMemoryGuard(min_free_ram_percentage=0.10)
guard.check_memory_safety()  # raises MemoryPressureError if RAM is too low
```

::: dense_armor.core.memory

## Hardware & profiling

**`AIHardwareProfiler`** detects the host's CPU/RAM/JAX backend and computes a safe maximum
tensor size for it. Useful for picking batch/tensor sizes automatically instead of hardcoding
a limit that's wrong on a different machine.

```python
from dense_armor.core import AIHardwareProfiler

profile = AIHardwareProfiler()
print(profile.get_profile_summary())
# Processor: ... | RAM: 7.9 GB | Engine: CPU (JAX Accelerato) | SafeMaxDim: 2048
```

**`StochasticAdversarialNoise`** injects synthetic noise (bitflip, dropout, gaussian blur) into
a tensor while preserving its norm. Useful for generating attack-like data to stress-test a
detector, without needing a real adversarial-example generator.

```python
from dense_armor.core import StochasticAdversarialNoise
import numpy as np

out = StochasticAdversarialNoise.inject_noise(
    np.array([1.0, 1.0, 1.0, 1.0]), "bitflip", intensity=1.0, seed=0
)
# out -> [-0.5 -0.5 -0.5 -0.5]  (all flipped + renormalized)
```

::: dense_armor.core.noise

---

**`PipelineProfiler`** measures JIT latency in microseconds, with the first (compilation)
call timed separately from steady-state calls. Useful for checking whether a pipeline is
actually running at its compiled speed, or silently recompiling on every call.

```python
from dense_armor.core import DynamicAICodegen, PipelineProfiler
import numpy as np

codegen = DynamicAICodegen()
ops = codegen.compile_pipeline(["relu", "tanh"])
stats = PipelineProfiler.measure_microseconds(codegen, np.array([1.0, -2.0, 3.0]), ops, repetitions=5)
# stats -> {"warmup_compilation_us": ..., "mean_execution_us": ..., "repetitions": 5, ...}
```

::: dense_armor.core.profiler

## Tensors & configuration

**`TensorVault`** is a small library of static (`invert`, `identity`, `edge_detector`, `blend`)
and parametric (`scale_project`, `amplify`, `bias_shift`) transformation matrices, with backend
(JAX/NumPy) and precision auto-detected. Useful when you need one of these common matrices
without hand-writing it for both backends.

```python
from dense_armor.core import TensorVault

vault = TensorVault()
edge = vault.get_static_transform("edge_detector")
# edge -> [-1. 2. -1.]
```

::: dense_armor.core.tensor

---

**`ParametricScenarioSimulator`** runs parallel Monte Carlo simulations over time (via
`jax.vmap`), plus a stochastic decision collapse driven by a probability distribution. Useful
for scenario sweeps where each scenario evolves independently from the same starting state.

```python
from dense_armor.core import ParametricScenarioSimulator
import numpy as np

sim = ParametricScenarioSimulator()
result, collapsed = sim.collapse_decision(np.array([0.1, 0.2, 0.3, 0.4]), target_idx=2)
# result -> 0 or 1 (stochastic); collapsed -> the vector renormalized after the choice
```

**`BitwisePermutationEngine`** swaps elements of a combinatorial vector (a 2^n-sized space)
based on target/control bit masks. Useful for combinatorial state manipulation where indices
are addressed by bit position rather than a plain integer index.

```python
from dense_armor.core import BitwisePermutationEngine
import numpy as np

engine = BitwisePermutationEngine(n_elements=2)  # 2^2 = 4 states
out = engine.apply_bitwise_swap(np.array([0., 1., 2., 3.]), target_bit=1, control_bit=0)
# out -> [0. 1. 3. 2.]
```

::: dense_armor.core.vector

---

**`SIGNAL_STABILIZER_PRESETS`** are 4 empirically-calibrated parameter sets
(`balanced_v2`, `cifar10_best_v1`, `pure_1d_time_v1`, `cifar10_hardened_lyapunov`) for
[`AdaptiveSignalStabilizer`](engine.md) (Orca's Stage 1). Useful as a starting point instead of
guessing thresholds/damping/alpha from scratch.

```python
from dense_armor.core.preset import SIGNAL_STABILIZER_PRESETS
from dense_armor.core.engine import AdaptiveSignalStabilizer

stabilizer = AdaptiveSignalStabilizer(**SIGNAL_STABILIZER_PRESETS["balanced_v2"])
```

::: dense_armor.core.preset

## Logging & provenance

**`MinimalConsoleFormatter`** / **`CompactJsonFormatter`** are two `logging.Formatter`
subclasses — one human-readable for the console, one compact JSON for a log file. Useful to
plug into standard `logging` when you want either format without writing a formatter by hand.

```python
import logging
from dense_armor.core.logger import MinimalConsoleFormatter

handler = logging.StreamHandler()
handler.setFormatter(MinimalConsoleFormatter())
log = logging.getLogger("demo")
log.addHandler(handler)
log.setLevel(logging.INFO)
log.info("esempio")
# [13:07:27] [INFO] esempio
```

::: dense_armor.core.logger

---

**`AIEngineVisualizer`** exports a SHA-256-signed provenance archive (parameters, execution
environment, integrity hash) and plain-text trend reports comparing raw vs. filtered variance.
Useful when you need an auditable record of a run, not just its output.

```python
from dense_armor.core import AIEngineVisualizer

viz = AIEngineVisualizer(output_dir=".")
sha256 = viz.export_provenance_archive([{"step": 1, "value": 0.5}], filename="archive.json")
# sha256 -> "b40db71b3b16d081..." (64 hex chars, matches the hash written into archive.json)
```

::: dense_armor.core.visualizer

## Audio & data I/O

**`anwav(fpath)`** analyzes a WAV file: peak, RMS, estimated loudness (LUFS), crest factor,
with a plain-text compliance verdict. Useful as a quick sanity check on an audio file's levels.

```python
from dense_armor.utility.anwav import anwav

anwav("track.wav")
# -> File                      : track.wav
# -> Picco Massimo             : -6.02 dBFS
# ...
# [VERDETTO STANDARD]:
#    CONFORME (Peak): Picco in sicurezza sotto i -1.0 dB.
```

::: dense_armor.utility.anwav

---

**`diag(iorig, ifilt)`** compares two audio signals (file paths or NumPy arrays): structural
fidelity, removed energy, distortion peak. Useful for checking how much an audio filter/process
actually changed a signal, beyond just listening to it.

```python
from dense_armor.utility.diagnostic import diag
import numpy as np

rng = np.random.default_rng(0)
originale = rng.normal(size=2000).astype(np.float32)
filtrato = originale * 0.98
risultato = diag(originale, filtrato)
# risultato["fedelta"] -> 99.96  (percent structural fidelity preserved)
```

::: dense_armor.utility.diagnostic

---

**`lodat(fpath, dname)`** reads a named tensor out of an HDF5 or NetCDF file. Useful as a thin,
uniform loader when a pipeline needs to accept either format without branching on the caller's
side.

```python
from dense_armor.utility.iodat import lodat
import h5py, numpy as np

with h5py.File("data.h5", "w") as f:
    f.create_dataset("temperature", data=np.arange(12).reshape(3, 4))

tensore = lodat("data.h5", "temperature")
# tensore.shape -> (3, 4)
```

::: dense_armor.utility.iodat

## Similarity search

**`apply_fast_resonance(matrix, query)`** scores cosine similarity between a query vector and
each row of a matrix, modulated by `apply_damping_blend` (the same operator Orca's gating
uses). Useful for a quick nearest-neighbor-style lookup consistent with the rest of the
package's math, rather than pulling in a separate similarity-search library.

```python
from dense_armor.utility.resonance_search import apply_fast_resonance
import numpy as np

rng = np.random.default_rng(0)
db = rng.standard_normal((5, 8)).astype(np.float32)
query = db[2].copy()  # an exact copy of row 2
scores = apply_fast_resonance(db, query)
# int(scores.argmax()) -> 2  (the matching row scores highest)
```

::: dense_armor.utility.resonance_search
