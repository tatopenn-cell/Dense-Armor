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
tensor size for it. Honest caveat: the RAM tiers behind `max_tensor_dim` (2048/4096/8192,
doubled on GPU/TPU) are a rough heuristic, not calibrated against anything specific to this
package -- treat it as a starting guess, not a guarantee.

```python
from dense_armor.core import AIHardwareProfiler

profile = AIHardwareProfiler()
print(profile.get_profile_summary())
# Processor: ... | RAM: 7.9 GB | Engine: CPU (JAX Accelerato) | SafeMaxDim: 2048
```

**`StochasticAdversarialNoise`** injects synthetic noise (bitflip, dropout, gaussian blur) into
a tensor while preserving its norm. Honest caveat: this is a generic noise injector, not a real
adversarial-example generator -- for actually testing the shield's robustness, the attacks in
[`test/test_boundA.py`](https://github.com/tatopenn-cell/Dense-Armor/blob/master/test/test_boundA.py)–[`test_boundE.py`](https://github.com/tatopenn-cell/Dense-Armor/blob/master/test/test_boundE.py)
(PGD/BIM/MI-FGSM, Carlini-Wagner, DeepFool, Fourier) are the real, calibrated benchmark; this
module overlaps with that suite rather than adding to it.

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
call timed separately from steady-state calls. This is the module that caught a real bug:
`DynamicAICodegen`'s kernels used to be re-defined (and re-`jax.jit`-wrapped) on every single
call, so they never reused XLA's compilation cache -- warm-up and steady-state timed almost
identically. Once fixed, the split is real: warm-up is 1700x+ slower than steady-state on a
small pipeline.

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
(JAX/NumPy) and precision auto-detected. Honest caveat: these are tiny, fixed matrices (2x2 or
a 3-element kernel) -- writing one inline is a single line of code. The real value here is the
backend/precision auto-detection, not the matrix catalog itself.

```python
from dense_armor.core import TensorVault

vault = TensorVault()
edge = vault.get_static_transform("edge_detector")
# edge -> [-1. 2. -1.]
```

::: dense_armor.core.tensor

---

**`ParametricScenarioSimulator`** runs parallel Monte Carlo simulations over time (via
`jax.vmap`), plus a stochastic decision collapse driven by a probability distribution. Honest
caveat: the per-step update (`next_state = current_state * 0.95 + param * 0.05`) is a fixed
exponential-moving-average weighting, not a configurable simulation model -- useful mainly if
that specific dynamic matches your scenario, not as a general-purpose simulator.

```python
from dense_armor.core import ParametricScenarioSimulator
import numpy as np

sim = ParametricScenarioSimulator()
result, collapsed = sim.collapse_decision(np.array([0.1, 0.2, 0.3, 0.4]), target_idx=2)
# result -> 0 or 1 (stochastic); collapsed -> the vector renormalized after the choice
```

**`BitwisePermutationEngine`** swaps elements of a combinatorial vector (a 2^n-sized space)
based on target/control bit masks. Honest caveat: each call performs exactly one
controlled-swap between one pair of indices -- a single primitive, not a general permutation
engine. Narrower than the name suggests.

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
[`AdaptiveSignalStabilizer`](engine.md) (Orca's Stage 1). Verified, not just declared: on the
same noisy series with an outlier, `pure_1d_time_v1` (tuned for a more reactive regime) leaves
over 2x the residual variance of `balanced_v2` -- the presets genuinely configure different
filtering behavior, not just different numbers that happen to look distinct.

```python
from dense_armor.core.preset import SIGNAL_STABILIZER_PRESETS
from dense_armor.core.engine import AdaptiveSignalStabilizer

stabilizer = AdaptiveSignalStabilizer(**SIGNAL_STABILIZER_PRESETS["balanced_v2"])
```

::: dense_armor.core.preset

## Logging & provenance

**`MinimalConsoleFormatter`** / **`CompactJsonFormatter`** are two `logging.Formatter`
subclasses — one human-readable for the console, one compact JSON for a log file. Honest
caveat: fairly thin wrappers around `logging.Formatter` -- `CompactJsonFormatter`'s structured
fields (module/filename/line number, one JSON object per event) are the main reason to reach
for this over writing a one-line formatter yourself.

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
uses). Verified, not just declared: the modulation is load-bearing, not decorative --
`kappa` (the damping weight) measurably changes the score (`kappa=0` vs. `kappa=1` differ well
beyond floating-point noise on the same inputs), so this is genuinely different from plain
cosine similarity, not a rebrand of it.

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
