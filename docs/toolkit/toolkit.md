# Toolkit (standalone utilities)

A second part of the package, under `core/` and `utility/`, independent of
[`Armatura`](../shields/armatura.md) and [`Orca`](../shields/orca.md) — none of it
participates in the anomaly shield. Generic tools for JAX / NumPy pipelines.

Every example below was run for real before being written down. Each module is tested
on its own (`test/test_chunk.py`, `test_compiler.py`, `test_memory.py`,
`test_preset.py`, `test_tensor.py`, `test_noise.py`, `test_vector.py`,
`test_profiler.py`, `test_visualizer.py`, `test_logger.py`, `test_anwav.py`,
`test_diagnostic.py`, `test_iodat.py`, `test_resonance_search.py`).

## Pipeline and chunking

**`DynamicAICodegen`** compiles a list of operation names (`relu`, `sigmoid`, `tanh`,
`scale`, `dropout`, `clip`, `l2_normalize`, `identity`) into a single JIT-compiled JAX
pipeline. Useful when you want to describe a transformation pipeline declaratively
(as data, not as hand-written JAX code) and still get one compiled kernel plus a
gradient for free.

```python
from dense_armor.core import DynamicAICodegen

codegen = DynamicAICodegen()
ops = codegen.compile_pipeline(["relu", "l2_normalize"])
out = codegen.run_dynamic_pipeline([-2.0, 3.0, -1.0, 4.0], ops)
```

`out` is `[0., 0.6, 0., 0.8]`: relu clips the negatives, then L2-normalize.

::: dense_armor.core.compiler

---

**`ImageChunker`** splits a large batch (or a long list of compiled operations) into
fixed-size blocks, and merges the results back. Useful when a batch does not fit in
memory in one shot, or when a long instruction list would otherwise force XLA to
recompile every time its length changes.

```python
import numpy as np
from dense_armor.core.chunk import ImageChunker

chunker = ImageChunker(chunk_size=2)
chunks = chunker.split_array(np.arange(5))
merged = chunker.merge_chunks(chunks)
```

`chunks` is `[array([0, 1]), array([2, 3]), array([4])]`; `merged` is
`array([0, 1, 2, 3, 4])`.

::: dense_armor.core.chunk

## Memory guard

**`UniversalMemoryGuard`** checks free RAM (and VRAM, if an NVIDIA GPU is present)
before a heavy allocation, and computes how many chunks a batch needs to fit safely.
Useful as a guard-rail right before a large `jax` / `numpy` allocation you do not want
to OOM on.

```python
from dense_armor.core import UniversalMemoryGuard

guard = UniversalMemoryGuard(min_free_ram_percentage=0.10)
guard.check_memory_safety()
```

Raises `MemoryPressureError` if free RAM is below 10 %.

::: dense_armor.core.memory

## Hardware and profiling

**`AIHardwareProfiler`** detects the host's CPU / RAM / JAX backend and computes a
safe maximum tensor size for it.

```python
from dense_armor.core import AIHardwareProfiler

profile = AIHardwareProfiler()
print(profile.get_profile_summary())
```

**Honest caveat**: the RAM tiers behind `max_tensor_dim` (2048 / 4096 / 8192, doubled
on GPU / TPU) are a rough heuristic, not calibrated against anything specific to this
package. Treat it as a starting guess, not a guarantee.

**`StochasticAdversarialNoise`** injects synthetic noise (bitflip, dropout, Gaussian
blur) into a tensor while preserving its norm.

```python
import numpy as np
from dense_armor.core import StochasticAdversarialNoise

out = StochasticAdversarialNoise.inject_noise(
    np.array([1.0, 1.0, 1.0, 1.0]), "bitflip", intensity=1.0, seed=0,
)
```

**Honest caveat**: this is a generic noise injector, not a real adversarial-example
generator. For actually testing the shield's robustness, the attacks in
`test/test_boundA.py`–`test_boundE.py` (PGD / BIM / MI-FGSM, Carlini-Wagner, DeepFool,
Fourier) are the real, calibrated benchmark; this module overlaps with that suite
rather than adding to it.

::: dense_armor.core.noise

---

**`PipelineProfiler`** measures JIT latency in microseconds, with the first
(compilation) call timed separately from steady-state calls. This is the module that
caught a real bug: `DynamicAICodegen`'s kernels used to be re-defined (and
re-`jax.jit`-wrapped) on every single call, so they never reused XLA's compilation
cache — warm-up and steady-state timed almost identically. Once fixed, the split is
real: warm-up is 1700×+ slower than steady-state on a small pipeline.

```python
import numpy as np
from dense_armor.core import DynamicAICodegen, PipelineProfiler

codegen = DynamicAICodegen()
ops = codegen.compile_pipeline(["relu", "tanh"])
stats = PipelineProfiler.measure_microseconds(
    codegen, np.array([1.0, -2.0, 3.0]), ops, repetitions=5,
)
```

`stats` is a dict with keys `warmup_compilation_us`, `mean_execution_us`,
`repetitions`.

::: dense_armor.core.profiler

## Tensors and configuration

**`TensorVault`** is a small library of static (`invert`, `identity`, `edge_detector`,
`blend`) and parametric (`scale_project`, `amplify`, `bias_shift`) transformation
matrices, with backend (JAX / NumPy) and precision auto-detected.

```python
from dense_armor.core import TensorVault

vault = TensorVault()
edge = vault.get_static_transform("edge_detector")
```

`edge` is `[-1., 2., -1.]`.

**Honest caveat**: these are tiny, fixed matrices (2×2 or a 3-element kernel) —
writing one inline is a single line of code. The real value here is the
backend/precision auto-detection, not the matrix catalog itself.

::: dense_armor.core.tensor

---

**`ParametricScenarioSimulator`** runs parallel Monte Carlo simulations over time (via
`jax.vmap`), plus a stochastic decision collapse driven by a probability distribution.

```python
import numpy as np
from dense_armor.core import ParametricScenarioSimulator

sim = ParametricScenarioSimulator()
result, collapsed = sim.collapse_decision(np.array([0.1, 0.2, 0.3, 0.4]), target_idx=2)
```

**Honest caveat**: the per-step update (`next_state = current_state * 0.95 + param * 0.05`)
is a fixed exponential-moving-average weighting, not a configurable simulation model.
Useful mainly if that specific dynamic matches your scenario, not as a general-purpose
simulator.

**`BitwisePermutationEngine`** swaps elements of a combinatorial vector (a
`2^n`-sized space) based on target / control bit masks.

```python
import numpy as np
from dense_armor.core import BitwisePermutationEngine

engine = BitwisePermutationEngine(n_elements=2)
out = engine.apply_bitwise_swap(np.array([0., 1., 2., 3.]), target_bit=1, control_bit=0)
```

`out` is `[0., 1., 3., 2.]`.

**Honest caveat**: each call performs exactly one controlled swap between one pair of
indices — a single primitive, not a general permutation engine. Narrower than the
name suggests.

::: dense_armor.core.vector

---

**`SIGNAL_STABILIZER_PRESETS`** are 4 empirically-calibrated parameter sets
(`balanced_v2`, `cifar10_best_v1`, `pure_1d_time_v1`, `cifar10_hardened_lyapunov`) for
[`AdaptiveSignalStabilizer`](../shields/engine.md) (Orca's Stage 1).

```python
from dense_armor.core.preset import SIGNAL_STABILIZER_PRESETS
from dense_armor.core.engine import AdaptiveSignalStabilizer

stabilizer = AdaptiveSignalStabilizer(**SIGNAL_STABILIZER_PRESETS["balanced_v2"])
```

Verified, not just declared: on the same noisy series with an outlier,
`pure_1d_time_v1` (tuned for a more reactive regime) leaves over 2× the residual
variance of `balanced_v2` — the presets genuinely configure different filtering
behaviour, not just different numbers that happen to look distinct.

::: dense_armor.core.preset

## Logging and provenance

**`MinimalConsoleFormatter`** / **`CompactJsonFormatter`** are two
`logging.Formatter` subclasses — one human-readable for the console, one compact JSON
for a log file.

```python
import logging
from dense_armor.core.logger import MinimalConsoleFormatter

handler = logging.StreamHandler()
handler.setFormatter(MinimalConsoleFormatter())
log = logging.getLogger("demo")
log.addHandler(handler)
log.setLevel(logging.INFO)
log.info("esempio")
```

```
[13:07:27] [INFO] esempio
```

**Honest caveat**: fairly thin wrappers around `logging.Formatter` —
`CompactJsonFormatter`'s structured fields (module / filename / line number, one JSON
object per event) are the main reason to reach for this over writing a one-line
formatter yourself.

::: dense_armor.core.logger

---

**`AIEngineVisualizer`** exports a SHA-256-signed provenance archive (parameters,
execution environment, integrity hash) and plain-text trend reports comparing raw vs.
filtered variance. Useful when you need an auditable record of a run, not just its
output.

```python
from dense_armor.core import AIEngineVisualizer

viz = AIEngineVisualizer(output_dir=".")
sha256 = viz.export_provenance_archive([{"step": 1, "value": 0.5}], filename="archive.json")
```

`sha256` is a 64-hex-character string that matches the hash written into
`archive.json`.

::: dense_armor.core.visualizer

## Audio and data I/O

**`anwav(fpath)`** analyzes a WAV file: peak, RMS, estimated loudness (LUFS), crest
factor, with a plain-text compliance verdict.

```python
from dense_armor.utility.misc.anwav import anwav

anwav("track.wav")
```

Prints a summary table and a verdict line (`CONFORME (Peak): Picco in sicurezza sotto
i -1.0 dB` or similar).

::: dense_armor.utility.misc.anwav

---

**`diag(iorig, ifilt)`** compares two audio signals (file paths or NumPy arrays):
structural fidelity, removed energy, distortion peak. Useful for checking how much an
audio filter / process actually changed a signal, beyond just listening to it.

```python
import numpy as np
from dense_armor.utility.misc.diagnostic import diag

rng = np.random.default_rng(0)
originale = rng.normal(size=2000).astype(np.float32)
filtrato = originale * 0.98
risultato = diag(originale, filtrato)
```

`risultato["fedelta"]` is close to 99.96 — the percent of structural fidelity
preserved.

::: dense_armor.utility.misc.diagnostic

---

**`lodat(fpath, dname)`** reads a named tensor out of an HDF5 or NetCDF file. Useful
as a thin, uniform loader when a pipeline needs to accept either format without
branching on the caller's side.

```python
import h5py, numpy as np
from dense_armor.utility.misc.iodat import lodat

with h5py.File("data.h5", "w") as f:
    f.create_dataset("temperature", data=np.arange(12).reshape(3, 4))

tensore = lodat("data.h5", "temperature")
```

`tensore.shape` is `(3, 4)`.

::: dense_armor.utility.misc.iodat

## Similarity search

**`apply_fast_resonance(matrix, query)`** scores cosine similarity between a query
vector and each row of a matrix, modulated by `apply_damping_blend` (the same operator
Orca's gating uses).

```python
import numpy as np
from dense_armor.utility.anomaly.resonance_search import apply_fast_resonance

rng = np.random.default_rng(0)
db = rng.standard_normal((5, 8)).astype(np.float32)
query = db[2].copy()
scores = apply_fast_resonance(db, query)
```

`int(scores.argmax())` is `2` — the matching row scores highest.

### An honest finding: the modulation does not change ranking

`kappa` (the damping weight) does measurably change the score values: `kappa=0` vs
`kappa=1` differ well beyond floating-point noise on the same inputs.

But for retrieval, what matters is *ranking*, not the absolute score, and there the
modulation is a **confound, not a real effect**. A real benchmark on quantumrag
(1855 chunks, 12 labelled queries, Mean Reciprocal Rank) gave MRR = 0.8125 for plain
cosine, and the identical MRR = 0.8125 for `apply_fast_resonance` with its real
constants. 30 trials with `kappa` / `delta_eff` / `stress_segnale` fully randomized
(wide ranges, some out of the intended scale) all landed on MRR = 0.8125 too,
std = 0.0000. The modulation correlates with plain cosine at 0.999996 and never once
changed which row ranked first.

Use this for the same job plain cosine similarity does; the modulation is not adding
retrieval value. See
`test_apply_fast_resonance_ranking_e_indistinguibile_da_cosine_puro` in
`test/test_resonance_search.py` for the full numbers.

::: dense_armor.utility.anomaly.resonance_search

---

## Details

The toolkit is deliberately kept separate from the shields. Every module in it can be
removed from the package and the shields would still work: they use only a few of the
utilities internally (`SIGNAL_STABILIZER_PRESETS`, for instance), and those are imported
directly rather than through a "toolkit" interface. The split is for the user: the
shields on the top of the docs tree, the toolkit at the bottom, no confusion about
which is which.

The honest caveats above are not decoration. Each one is a place where a real number
was checked and the module turned out to be narrower than its name, or the effect it
claims to provide turned out to be indistinguishable from a simpler alternative. Keeping
them in the docs is the point: a user who reaches for `StochasticAdversarialNoise` as
if it were a real adversarial benchmark should know it is not, before they write a
paper about it.
