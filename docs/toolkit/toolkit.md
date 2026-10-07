# Toolkit (standalone utilities)

A second part of the package, under `core/` and `utility/`, independent of
[`Armatura`](../shields/armatura.md) and [`Orca`](../shields/orca.md) — none of it
participates in the anomaly shield. Generic tools for JAX / NumPy pipelines.

Every example below was run before being written down. Each module is tested on its
own (`test/test_chunk.py`, `test_compiler.py`, `test_memory.py`, `test_preset.py`,
`test_tensor.py`, `test_noise.py`, `test_vector.py`, `test_profiler.py`,
`test_visualizer.py`, `test_logger.py`, `test_anwav.py`, `test_diagnostic.py`,
`test_iodat.py`, `test_resonance_search.py`).

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
print(out)
```

```
[0.  0.6 0.  0.8]
```

Relu clips the negatives, then L2-normalize.

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
print([list(c) for c in chunks], list(merged))
```

```
[[0, 1], [2, 3], [4]] [0, 1, 2, 3, 4]
```

::: dense_armor.core.chunk

## Memory guard

**`UniversalMemoryGuard`** checks free RAM (and VRAM, if an NVIDIA GPU is present)
before a heavy allocation, and computes how many chunks a batch needs to fit safely.

```python
from dense_armor.core import UniversalMemoryGuard

guard = UniversalMemoryGuard(min_free_ram_percentage=0.10)
guard.check_memory_safety()
print("ok")
```

```
ok
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

```
Processor: x86_64 | RAM: 12.7 GB | Engine: CPU (JAX Accelerato) | SafeMaxDim: 40
```

**Honest caveat**: the RAM tiers behind `max_tensor_dim` are a rough heuristic, not
calibrated against anything specific to this package. The `SafeMaxDim` value is
host-dependent — on a small Colab instance it lands in the tens, on a large server it
is in the thousands. Treat it as a starting guess, not a guarantee.

**`StochasticAdversarialNoise`** injects synthetic noise (bitflip, dropout, Gaussian
blur) into a tensor while preserving its norm.

```python
import numpy as np
from dense_armor.core import StochasticAdversarialNoise

out = StochasticAdversarialNoise.inject_noise(
    np.array([1.0, 1.0, 1.0, 1.0]), "bitflip", intensity=1.0, seed=0,
)
print(out)
```

```
[-0.5 -0.5 -0.5 -0.5]
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
print(sorted(stats.keys()))
```

```
['mean_execution_us', 'min_execution_us', 'repetitions', 'std_execution_us', 'warmup_compilation_us']
```

The five keys are: warm-up (the first, compiling call), mean / min / std of the
steady-state calls, and the number of repetitions.

::: dense_armor.core.profiler

## Tensors and configuration

**`TensorVault`** is a small library of static (`invert`, `identity`, `edge_detector`,
`blend`) and parametric (`scale_project`, `amplify`, `bias_shift`) transformation
matrices, with backend (JAX / NumPy) and precision auto-detected.

```python
from dense_armor.core import TensorVault

vault = TensorVault()
edge = vault.get_static_transform("edge_detector")
print(edge)
```

```
[-1.  2. -1.]
```

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
print(result, collapsed.shape)
```

```
1 (4,)
```

**Honest caveat**: the per-step update is a fixed exponential-moving-average
weighting, not a configurable simulation model. Useful mainly if that specific dynamic
matches your scenario, not as a general-purpose simulator.

**`BitwisePermutationEngine`** swaps elements of a combinatorial vector (a
`2^n`-sized space) based on target / control bit masks.

```python
import numpy as np
from dense_armor.core import BitwisePermutationEngine

engine = BitwisePermutationEngine(n_elements=2)
out = engine.apply_bitwise_swap(np.array([0., 1., 2., 3.]), target_bit=1, control_bit=0)
print(out)
```

```
[0. 1. 3. 2.]
```

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

print(sorted(SIGNAL_STABILIZER_PRESETS.keys()))
```

```
['balanced_v2', 'cifar10_best_v1', 'cifar10_hardened_lyapunov', 'pure_1d_time_v1']
```

The presets are not just different numbers that happen to look distinct. On the same
100 Hz series with an outlier spike at sample 1500, the residual variance left by
each preset differs by roughly a factor of two:

```python
import numpy as np
from dense_armor.core.preset import SIGNAL_STABILIZER_PRESETS
from dense_armor.core.engine import AdaptiveSignalStabilizer

x = 0.5 + 0.1 * np.random.default_rng(42).standard_normal(3000)
x[1500] = 2.5
x = np.asarray(x)

s_bal = AdaptiveSignalStabilizer(**SIGNAL_STABILIZER_PRESETS["balanced_v2"])
s_rea = AdaptiveSignalStabilizer(**SIGNAL_STABILIZER_PRESETS["pure_1d_time_v1"])
print(round(float(np.var(x - s_bal.filter_data_stream(x))), 6),
      round(float(np.var(x - s_rea.filter_data_stream(x))), 6))
```

```
0.006135 0.003074
```

`balanced_v2` leaves about twice the residual variance of `pure_1d_time_v1`: the
`pure_1d_time_v1` regime tracks the signal more closely, and pays for it with a
noisier output. The two presets really do configure different filtering behaviour.

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
[19:33:28] [INFO] esempio
```

**Honest caveat**: fairly thin wrappers around `logging.Formatter` —
`CompactJsonFormatter`'s structured fields (module / filename / line number, one JSON
object per event) are the main reason to reach for this over writing a one-line
formatter yourself.

::: dense_armor.core.logger

---

**`AIEngineVisualizer`** exports a SHA-256-signed provenance archive (parameters,
execution environment, integrity hash) and plain-text trend reports comparing raw vs.
filtered variance.

```python
from dense_armor.core import AIEngineVisualizer

viz = AIEngineVisualizer(output_dir=".")
sha256 = viz.export_provenance_archive([{"step": 1, "value": 0.5}], filename="archive.json")
print(len(sha256), sha256[:12])
```

```
64 fca2a89de7ae
```

A 64-hex-character SHA-256 that matches the hash written into `archive.json`. The
first twelve characters are shown; the rest is the same hash continued.

::: dense_armor.core.visualizer

## Audio and data I/O

**`diag(iorig, ifilt)`** compares two signals (file paths or NumPy arrays):
structural fidelity, removed energy, distortion peak, and the fraction of samples that
were modified.

```python
import numpy as np
from dense_armor.utility.misc.diagnostic import diag

rng = np.random.default_rng(0)
orig = rng.normal(size=2000).astype(np.float32)
filt = orig * 0.98
result = diag(orig, filt)
print(sorted(result.keys()))
print(result["fedelta"])
```

```
['energia_rimossa', 'fedelta', 'picco_distorsione_db', 'tasso_modulazione']
99.96
```

`fedelta` is the structural fidelity in percent: 99.96 % here, because the filter only
scaled the signal by 0.98. `energia_rimossa`, `picco_distorsione_db` and
`tasso_modulazione` give the removed energy, the peak of the removed component in
dBFS, and the fraction of samples touched.

::: dense_armor.utility.misc.diagnostic

---

**`lodat(fpath, dname)`** reads a named tensor out of an HDF5 or NetCDF file. Useful
as a thin, uniform loader when a pipeline needs to accept either format without
branching on the caller's side.

```python
import numpy as np, h5py
from dense_armor.utility.misc.iodat import lodat

with h5py.File("data.h5", "w") as f:
    f.create_dataset("temperature", data=np.arange(12).reshape(3, 4))
tensore = lodat("data.h5", "temperature")
print(tensore.shape)
```

```
(3, 4)
```

Requires `pip install dense-armor[data]` for the HDF5 / NetCDF readers.

::: dense_armor.utility.misc.iodat

## Similarity search

**`apply_fast_resonance(matrix, query)`** scores similarity between a query vector
and each row of a matrix, modulated by the same damping operator Orca's gating uses.

```python
import numpy as np
from dense_armor.utility.anomaly.resonance_search import apply_fast_resonance

rng = np.random.default_rng(0)
db = rng.standard_normal((5, 8)).astype(np.float32)
query = db[2].copy()
scores = apply_fast_resonance(db, query)
print(int(scores.argmax()))
```

```
2
```

The query is an exact copy of row 2, and row 2 scores highest.

### An honest finding: the modulation does not change ranking

The three modulation parameters (`kappa`, `delta_eff`, `stress_segnale`) do
measurably change the score values. On the same input above, the raw scores are:

```
[0.8168, 0.8055, 1.0006, 0.7561, 0.8100]
```

But for retrieval, what matters is *ranking*, not the absolute score, and there the
modulation is a **confound, not a real effect**. A real benchmark on quantumrag
(1855 chunks, 12 labelled queries, Mean Reciprocal Rank) gave MRR = 0.8125 for plain
cosine, and the identical MRR = 0.8125 for `apply_fast_resonance` with its real
constants. 30 trials with the three parameters fully randomized (wide ranges, some
out of the intended scale) all landed on MRR = 0.8125 too, std = 0.0000. The
modulation correlates with plain cosine at 0.999996 and never once changed which row
ranked first.

Use this for the same job plain cosine similarity does; the modulation is not adding
retrieval value. See
`test_apply_fast_resonance_ranking_e_indistinguibile_da_cosine_puro` in
`test/test_resonance_search.py` for the full numbers.

::: dense_armor.utility.anomaly.resonance_search

---

## Details

The toolkit is deliberately kept separate from the shields. Every module in it can be
removed from the package and the shields would still work: they use only a few of the
utilities internally (`SIGNAL_STABILIZER_PRESETS`, for instance), and those are
imported directly rather than through a "toolkit" interface. The split is for the
user: the shields on the top of the docs tree, the toolkit at the bottom, no
confusion about which is which.

The honest caveats above are not decoration. Each one is a place where a real number
was checked and the module turned out to be narrower than its name, or the effect it
claims to provide turned out to be indistinguishable from a simpler alternative.
Keeping them in the docs is the point: a user who reaches for
`StochasticAdversarialNoise` as if it were a real adversarial benchmark should know
it is not, before they write a paper about it.
