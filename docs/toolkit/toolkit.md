# Toolkit (standalone utilities)

A set of general-purpose helpers for fast number-crunching programs, usable on their own. This page shows each one with an example.

A second part of the package, under `core/` and `utility/`, independent of
[`Armor`](../shields/armor.md) and [`Orca`](../shields/orca.md) — none of it
participates in the anomaly shield. Generic tools for JAX / NumPy pipelines.

Every example below was run before being written down. Each module is tested on its
own (`test_memory.py`, `test_preset.py`, `test_noise.py`, `test_diagnostic.py`,
`test_iodat.py`, `test_resonance_search.py`).

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
Processor: x86_64 | RAM: 12.7 GB | Engine: CPU (JAX Accelerated) | SafeMaxDim: 40
```

**Honest caveat**: the RAM tiers behind `max_tensor_dim` are a rough heuristic, not
calibrated against anything specific to this package. The `SafeMaxDim` value is
host-dependent — on a small Colab instance it lands in the tens, on a large server it
is in the thousands. Treat it as a starting guess, not a guarantee.

::: dense_armor.core.noise

## Configuration

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

## Audio and data I/O

**`diag(iorig, ifilt, threshold=0.05)`** compares two signals (file paths or
NumPy arrays): structural fidelity, removed energy, distortion peak, the fraction of
samples that were modified, and a one-word verdict.

```python
import numpy as np
from dense_armor.utility.infrastructure.misc.diagnostic import diag

rng = np.random.default_rng(0)
orig = rng.normal(size=2000).astype(np.float32)
filt = orig * 0.98
result = diag(orig, filt)
print(sorted(result.keys()))
print(result["fidelity"], result["verdict"])
```

```
['fidelity', 'modulation_rate', 'peak_distortion_db', 'removed_energy', 'verdict']
99.96 micro_edits
```

`fidelity` is the structural fidelity in percent: 99.96 % here, because the filter
only scaled the signal by 0.98. `removed_energy`, `peak_distortion_db` and
`modulation_rate` give the removed energy (variance of the difference, computed with
`RunningMoments`), the peak of the removed component in dBFS, and the fraction of
samples touched. `verdict` summarises the fidelity: `micro_edits` above 99.5,
`balanced` between 95 and 99.5, `aggressive` below 95.

::: dense_armor.utility.infrastructure.misc.diagnostic

---

**`lodat(fpath, dname)`** reads a named tensor out of an HDF5 or NetCDF file. Useful
as a thin, uniform loader when a pipeline needs to accept either format without
branching on the caller's side.

```python
import numpy as np, h5py
from dense_armor.utility.infrastructure.misc.iodat import lodat

with h5py.File("data.h5", "w") as f:
    f.create_dataset("temperature", data=np.arange(12).reshape(3, 4))
tensore = lodat("data.h5", "temperature")
print(tensore.shape)
```

```
(3, 4)
```

Requires `pip install dense-armor[data]` for the HDF5 / NetCDF readers.

::: dense_armor.utility.infrastructure.misc.iodat

## Similarity search

**`apply_fast_resonance(matrix, query)`** scores similarity between a query vector
and each row of a matrix, modulated by the same damping operator Orca's gating uses.

```python
import numpy as np
from dense_armor.utility.online.anomaly.resonance_search import apply_fast_resonance

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

::: dense_armor.utility.online.anomaly.resonance_search

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
