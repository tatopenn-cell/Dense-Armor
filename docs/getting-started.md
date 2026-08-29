# Getting Started

## Install

```bash
pip install dense-armor                 # core: numpy + jax
pip install "dense-armor[quantum]"      # + Dense-Evolution (NISQ simulator)
pip install "dense-armor[audio,data]"   # + WAV, HDF5, NetCDF
```

```python
# required before any import -- needed by the 64-bit gating
import jax
jax.config.update("jax_enable_x64", True)
```

```powershell
# PowerShell equivalent, before launch
$env:JAX_ENABLE_X64="True"
```

To run the test suite locally (clone the repo first -- not needed if you only `pip install`ed):

```bash
git clone https://github.com/tatopenn-cell/Dense-Armor.git
cd Dense-Armor && pip install -e ".[dev]"
pytest test/ -v
```

## Quickstart

```powershell
python -m dense_armor --json 1.2 1.3 9999 1.25 nan 1.3
```

```
> anomaly @ index 2 (spike 9999)
> anomaly @ index 4 (NaN)
> everything else: unchanged
```

### Protecting a real model

```python
from dense_armor.utility.orca import Orca

orca = Orca()
protected_output = orca.protect_and_forward(
    my_model,                                  # callable: x -> output (JAX/NumPy)
    corrupted_data,                            # tensor from the sensor/pipeline
    x_reference=known_clean_reference,         # optional but recommended
)

orca.margine_ingresso_medio, orca.margine_uscita_medio   # how much to trust the output
```

### Routing each point to the right corrector (`use_arbiter`)

```python
orca = Orca()
protected_output = orca.protect_and_forward(my_model, corrupted_data, use_arbiter=True)

orca.etichette_arbitro          # 'clean'/'spike'/'regime' array, one per point
orca.incertezza_arbitro_media   # 0..1: how ambiguous the classification itself was
orca.tipi_corruzione_visti(corrupted_data.shape)   # Counter, filled once x_reference is known
```

Off by default. Each point is classified against a wide, causal reference window (only points
before it), then routed: `spike` (isolated impulse) gets hard-rejected to that window's median;
`regime` (a sustained level change, recognized by the length and internal coherence of the
run of anomalous points) passes through raw, fully trusted; `clean` keeps whatever the standard
4-phase shield already produced -- not the raw value, since a genuinely continuous signal still
needs `AdaptiveSignalStabilizer`'s soft damping. Verified on the same 7 scenarios
[`test/testKalman.py`](https://github.com/tatopenn-cell/Dense-Armor/blob/master/test/testKalman.py)
uses: never worse than the default, better on 5/7. See [Arbiter](api/arbiter.md) for the full
design and a real bug found and fixed along the way (symmetric reference window → causal).

### A 1D series (training loss, metrics, token stream)

```python
from dense_armor import Armatura

a = Armatura(livello_ia=0.0)   # 0 = actively filter · 1 = mark only
pulito, K, anomalie = a.analizza(serie)
```

### Standalone robust filters

```python
from dense_armor.utility.robust_filters import pressure_valve

pulito, anomalie, pressione, soglia_effettiva = pressure_valve(serie)
```

No configuration needed beyond the defaults -- `pressure_valve` combines four classic
detectors (Chauvenet, Tukey, Hampel, sigma-clipping) via a minimum-variance estimator and a
Jensen-Shannon-modulated threshold. See [Robust filters](api/robust_filters.md) for the full
math.

### Standalone toolkit

A second, independent part of the package (`core/`/`utility/`) -- none of it participates in
the anomaly shield above. Two examples out of the 14 modules:

```python
from dense_armor.core import DynamicAICodegen

codegen = DynamicAICodegen()
ops = codegen.compile_pipeline(["relu", "l2_normalize"])
out = codegen.run_dynamic_pipeline([-2.0, 3.0, -1.0, 4.0], ops)
# out -> [0. 0.6 0. 0.8]  (relu clips negatives, then L2-normalized)
```

```python
from dense_armor.core import UniversalMemoryGuard

guard = UniversalMemoryGuard(min_free_ram_percentage=0.10)
guard.check_memory_safety()  # raises MemoryPressureError if RAM is too low
```

See [Toolkit](api/toolkit.md) for the full list -- an op-compiler, chunking, hardware
profiling, adversarial noise injection, presets, logging/provenance export, and audio/HDF5/
NetCDF I/O helpers.
