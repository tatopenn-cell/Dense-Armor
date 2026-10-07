# Getting Started

## Install

```bash
pip install dense-armor
pip install "dense-armor[quantum]"
pip install "dense-armor[audio,data]"
```

`[quantum]` adds the Dense-Evolution NISQ simulator; `[audio,data]` adds WAV, HDF5 and
NetCDF readers.

```python
import jax
jax.config.update("jax_enable_x64", True)
```

The JAX configuration must run before any other `dense_armor` import: it enables 64-bit
floats, which the shields need.

To run the test suite locally (clone the repository first, not needed if you only
`pip install`ed):

```bash
git clone https://github.com/tatopenn-cell/Dense-Armor.git
cd Dense-Armor && pip install -e ".[dev]"
pytest test/ -v
```

## Quickstart

```bash
python -m dense_armor --json 1.2 1.3 9999 1.25 nan 1.3
```

```
> anomaly @ index 2 (spike 9999)
> anomaly @ index 4 (NaN)
> everything else: unchanged
```

## Protect a real model

```python
from dense_armor.protect.orca import Orca

import numpy as np
rng = np.random.default_rng(0)
ref = np.sin(np.linspace(0, 6, 200))[None, :] + 0.01 * rng.standard_normal((4, 200))
data = ref.copy()
data[0, 50] = 999.0
orca = Orca(min_free_ram_percentage=0.05)
protected = orca.protect_and_forward(lambda z: 2.0 * z, data, x_reference=ref)

orca.margine_ingresso_medio, orca.margine_uscita_medio
```

`my_model` is any callable from a JAX / NumPy tensor to a tensor. `x_reference` is
optional but recommended: if a clean reference signal is available (the previous reading,
a nominal trajectory), the input shield uses it; otherwise it falls back to a robust
blind estimate.

`orca.margine_ingresso_medio` and `orca.margine_uscita_medio` are two numbers in
`[0, 1]` saying how much to trust the input and the output. Near 1 means clean; near 0
means the shield found a lot of anomalies.

## Route each point to the right corrector

```python
orca = Orca(min_free_ram_percentage=0.05)
protected = orca.protect_and_forward(lambda z: 2.0 * z, data, use_arbiter=True)

orca.etichette_arbitro
orca.incertezza_arbitro_media
```

Each point is classified as `clean`, `spike` or `regime` against a causal reference
window (only points before it):

- `spike` (isolated impulse) — hard-rejected to the window median.
- `regime` (sustained level change) — passed through raw, fully trusted.
- `clean` — whatever the standard 4-phase shield already produced (not the raw value; a
  genuinely continuous signal still needs the adaptive stabilizer's soft damping).

Off by default. Verified on the same 7 scenarios `test/testKalman.py` uses: never worse
than the default, better on 5/7.

## A 1D series (training loss, metrics, token stream)

```python
from dense_armor import Armatura

a = Armatura(livello_ia=0.0)
pulito, K, anomalie = a.analizza(serie)
```

`livello_ia = 0` actively filters; `livello_ia = 1` only marks.

## Standalone robust filters

```python
from dense_armor.anomaly.robust_filters import pressure_valve

pulito, anomalie, pressione, soglia_effettiva = pressure_valve(serie)
```

`pressure_valve` combines four classic detectors (Chauvenet, Tukey, Hampel, sigma
clipping) via a minimum-variance estimator and a Jensen–Shannon-modulated threshold.
See [Robust filters](anomaly/robust_filters.md) for the full math.

## Standalone toolkit

A second, independent part of the package. Two examples out of 14 modules:

```python
from dense_armor.core import DynamicAICodegen

codegen = DynamicAICodegen()
ops = codegen.compile_pipeline(["relu", "l2_normalize"])
out = codegen.run_dynamic_pipeline([-2.0, 3.0, -1.0, 4.0], ops)
```

`out` is `[0., 0.6, 0., 0.8]`: relu clips negatives, then L2-normalize.

```python
from dense_armor.core import UniversalMemoryGuard

guard = UniversalMemoryGuard(min_free_ram_percentage=0.10)
guard.check_memory_safety()
```

Raises `MemoryPressureError` if free RAM drops below 10 %.

See [Toolkit](toolkit/toolkit.md) for the full list.
