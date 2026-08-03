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
