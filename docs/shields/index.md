# Shields

A **shield** is the shortest path from "I have a signal" to "I have a clean signal". It
sits between the data and whoever uses it, and it cleans what passes through. You give
it a series or a model; you get back a cleaned series or a protected prediction.

There are two shields, because there are two things a user can have in front of them.

## A single series: `Armatura`

You have one array: a loss curve that jumps, a sensor channel that drops a reading, a
token-likelihood stream that goes to NaN. `Armatura` looks at it point by point and
returns the same series with the broken samples replaced by the local baseline. No
intermediate state, no model, no reference needed.

```python
from dense_armor import Armatura

clean, K, anomalies = Armatura(livello_ia=0.0).analizza([1.2, 1.3, 9999, 1.25, float("nan"), 1.3])
print(clean.round(3), anomalies)
```

```
[1.2  1.3  1.25 1.25 1.3  1.3 ] [2, 4]
```

The 9999 at index 2 and the NaN at index 4 are replaced (by 1.25 and 1.3); `anomalies` lists
their indices.

See **[Armatura](armatura.md)**.

## A whole model: `Orca`

You have a model (any callable from a tensor to a tensor) and a corrupted input. `Orca`
purifies the input, runs the model, and checks that the response itself isn't corrupted
before it hands it back. Optional `use_arbiter=True` classifies each point as clean /
spike / regime and routes it to the right corrector instead of one gate for the whole
signal.

```python
import numpy as np
from dense_armor.utility.protect.orca import Orca

rng = np.random.default_rng(0)
ref = np.sin(np.linspace(0, 6, 200))[None, :] + 0.01 * rng.standard_normal((4, 200))
data = ref.copy()
data[0, 50] = 999.0

orca = Orca(min_free_ram_percentage=0.05)
protected = orca.protect_and_forward(lambda z: 2.0 * z, data, x_reference=ref)
print(protected.shape)
```

```
(4, 200)
```

The input to the model is the cleaned version of `data`: the value 999 at `(0, 50)` is
replaced before the model ever sees it. The model itself is the simplest possible
callable (identity times two), chosen here so the shield is the only thing acting.

See **[Orca](orca.md)**.

## The two engines inside

- **[Hybrid engine](hybrid_engine.md)** — the engine behind `Armatura`. Generalises the
  logic already verified in `Dense-Evolution`'s own `ia_utils.vector_healing`.
- **[Adaptive engine](engine.md)** — `AdaptiveSignalStabilizer`, the causal recursive
  filter that is `Orca`'s Stage 1. `Orca` adds a Collatz-based gate (Stage 2) on top.

Both are also usable on their own, but the two shields above are the recommended way in.
