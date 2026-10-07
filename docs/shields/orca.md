# Orca (full model shield)

`Armatura` protects one series. But if you have a model — any callable that takes a
tensor and returns a tensor — the corruption can happen twice: once on the way in
(garbage input) and once on the way out (the model itself producing a wrong response).
`Orca` shields both sides in one call: it purifies the input, runs the model, and
checks that the response itself isn't corrupted.

## The model and the input

A small model, the identity times two, and a 4×200 batch of sine waves with one absurd value at
`(0, 50)`; `ref` is a clean reference of the same shape.

## 1. Protect the forward pass

```python
import numpy as np
from dense_armor.utility.protect.orca import Orca

rng = np.random.default_rng(0)
ref = np.sin(np.linspace(0, 6, 200))[None, :] + 0.01 * rng.standard_normal((4, 200))
data = ref.copy()
data[0, 50] = 999.0
orca = Orca(min_free_ram_percentage=0.05)
out = orca.protect_and_forward(lambda z: 2.0 * z, data, x_reference=ref)
print(out.shape, round(float(out[0, 50]), 4), round(float(2 * ref[0, 50]), 4))
```

```
(4, 200) 2.0074 2.0031
```

`out` is the model's output on the cleaned input. The spike at `(0, 50)` never reaches the model:
the output there is 2.0074, next to twice the clean value (2.0031), instead of 1998.
`min_free_ram_percentage` lowers Orca's memory guard (default: 15 % free RAM required).

The four phases of `protect_and_forward`, in order:

1. **Input shield** — purify `x_corrupted` against `x_reference` (`use_input_shield`).
2. **Model call** — run the callable on the purified tensor (`use_model_injection`).
3. **Output shield** — check the model's response against its expected response to the
   clean reference (`use_output_shield`).
4. **Report** — return the protected output.

Every phase can be disabled individually: `use_input_shield=False`,
`use_model_injection=False`, `use_output_shield=False`. All three default to `True`.

## 2. What each argument does

- `ai_model_callable` — any callable `z -> z_out`. In the example it is the identity
  scaled by two, chosen so the shield is the only thing acting on the tensor.
- `x_corrupted` — the tensor that would be passed to the model. It is cleaned before
  the model sees it.
- `x_reference` — optional but recommended. A clean tensor of the same shape as
  `x_corrupted`. When present, the input shield compares each sample against the
  corresponding reference sample instead of falling back to a robust blind estimate.
- `use_arbiter=True` — routes each point through the per-point classifier described in
  [Arbiter](../protect/arbiter.md): clean passes through whatever the standard shield
  produced, spike gets hard-rejected to the window median, regime passes through raw.
  Off by default.

## 3. Where the two stages come from

`Orca`'s pipeline has two stages:

1. **Stage 1 — `AdaptiveSignalStabilizer`**: a causal recursive filter with a sigmoid
   damping curve. This is the [adaptive engine](engine.md). It does the actual
   purification of the input and the response.
2. **Stage 2 — Collatz-based gate**: a decision on how much to damp toward the clean
   reference. This is where `Orca` chooses between "reject the sample" and "trust it".

With `use_arbiter=True`, a third layer sits between Stage 1 and Stage 2 and decides,
point by point, which of the two Stage-1 behaviours to use.

## 4. What the shield saw, after the fact

```python
import numpy as np
from dense_armor.utility.protect.orca import Orca

rng = np.random.default_rng(0)
ref = np.sin(np.linspace(0, 6, 200))[None, :] + 0.01 * rng.standard_normal((4, 200))
data = ref.copy()
data[0, 50] = 999.0
orca = Orca(min_free_ram_percentage=0.05)
orca.protect_and_forward(lambda z: 2.0 * z, data, x_reference=ref, use_arbiter=True)
print(orca.tipi_corruzione_visti(data.shape[1:]))
```

```
Counter({'clean': 721, 'regime': 75, 'spike': 4})
```

`tipi_corruzione_visti(row_shape)` returns how many times each Arbiter label was seen for rows of
that shape (here `(200,)`), counted only when `use_arbiter=True` and `x_reference` is given. Over the
4 × 200 samples: 721 clean, 75 regime (the sine's own slow changes of level), 4 spikes.

## API reference

::: dense_armor.utility.protect.orca

---

**See also**: [Adaptive engine](engine.md) — `AdaptiveSignalStabilizer`, Orca's Stage 1.
[Arbiter](../protect/arbiter.md) — the per-point classifier used by
`use_arbiter=True`. [Armatura](armatura.md) — if you have a bare series rather than a
model.
