# Orca (full model shield)

`Armatura` protects one series. But if you have a model — any callable that takes a
tensor and returns a tensor — the corruption can happen twice: once on the way in
(garbage input) and once on the way out (the model itself producing a wrong response).
`Orca` shields both sides in one call: it purifies the input, runs the model, and
checks the response isn't itself corrupted.

## The model and the input

A small model: identity plus a scaling. A corrupted input: the same 100 Hz signal as on
the previous page, with a spike at 100 and a NaN at 200.

```python
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)

rng = np.random.default_rng(42)
x = 0.5 + 0.1 * rng.standard_normal(300)
x[100] = 9999.0
x[200] = float("nan")

def my_model(z):
    return z * 2.0
```

## 1. Protect the forward pass

```python
from dense_armor.protect.orca import Orca

orca = Orca()
protected = orca.protect_and_forward(my_model, x)
```

`protected` is the model's output on the cleaned input: 300 values around 1.0, with the
two broken samples replaced. The model itself never saw the spike or the NaN, so it
never had a chance to blow up.

## 2. Trust scores

```python
orca.margine_ingresso_medio, orca.margine_uscita_medio
```

Two numbers in `[0, 1]`. `margine_ingresso_medio` says how clean the input was;
`margine_uscita_medio` says how clean the output came back. Near 1 means "trust it";
near 0 means "the shield found a lot of anomalies on this side".

On the running example, both scores are high, because only 2 out of 300 samples were
touched on each side.

## 3. With a reference

If you have a clean reference (the previous reading, a nominal trajectory, a training
sample), pass it in. The input shield compares the current sample to the reference
instead of falling back to a robust blind estimate.

```python
ref = 0.5 + 0.1 * rng.standard_normal(300)
protected = orca.protect_and_forward(my_model, x, x_reference=ref)
```

The difference is subtle on this example (both work), but it matters when the signal is
genuinely drifting: the shield then knows that a slow change is intended and does not
flag it.

## 4. Route each point to the right corrector

By default, `Orca` applies one gate to the whole signal. With `use_arbiter=True`, each
point is classified — clean, spike, or regime — and routed to the correct corrector.

```python
orca = Orca()
protected = orca.protect_and_forward(my_model, x, use_arbiter=True)

orca.etichette_arbitro
orca.incertezza_arbitro_media
```

`etichette_arbitro` is a per-sample array of `"clean"`, `"spike"` or `"regime"`. On the
running example, indices 100 and 200 are `"spike"` (isolated and non-finite), the rest
are `"clean"`. `incertezza_arbitro_media` is a single number in `[0, 1]`: how ambiguous
the classification itself was, averaged over the series. Near 0 means "the Arbiter was
sure on every sample".

The difference between `clean` and `regime` matters when the signal is supposed to
change: a `spike` is replaced by the local median, a `regime` passes through raw
(followed). This is the reason the Arbiter exists — a purely local check cannot tell
"bad sample" from "intended change", but the length of the run of deviating samples can.

## 5. Where the two stages come from

`Orca`'s pipeline has two stages:

1. **Stage 1 — `AdaptiveSignalStabilizer`**: a causal recursive filter with a sigmoid
   damping curve. This is the [adaptive engine](engine.md). It does the actual
   purification of the input and the response.
2. **Stage 2 — Collatz-based gate**: a decision on how much to damp toward the clean
   reference. This is where `Orca` chooses between "reject the sample" and "trust it".

With `use_arbiter=True`, a third layer sits between Stage 1 and Stage 2 and decides,
point by point, which of the two Stage-1 behaviours to use.

## API reference

::: dense_armor.protect.orca

---

**See also**: [Adaptive engine](engine.md) — `AdaptiveSignalStabilizer`, Orca's Stage 1.
[Arbiter](../protect/arbiter.md) — the per-point classifier used by
`use_arbiter=True`. [Armatura](armatura.md) — if you have a bare series rather than a
model.
