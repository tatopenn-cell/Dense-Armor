# Mixed precision: how do I train with half the memory?

A float32 parameter is four bytes. Half precision is two bytes: half the
memory, and on a modern GPU half the bandwidth. The catch is that the range
of a float16 is much smaller than float32: the largest finite value is
`65504`, the smallest positive normal value is `2^-14 = 6.10e-5`, and the
smallest positive value of any kind (a subnormal) is `2^-24 = 5.96e-8`. A
gradient smaller than `2^-24` becomes zero, and the parameter stops learning.

## 1. The gradient that becomes zero

A sigmoid MLP with twelve hidden layers and a normalised input has
first-layer gradients of the order of `1e-9`: below `2^-24`. In float16
the first-layer gradient is stored as zero, and the first layer does not
move. The loss scaling multiplies the loss by a large constant before the
backward, so the gradients are `S` times larger and survive the float16
representation; after the backward they are divided by `S` again.

The next block builds the model, takes one step in float16 with scale `1`
and with scale `8192`, and prints the largest magnitude of the
first-layer gradient.

```python
import numpy as np
import jax
import jax.numpy as jnp
from dense_armor.utility.neural.models import mlp
from dense_armor.utility.neural.losses import MSE
from dense_armor.utility.neural.compiled import make_step
from dense_armor.utility.neural.schedules import ConstantLR
from dense_armor.utility.optim import SGD

X = np.random.default_rng(1).normal(size=(16, 2)).astype(np.float32)
y = np.random.default_rng(2).normal(size=(16, 1)).astype(np.float32)

def first_grad(scale):
    m = mlp([2] + [4] * 12 + [1], jax.random.PRNGKey(0),
            activation='sigmoid')
    step = make_step(
        m, MSE(), SGD(0.01), ConstantLR(0.01), 0.0, 'l2', None,
        compute_dtype='float16', loss_scale=scale,
    )
    values = [jnp.array(p.value, dtype=jnp.float32) for p in m.params()]
    grads16, _, _ = step.grads(values, X, y, scale)
    return np.asarray(grads16[0])

g1 = first_grad(1.0)
g8192 = first_grad(8192.0)
print(f'scale 1:    {abs(g1).max():.3e}')
print(f'scale 8192: {abs(g8192).max():.3e}')
```

```
scale 1:    0.000e+00
scale 8192: 2.235e-05
```

With scale `1` every element of the first-layer gradient is exactly zero
in float16. With scale `8192` the largest is `2.2e-5`: the gradient
survives, and after the backward it is divided by `8192` to give back
`2.7e-9`.

## 2. The scale of the loss

The `loss_scale` argument of `Trainer` accepts `None` (no scaling), a
positive float (a constant factor), or `'dynamic'`. It works with
`compute_dtype` or alone, and needs `compile=True`.

- `'dynamic'` starts at `init_scale = 65536`, halves and skips the step on
  an overflow, and doubles the factor after `2000` consecutive steps
  without an overflow.
- the dynamic behaviour is the default of `torch.cuda.amp.GradScaler`
  (PyTorch) and of `apex.amp.GradScaler` (NVIDIA), not a formula of the
  paper. The paper describes a constant factor and a skip on overflow; the
  growth window is our choice, taken from those two libraries.

The next block trains the same small MLP in float32 and in float16 with
the dynamic scale, for five epochs of eight steps, and prints the loss of
each epoch and the scale at the end of each epoch.

```python
import numpy as np
import jax
from dense_armor.utility.neural.models import mlp
from dense_armor.utility.neural.losses import MSE
from dense_armor.utility.neural.trainer import Trainer
from dense_armor.utility.optim import SGD

X = np.random.default_rng(1).normal(size=(64, 2)).astype(np.float32)
y = np.random.default_rng(2).normal(size=(64, 1)).astype(np.float32)

def run(**kw):
    m = mlp([2, 4, 1], jax.random.PRNGKey(0))
    t = Trainer(m, MSE(), SGD(0.01), compile=True, **kw)
    t.fit(X, y, epochs=5, batch_size=8, key=jax.random.PRNGKey(0))
    return t.history_

h32 = run()
h16 = run(compute_dtype='float16', loss_scale='dynamic')
print([round(v, 4) for v in h32['train_loss']])
print([round(v, 4) for v in h16['train_loss']])
print([float(s) for s in h16['loss_scale']])
```

```
[3.8462, 1.9789, 1.4705, 1.2851, 1.1959]
[4.7503, 3.0693, 1.8122, 1.4383, 1.2735]
[4096.0, 4096.0, 4096.0, 4096.0, 4096.0]
```

The float16 run follows the float32 run, a few steps behind. The scale
starts at `65536` and ends the first epoch at `4096`, which is four
halvings: four steps overflowed and were skipped, so the weights did not
move in them and the loss of the first epoch is higher. The scale does
not move again in the next epochs.

## 3. bfloat16: the same range as float32

A bfloat16 has the exponent of float32 (eight bits) and a short mantissa
(seven bits). The range is the same as float32, the precision is lower.
No loss scaling is needed; the gradients do not underflow.

```python
import numpy as np
import jax
from dense_armor.utility.neural.models import mlp
from dense_armor.utility.neural.losses import MSE
from dense_armor.utility.neural.trainer import Trainer
from dense_armor.utility.optim import SGD

X = np.random.default_rng(1).normal(size=(16, 2)).astype(np.float32)
y = np.random.default_rng(2).normal(size=(16, 1)).astype(np.float32)

def run(**kw):
    m = mlp([2, 4, 1], jax.random.PRNGKey(0))
    t = Trainer(m, MSE(), SGD(0.01), compile=True, **kw)
    t.fit(X, y, epochs=4, batch_size=8, key=jax.random.PRNGKey(0))
    return t.history_['train_loss']

print([round(v, 6) for v in run()])
print([round(v, 6) for v in run(compute_dtype='bfloat16')])
```

```
[3.387116, 2.700739, 2.257435, 1.967672]
[3.390043, 2.706822, 2.278326, 1.958916]
```

The two trajectories are within `1%` of each other at every epoch: the
lower precision of bfloat16 does not diverge on this task, and the loss
scaling is not needed.

## Details

### References

- Micikevicius et al. 2017 (arXiv:1710.03740). Section 3.1, page 2: FP32
  master copy of the weights. Section 3.2, page 4: values below `2^-24`
  become zero in FP16, loss scaling. Section 3.2, page 5: overflow skip.
  Section 3.3, page 5: reductions in float32.

### Choices stated explicitly

- **Reductions in float32 (Section 3.3).** The reductions of `LayerNorm`,
  of the attention softmax and of the cross-entropy stay in the compute
  dtype, not in float32. The paper recommends float32 for reductions.
  Not implemented: it needs changes in `norm.py`, `attention.py` and
  `losses.py`.
- **Dynamic scaling.** The paper describes a constant factor and a skip
  on overflow; the growth window of `2000` good steps is the default of
  `torch.cuda.amp.GradScaler` and `apex.amp.GradScaler`. `init_scale =
  65536` and `backoff = 0.5` are the same defaults. The scale never goes
  below `1`.
- **The scale is an argument of the compiled step.** The step multiplies
  the loss gradient by the scale it receives, divides the float32
  gradients by the same scale, and then clips them. The clipping norm is
  therefore the one of the unscaled gradient.
- **`compute_dtype=None` and `loss_scale=None`** keep float32 and the
  existing behaviour: the losses of the default compiled trainer equal
  the values stored from the master branch, and `history_` has the same
  three keys.

### Measured cost

`benchmarks/mixed_precision.py` runs a `transformer_encoder` of 6 layers,
`d_model=512`, sequence 256, batch 32, on synthetic data (seed 0), with
SGD at learning rate `0.01`. Each mode runs in a process of its own,
because the peak memory of a process only grows. The loss is the loss of
step 200 on one batch repeated; the speed is the median of 3 sessions of
20 steps.

TPU: `TPU v5 lite`, jax 0.11.2, `benchmarks/mixed_precision_tpu.json`.

| mode | loss at step 1 | loss at step 200 | steps/s, median (range) | peak bytes |
|---|---|---|---|---|
| float32 | 2.2080 | 0.946706 | 77.20 (76.76 to 77.31) | 2 815 623 680 |
| bfloat16 | 2.2080 | 0.946207 | 18.22 (17.31 to 18.77) | 456 712 704 |
| float16 (dynamic) | 2.2080 | 0.946104 | 18.65 (18.42 to 18.84) | 461 934 080 |

GPU: a Kaggle GPU kernel, `benchmarks/mixed_precision_gpu.json` (the model
of the card is not recorded in that file).

| mode | loss at step 1 | loss at step 200 | steps/s, median (range) | peak bytes |
|---|---|---|---|---|
| float32 | 2.2085 | 0.945989 | 3.386 (3.386 to 3.396) | 2 407 757 056 |
| bfloat16 | 2.2085 | 0.945990 | 2.734 (2.712 to 2.739) | 2 463 308 800 |
| float16 (dynamic) | 2.2084 | 0.945968 | 2.783 (2.782 to 2.790) | 2 463 308 800 |

The claims were written before the first run and are reported unchanged:

| Claim | TPU v5 lite | GPU |
|---|---|---|
| 1. loss of bfloat16 at step 200 within 5% of float32 | **holds**, `5.3e-4` | **holds**, `4.4e-7` |
| 2. peak memory of the lower precision at most 0.7 times float32 | **holds**, ratio `0.162` | **does not hold**, ratio `1.023` |
| 3. the faster of bfloat16 and float16 has at least 1.3 times the steps/s of float32 | **does not hold**, float16, ratio `0.242` | **does not hold**, float16, ratio `0.822` |

In every run the loss falls from about `2.21` to about `0.946`, so claim 1
is tested on a model that trains.

What these numbers do not separate: the float32 mode is the plain compiled
step, one compiled function that includes the optimizer update. The two
lower-precision modes use the mixed step, which is compiled up to the
gradients; the unscale and the optimizer update then run outside the
compiled function, and the overflow flag is read on the host at every
step. The speed and the peak memory of the lower precision therefore
include the cost of that structure, and the effect of the dtype alone is
not isolated. The TPU peak memory ratio of `0.162` is smaller than the
factor 2 that halving the activations would give; the reason is not
measured.

### Not measured

The effect of the dtype alone on speed and memory (a mixed step that
includes the optimizer update in the compiled function), the reason of the
TPU memory ratio, the model of the GPU, the float32 reductions of Section
3.3, training at scale on a natural-language corpus, and a comparison with
a statistic (Wilcoxon, A12): the claims above compare one run per mode.

### Development diary

A first version of the mixed step captured `loss_scale` as a constant in
the jitted graph. The scale then could not change at runtime, and the
dynamic scaling skipped every step: the loss of the first version of
Section 2 stayed at `5.348` for five epochs while a constant scale of `1`
trained like float32. The page explained that result by the range of
float16, which was wrong. The scale is now an argument of the step, and
a test checks that the update of the step that triggers the growth equals
the update of a constant scale of the same value.
