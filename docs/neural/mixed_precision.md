# Mixed precision: how do I train with half the memory?

A float32 parameter is four bytes. Half precision is two bytes: half the
memory, and on a modern GPU half the bandwidth. The catch is that the range
of a float16 is much smaller than float32: the largest finite value is
`65504` and the smallest positive normal value is `2^-24 = 5.96e-8`. A
gradient smaller than that becomes zero, and the parameter stops learning.

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

~~~python
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
    grads, _ = step.mixed(values, X, y)
    return np.asarray(grads[0])

g1 = first_grad(1.0)
g8192 = first_grad(8192.0)
print(f'scale 1:    {abs(g1).max():.3e}')
print(f'scale 8192: {abs(g8192).max():.3e}')
~~~

~~~
scale 1:    0.000e+00
scale 8192: 2.235e-05
~~~

With scale `1` every element of the first-layer gradient is exactly zero
in float16. With scale `8192` the largest is `2.2e-5`: the gradient
survives, and after the backward it is divided by `8192` to give back
`2.7e-9`.

## 2. The scale of the loss

The `loss_scale` argument of `Trainer` accepts `None` (no scaling), a
positive float (a constant factor), or `'dynamic'`:

- `'dynamic'` starts at `init_scale = 65536`, halves and skips the step on
  an overflow, and doubles the factor after `2000` consecutive steps
  without an overflow.
- the dynamic behaviour is the default of `torch.cuda.amp.GradScaler`
  (PyTorch) and of `apex.amp.GradScaler` (NVIDIA), not a formula of the
  paper. The paper describes a constant factor and a skip on overflow; the
  growth window is our choice, taken from those two libraries.

~~~python
import numpy as np
import jax
from dense_armor.utility.neural.models import mlp
from dense_armor.utility.neural.losses import MSE
from dense_armor.utility.neural.trainer import Trainer
from dense_armor.utility.optim import SGD

X = np.random.default_rng(1).normal(size=(64, 2)).astype(np.float32)
y = np.random.default_rng(2).normal(size=(64, 1)).astype(np.float32)
m = mlp([2, 4, 1], jax.random.PRNGKey(0))
t = Trainer(m, MSE(), SGD(0.01), compile=True,
            compute_dtype='float16', loss_scale='dynamic')
t.fit(X, y, epochs=5, batch_size=8, key=jax.random.PRNGKey(0))
print([round(v, 4) for v in t.history_['train_loss']])
print([float(s) for s in t.history_['loss_scale']])
~~~

~~~
[5.348, 5.348, 5.348, 5.348, 5.348]
[256.0, 1.0, 1.0, 1.0, 1.0]
~~~

The loss stays at `5.348`: every step overflows, the scale halves from
`65536` to `1` in two steps, and from `1` it cannot halve further. A scale
of `1` is no scaling: this MLP, with these weights and this input, is in
the wrong range for float16. The history shows the halving; it does not
show a training that succeeds.

## 3. bfloat16: the same range as float32

A bfloat16 has the exponent of float32 (eight bits) and a short mantissa
(seven bits). The range is the same as float32, the precision is lower.
No loss scaling is needed; the gradients do not underflow.

~~~python
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
~~~

~~~
[3.387116, 2.700739, 2.257435, 1.967672]
[3.390043, 2.706822, 2.278326, 1.958916]
~~~

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
  **Reported, not done**: the fix needs changes in `norm.py`,
  `attention.py`, `losses.py`, not in item A of N4.
- **Dynamic scaling.** The paper describes a constant factor and a skip
  on overflow; the growth window of `2000` good steps is the default of
  `torch.cuda.amp.GradScaler` and `apex.amp.GradScaler`. `init_scale =
  65536` and `backoff = 0.5` are the same defaults.
- **`compute_dtype=None`** keeps float32 and the existing behaviour bit
  for bit.

### Measured cost

`benchmarks/mixed_precision.py` runs a transformer encoder of 6 layers,
`d_model=512`, sequence 256, batch 32, in three modes on the CPU of this
session. Values in `benchmarks/mixed_precision_cpu.json`, `backend: 'cpu'`.

| mode | loss at step 20 | steps/s (median) |
|---|---|---|
| float32 | 0.580604 | 87.63 |
| bfloat16 | 0.581430 | 88.18 |
| float16 (dynamic) | 2.466645 | 112.53 |

Claim 1 (bfloat16 within 5% of float32): **holds**, relative difference
0.14%. Claim 2 (memory 0.7x of float32): **not measured**, no GPU. Claim
3 (faster of bfloat16/float16 has 1.3x steps/s of float32): **not
measured** on CPU.

### Not measured

GPU speed, GPU peak memory, training at scale, the trajectory of a network
that learns with float16 in a real task. The page shows the underflow, the
loss scaling and the bfloat16 range on small models. A demonstration that
float16 training with loss scaling matches float32 on a task where the
network actually learns is **not** part of this page.

### Development diary

The underflow is real and the loss scaling fixes it in the gradient
(Section 1). But the loss on these small networks does not decrease
whether the scaling is on or off: the update `lr * grad = 0.01 * 2.7e-9
= 2.7e-11` is too small to move a float32 master weight, even after the
gradient survives. A network larger than the ones used here, or a task
with a stronger gradient signal, is needed for that claim.