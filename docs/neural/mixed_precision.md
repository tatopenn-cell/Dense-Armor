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

CPU (`benchmarks/mixed_precision_cpu.json`, `backend: cpu`):

| mode | loss at step 20 | steps/s (median) |
|---|---|---|
| float32 | 0.580604 | 122.13 |
| bfloat16 | 0.581430 | 87.23 |
| float16 (dynamic) | 0.595491 | 78.29 |

GPU (T4, `benchmarks/mixed_precision_gpu.json`, `backend: gpu`):

| mode | loss at step 200 | steps/s (median) | peak bytes |
|---|---|---|---|
| float32 | 0.946005 | 3.35 | 2 333 349 632 |
| bfloat16 | 0.945963 | 2.54 | 2 411 058 688 |
| float16 (dynamic) | 0.945946 | 2.39 | 2 476 843 008 |

Claim 1 (bfloat16 within 5% of float32): **holds**. On CPU at step
20 the relative difference is 0.14%; on GPU at step 200 the relative
difference is `4.4e-5`. Claim 2 (memory of the lower precision at most
0.7 times float32): **does not hold**. On GPU the ratio is `1.033`: the
mixed-precision run uses about 3% more memory than float32. Claim 3 (the
faster of bfloat16 and float16 has at least 1.3 times the steps per
second of float32): **does not hold**. On GPU the ratio is `0.759`:
bfloat16 is about 24% slower than float32, not faster.

The reason for claims 2 and 3: the T4 GPU of the run is a Turing card,
which has no bfloat16 tensor core. The `bfloat16` arithmetic is done in
software, and the mixed step adds a copy of the master weights in
float32 plus the float16 gradients in the graph, so the peak memory
grows. The claims are reported as they were written before the run;
the numbers above say they do not hold on this card. On an Ampere
(A100, A10, RTX 30/40) or Hopper (H100) the tensor cores have bfloat16,
and the result may differ; that is not measured here.

### Not measured

GPU speed, GPU peak memory, training at scale, the trajectory of a network
that learns with float16 on a task whose gradient signal is strong enough. The page shows the underflow, the
loss scaling and the bfloat16 range on small models. A demonstration that
float16 training with loss scaling matches float32 on a task where the
network actually learns is **not** part of this page.

### Development diary

A first version of the mixed step captured `loss_scale` as a constant in
the jitted graph, so the scale could not change at runtime and the
dynamic scaling did not work. The fix passes the scale as a runtime
argument of the jitted step, and `Trainer.fit` passes `self._scale` at
every step. The block of Section 2 shows the corrected behaviour.

The underflow exists and the loss scaling fixes it in the gradient
(Section 1). But the loss on these small networks does not decrease
whether the scaling is on or off. The reason is in the master-weight
update, not in the gradient: `lr * grad = 0.01 * 2.7e-9 = 2.7e-11`, and
a float32 weight of `0.9` (or even `0.01`) has a precision of about
`1e-7` (or `1e-9`), so the sum `w - lr * grad` returns `w` unchanged.
The loss scaling makes the gradient survive the float16 representation,
but the update is lost in the float32 master weight before the loss can
move.

A scan of eighteen configurations (hidden layers `8`, `10`, `12`, initial
weight scale `0.1` and `0.01`, `lr` in `0.1`, `0.5`, `1.0`, twenty
epochs) shows the same trajectory with and without a loss scale of
`1024` in all eighteen: the difference between the scaled and the
unscaled final loss is below the print precision. A configuration where
the scale changes the trajectory needs a master weight small enough that
`lr * grad` is above its float32 precision, and on a network whose
gradient is above `2^-24` that weight is below `2e-4`: on these models
the two requirements (underflow in the gradient, visible update in the
weight) do not meet. A network larger than the ones used here, or a task
with a stronger gradient signal, is needed for a demonstration that the
loss scaling makes the training match float32. Reported, not
demonstrated.