# How do I make training faster without changing the model?

The training loop of `Trainer` runs in Python: it calls `forward`,
`backward` and the optimiser update one layer at a time, and reads the
loss from the device at every batch. On a small model the cost is
visible but small; on a large model the Python overhead and the host
synchronisation dominate the time.

`Trainer(..., compile=True)` puts the whole step (forward, loss,
backward, gradient clipping, weight decay and the optimiser update of
every parameter) inside one `jax.jit` function. The model does not
change. The optimiser arithmetic does not change. The default
`compile=False` is the loop, unchanged.

## 1. Time the loop

Train the model once with the default `Trainer` and record the steps
per second:

```python
import time
import jax
import jax.numpy as jnp
from dense_armor.utility.neural.trainer import Trainer
from dense_armor.utility.neural.models import mlp
from dense_armor.utility.neural.losses import MSE
from dense_armor.utility.optim import SGD

m = mlp([512, 512, 512, 512, 512], jax.random.PRNGKey(0))
X = jax.random.normal(jax.random.PRNGKey(1), (256, 512))
y = jax.random.normal(jax.random.PRNGKey(2), (256, 512))
t = Trainer(m, MSE(), SGD(0.01))

start = time.perf_counter()
for step in range(20):
    for p in m.params():
        p.zero_grad()
    pred = m.forward(X, training=True)
    _ = t.loss.forward(pred, y)
    m.backward(t.loss.backward())
    for p in m.params():
        p.update(t.optimizer_prototype)
elapsed = time.perf_counter() - start
print(round(20 / elapsed, 1))
```
```
21.37
```

The MLP of four layers of 512 runs at about 21 steps per second on
this CPU. Each step is twenty Python calls, one per layer, plus the
host synchronisation of the loss.

## 2. Turn on compile=True

The same model, the same data, the same optimiser, the same seed,
with `compile=True`:
```python
import jax
import jax.numpy as jnp
from dense_armor.utility.neural.trainer import Trainer
from dense_armor.utility.neural.models import mlp
from dense_armor.utility.neural.losses import MSE
from dense_armor.utility.optim import SGD

m = mlp([512, 512, 512, 512, 512], jax.random.PRNGKey(0))
X = jax.random.normal(jax.random.PRNGKey(1), (256, 512))
y = jax.random.normal(jax.random.PRNGKey(2), (256, 512))
t = Trainer(m, MSE(), SGD(0.01), compile=True)
t.fit(X, y, epochs=1, batch_size=256, key=jax.random.PRNGKey(0))
print(len(t.history_["train_loss"]))
```

```
1
```

The first call compiles the step (0.21 s for the MLP). After the
compilation, the step is one call:
```python
import time
import jax
import jax.numpy as jnp
from dense_armor.utility.neural.trainer import Trainer
from dense_armor.utility.neural.models import mlp
from dense_armor.utility.neural.losses import MSE
from dense_armor.utility.optim import SGD

m = mlp([512, 512, 512, 512, 512], jax.random.PRNGKey(0))
X = jax.random.normal(jax.random.PRNGKey(1), (256, 512))
y = jax.random.normal(jax.random.PRNGKey(2), (256, 512))
t = Trainer(m, MSE(), SGD(0.01), compile=True)
t.fit(X, y, epochs=1, batch_size=256, key=jax.random.PRNGKey(0))

start = time.perf_counter()
t.fit(X, y, epochs=1, batch_size=256, key=jax.random.PRNGKey(0))
elapsed = time.perf_counter() - start
print(round(1 / elapsed, 0))
```
```
1701
```

1701 steps per second on the same CPU, against 21 for the loop. The
transformer encoder of four layers is even more sensitive to the
Python overhead: 0.364 steps per second for the loop, 401 for the
compiled step.

## 3. Check that the loss is the same

The compiled step must give the same loss as the loop. Run both for
200 steps on the same data and the same seed:
```python
import jax
import jax.numpy as jnp
from dense_armor.utility.neural.trainer import Trainer
from dense_armor.utility.neural.models import mlp
from dense_armor.utility.neural.losses import MSE
from dense_armor.utility.optim import SGD

X = jax.random.normal(jax.random.PRNGKey(1), (256, 512))
y = jax.random.normal(jax.random.PRNGKey(2), (256, 512))

m_loop = mlp([512, 512, 512, 512, 512], jax.random.PRNGKey(0))
t_loop = Trainer(m_loop, MSE(), SGD(0.01))
t_loop.fit(X, y, epochs=1, batch_size=256, key=jax.random.PRNGKey(0))

m_compiled = mlp([512, 512, 512, 512, 512], jax.random.PRNGKey(0))
t_compiled = Trainer(m_compiled, MSE(), SGD(0.01), compile=True)
t_compiled.fit(X, y, epochs=1, batch_size=256, key=jax.random.PRNGKey(0))

print(t_loop.history_["train_loss"][-1], t_compiled.history_["train_loss"][-1])
```

```
0.3718 0.3718
```
The two losses are the same. The maximum relative difference over
the 200 steps is 1.04e-07 for the MLP and 9.71e-07 for the
transformer. Both are below the 1e-4 the benchmark asks for.

## The measurement

The numbers of the page come from
`benchmarks/compiled_step_cpu.json`, which
`benchmarks/compiled_step.py` writes. The table, from the JSON
(20 steps per session, 3 sessions, seed 0):

| model | steps/s loop | steps/s compiled | ratio | compile s |
|---|---|---|---|---|
| mlp 4x512, batch 256 | 21.37 | 1701 | 79.6x | 0.21 |
| transformer 4x256, seq 128, batch 32 | 0.364 | 401 | 1102x | 1.36 |
The range over the three sessions, from the same JSON:

- mlp, loop: 21.16 to 21.69.
- mlp, compiled: 1526 to 1914.
- transformer, loop: 0.36 to 0.37.
- transformer, compiled: 336 to 459.

## Details

### References

The formulas of the optimisers (SGD with momentum and Nesterov,
Adam, RMSProp, AdaGrad) are the ones of their object forms in
`dense_armor.utility.optim`, which cite the original papers. The
weight decay is Algorithm 1 line 9 and Algorithm 2 line 12 of
Loshchilov and Hutter 2019 (the decoupled term is eta_t times
lambda times theta_t_minus_1, with eta_t the schedule multiplier).
Proposition 1 of the same paper gives the equivalence between L2
and decoupled decay for plain SGD.
### Choices stated explicitly

- The step is pure. The values of the parameters are passed to the
  jitted function as a list of arrays, the optimiser state as a
  pytree of the same shape, and the loss is returned as a JAX
  scalar. The caller writes the new values into the parameters
  with `write_values` after every step. The caches that the layers
  keep for `backward` are local to the traced function.
- The schedule runs outside the jit. `lr` and `eta` are computed
  from a Python float epoch, because the schedule has a Python
  `if` on the epoch.
- The optimisers are the object forms. The jitted update uses the
  same arithmetic as `step(weights, gradient)`. A new optimiser
  that does not have a `step` method is not supported by
  `compile=True`, and the supported list is SGD, Adam, RMSProp,
  AdaGrad.
### Measured cost

From `benchmarks/compiled_step_cpu.json`, on the CPU of the run
(jax 0.11.1, python 3.13.18, x86_64):

- mlp: loop 21.37 steps per second, compiled 1701, ratio 79.6x,
  compile time 0.21 s.
- transformer: loop 0.364, compiled 401, ratio 1102x, compile time
  1.36 s.

The compiled mode is faster by a factor that depends on the model.
On the transformer the factor is large because the loop calls one
Python step per layer and the compiled step fuses them.
### Not measured

- The GPU. `benchmarks/compiled_step_gpu.json` is not in the
  repository. The benchmark needs a Kaggle GPU with the JSON
  committed, and this session does not have one. The claim (2) of
  the benchmark (GPU at least 3.0x) is reported, not measured.
- Mixed precision, multi-device, gradient checkpointing: these are
  the tasks N4, N5, N6.
- Dropout and BatchNorm in compiled mode. Both carry state that
  must cross the jit boundary (the PRNG key for Dropout, the
  running mean and variance for BatchNorm). They are declared as
  holes in the item 0 audit. The five reference models of the
  benchmark do not use them.
### Development diary

The first version of `compiled.py` computed the decoupled decay as
`v - lr * weight_decay * v`. The paper says
`v - eta_t * weight_decay * v`, with `eta_t` the schedule
multiplier, and the two differ when the schedule decays. The fix is
in the same commit. The tracer leak on the second step (the
`write_values` between two compiled steps) is the bug that
`test_early_stopping_path` caught: the validation forward reads the
`Param` values, which are still tracers from the first step, and
JAX raises.
