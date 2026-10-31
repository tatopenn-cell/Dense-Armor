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
16.1
```

The block above is one run of 20 steps: 16.1 steps per second on the
machine that wrote this page; another machine or another load prints
another number. The median over three sessions in
`benchmarks/compiled_step_cpu.json` is 18.28 steps per second (range
11.32 to 21.43). Each step calls the layers one by one from Python.

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

The first call compiles the step (0.20 s for the MLP). After the
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
63.0
```

The block above times one `fit` of one batch: 63.0 steps per second on
the machine that wrote this page (the time depends on the machine). The
median over three sessions in `benchmarks/compiled_step_cpu.json` is
27.24 steps per second for the compiled step and 18.28 for the loop, a
ratio of 1.49x. On the transformer encoder of four layers the loop runs
at 0.365 steps per second and the compiled step at 0.427, a ratio of
1.17x.

## 3. Check that the loss is the same

The compiled step must give the same loss as the loop. Run both on
the same data and the same seed:
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
2.840958833694458 2.840958833694458
```
The two losses are the same. The relative difference between the loop and
the compiled step at the steps 1, 10, 50 and 200, from the two JSON files:

| model | backend | step 1 | step 10 | step 50 | step 200 |
|---|---|---|---|---|---|
| mlp | CPU | 0 | 0 | 7.89e-08 | 1.04e-07 |
| mlp | GPU | 0 | 0 | 0 | 0 |
| transformer | GPU | 0 | 3.42e-06 | 6.01e-08 | 2.52e-03 |

On the CPU the transformer was run for 3 steps only (9.71e-07 at step 3).
The loss of the compiled step is within 1e-4 of the loop for the mlp and
is not for the transformer at step 200 on the GPU.

## The measurement

The numbers of this section come from `benchmarks/compiled_step_cpu.json`
and `benchmarks/compiled_step_gpu.json`, which
`benchmarks/compiled_step.py` writes (20 steps per session, 3 sessions,
seed 0; the clock stops when the result of the step is ready). The CPU file
was written by a run on a CPU (jax 0.11.1, python 3.13.18). The GPU file was
written by a run in a Kaggle kernel with a Tesla T4 (jax 0.11.1, python
3.13.15); the model of the GPU is in the log of the kernel and not in the
JSON.

| model | backend | steps/s loop | steps/s compiled | ratio | compile s |
|---|---|---|---|---|---|
| mlp 4x512, batch 256 | CPU | 18.28 | 27.24 | 1.49x | 0.20 |
| transformer 4x256, seq 128, batch 32 | CPU | 0.365 | 0.427 | 1.17x | 2.16 |
| mlp 4x512, batch 256 | GPU | 58.44 | 566.36 | 9.69x | 0.18 |
| transformer 4x256, seq 128, batch 32 | GPU | 6.09 | 26.55 | 4.36x | 1.35 |

The range over the three sessions, from the same files:

- CPU, mlp: loop 11.32 to 21.43, compiled 16.90 to 30.77.
- CPU, transformer: loop 0.364 to 0.372, compiled 0.424 to 0.428.
- GPU, mlp: loop 54.90 to 59.14, compiled 516.25 to 567.54.
- GPU, transformer: loop 6.03 to 6.37, compiled 26.51 to 30.24.

The three claims of the benchmark, as the script prints them:

1. The loss of the compiled step is within 1e-4 of the loop at the steps 1,
   10, 50 and 200 on both models: holds for the mlp; does not hold for the
   transformer at step 200 on the GPU (2.52e-03); on the CPU the transformer
   was measured at 3 steps only.
2. On the GPU the compiled step is at least 3.0x the loop: holds (9.69x and
   4.36x).
3. On the CPU the compiled step is at least 1.5x the loop: does not hold
   (1.49x and 1.17x).

### Through `Trainer.fit`

The path a user takes is `Trainer.fit`, which also reads the loss once per
epoch and writes the parameters back after every step.
`benchmarks/compiled_fit_speed.py` times it: epochs after one warm-up epoch,
3 sessions, the clock stopped when the parameters are ready. The figures
below are the output of that script in a Kaggle kernel, on the commit
`3891c23` (CPU: 4 cores, `JAX_PLATFORMS=cpu`; GPU: Tesla T4). They are
printed by the script and not written in a JSON file.

| model | backend | steps/s `compile=False` | steps/s `compile=True` | ratio |
|---|---|---|---|---|
| mlp 4x512, batch 256 | CPU | 58.32 (51.01 to 59.31) | 111.07 (107.08 to 111.15) | 1.90x |
| mlp 4x512, batch 256 | GPU | 45.05 (44.86 to 46.48) | 191.56 (181.04 to 191.58) | 4.25x |
| transformer 4x256, seq 128, batch 32 | GPU | 6.12 (6.06 to 6.22) | 30.27 (30.25 to 30.36) | 4.95x |

The transformer is run on the GPU only.

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

From the two JSON files above. The ratios are medians of three sessions;
no Wilcoxon test and no A12 were computed. The compile time of the first
call is 0.20 s (mlp) and 2.16 s (transformer) on the CPU, and 0.18 s and
1.35 s on the GPU. The factor depends on the model and on the device.

### Behaviour of the compiled mode

- The state of the optimiser is kept between two calls of `fit`, as in the
  loop: `fit(2)` followed by `fit(2)` gives the same losses as `fit(4)` with
  `shuffle=False`
  (`test_second_fit_does_not_restart_the_optimizer`).
- The values of the parameters are read from the model at the start of
  every `fit`, so weights restored by early stopping, or set by hand, are
  the ones the next `fit` starts from.
- A model that contains Dropout, BatchNorm, GroupNorm or InstanceNorm is
  rejected with a `ValueError` that names the layer
  (`test_rejects_dropout`, `test_rejects_batchnorm`). The layers are found
  through the attributes `layers`, `inner`, `attn`, `ln1`, `ln2`, `ffn1`,
  `ffn2`, `dropout_attn` and `dropout_ffn`; a layer kept under another
  attribute name is not found.
- An optimiser other than SGD, Adam, RMSProp and AdaGrad is rejected with a
  `ValueError`.

### Not measured

- The speed of `Trainer.fit` is not in a JSON file: it is the printed
  output of `benchmarks/compiled_fit_speed.py`, run once per backend.
- The loss of the transformer at 200 steps on the CPU: it was run for 3
  steps, because 200 steps take about 27 minutes on that CPU.
- Mixed precision, multi-device and gradient checkpointing: the page does
  not cover them.

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
