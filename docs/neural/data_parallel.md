# Data parallelism: how do I train on more than one GPU?

One GPU holds only so many samples per step. With two GPUs the same step can use twice the samples: the batch is cut in two halves, each GPU computes the gradient of its half, the two gradients are averaged, and both GPUs apply the same update. The weights stay identical on every GPU, and the result is the one a single GPU would give with the whole batch.

## 1. The global batch

`Trainer(..., compile=True, devices=2)` trains on two devices. The `batch_size` of `fit` is the global batch: with `devices=2` and `batch_size=8`, each device receives 4 samples. The batch must be divisible by the number of devices, and so must the last smaller batch of an epoch.

The number of devices is fixed when JAX starts, so the next block starts a new Python process with four virtual CPU devices. It trains for three epochs on two of them, then asks for a batch of 3 on two devices.

```python
import os
import subprocess
import sys

code = """
import numpy as np
import jax
from dense_armor.core.deep.neural.models import mlp
from dense_armor.core.deep.neural.losses import MSE
from dense_armor.core.deep.neural.trainer import Trainer
from dense_armor.core.online.optim import SGD

X = np.random.default_rng(0).normal(size=(32, 2)).astype(np.float32)
y = np.random.default_rng(1).normal(size=(32, 1)).astype(np.float32)
m = mlp([2, 4, 1], jax.random.PRNGKey(0))
t = Trainer(m, MSE(), SGD(0.01), compile=True, devices=2)
t.fit(X, y, epochs=3, batch_size=8, key=jax.random.PRNGKey(0))
print(jax.device_count(), [round(v, 4) for v in t.history_['train_loss']])
try:
    t.fit(X, y, epochs=1, batch_size=3, key=jax.random.PRNGKey(0))
except ValueError as e:
    print(e)
"""
env = dict(os.environ, XLA_FLAGS="--xla_force_host_platform_device_count=4")
r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env)
print(r.stdout.strip())
```

```
4 [2.1086, 1.454, 1.0939]
batch_size 3 is not divisible by devices 2
```

`devices` is a number (the first devices), or a list of device ids, which are indices into `jax.devices()`. At least two devices are needed, they must exist, and `devices` needs `compile=True`. It can be combined with `compute_dtype` and `loss_scale` of [Mixed precision](mixed_precision.md).

## 2. The average of the gradients

Each device computes the gradient of the loss on its own shard, a mean over the shard. With `k` devices and shards of the same size `n`, the mean of the `k` shard gradients is the gradient of the mean over the whole batch of `k n` samples. That is why the batch must be divisible: with shards of different sizes the plain average would weight the samples unequally.

In symbols (Goyal et al. 2017, Section 2.3, page 4, equation (7)), with `w` the weights, `eta` the learning rate, `B_j` the shard of worker `j` and `L` the loss:

`w_next = w - eta * (1/k) * sum over j < k of grad L(B_j, w)`

The average is a `pmean`, an operation that every device takes part in and that returns the same result on every device. A hand case with two devices and two weights: the first device has the gradient `[1, 3]`, the second `[3, 5]`.

```python
import os
import subprocess
import sys

code = """
import jax
import jax.numpy as jnp

g = jnp.array([[1.0, 3.0], [3.0, 5.0]])
print(jax.pmap(lambda v: jax.lax.pmean(v, 'd'), axis_name='d')(g))
"""
env = dict(os.environ, XLA_FLAGS="--xla_force_host_platform_device_count=2")
r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env)
print(r.stdout.strip())
```

```
[[2. 4.]
 [2. 4.]]
```

Both devices hold `[2, 4]`, the mean of `[1, 3]` and `[3, 5]`. The average comes before the clipping of the gradient and before the update, so the clipping norm is the one of the whole batch.

The next block trains the same model on one, two and four devices, with the whole batch of 32 samples in one step per epoch, and prints the loss of the first three steps.

```python
import os
import subprocess
import sys

code = """
import numpy as np
import jax
from dense_armor.core.deep.neural.models import mlp
from dense_armor.core.deep.neural.losses import MSE
from dense_armor.core.deep.neural.trainer import Trainer
from dense_armor.core.online.optim import SGD

X = np.random.default_rng(0).normal(size=(32, 2)).astype(np.float32)
y = np.random.default_rng(1).normal(size=(32, 1)).astype(np.float32)

def run(devices):
    m = mlp([2, 4, 1], jax.random.PRNGKey(0))
    t = Trainer(m, MSE(), SGD(0.01), compile=True, devices=devices)
    t.fit(X, y, epochs=3, batch_size=32, key=jax.random.PRNGKey(0), shuffle=False)
    return [round(v, 5) for v in t.history_['train_loss']]

for d in (None, 2, 4):
    print(d, run(d))
"""
env = dict(os.environ, XLA_FLAGS="--xla_force_host_platform_device_count=4")
r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env)
print(r.stdout.strip())
```

```
None [2.42154, 2.16323, 1.94653]
2 [2.42154, 2.16323, 1.94653]
4 [2.42154, 2.16323, 1.94653]
```

`None` is one device. The three lines are the same: splitting the batch does not change the training.

## 3. The learning rate

A larger global batch gives a less noisy gradient, and the learning rate can grow with it. The linear scaling rule (Goyal et al. 2017, Section 2.1, page 2): when the batch is multiplied by `k`, multiply the learning rate by `k`. `linear_scaling(lr, batch, base_batch)` computes it. At the start of training a large learning rate can make the model diverge, so the paper raises the learning rate gradually over the first epochs (Section 2.2, page 3); `LinearWarmup` does that.

```python
from dense_armor.core.deep.neural.data_parallel import linear_scaling
from dense_armor.core.deep.neural.schedules_more import LinearWarmup

lr = linear_scaling(0.1, 256, 32)
s = LinearWarmup(0.1, lr, 5)
print(lr, [round(s(e), 2) for e in (0.0, 2.5, 5.0, 7.0)])
```

```
0.8 [0.1, 0.45, 0.8, 0.8]
```

A batch of 256 against a base of 32 multiplies the rate by 8, from `0.1` to `0.8`. The warmup starts at `0.1`, reaches `0.45` halfway through the five epochs and `0.8` at the end, and stays there. The schedule is given to `Trainer` through its `schedule` argument.

## Details

### References

- Goyal, P., Dollar, P., Girshick, R., Noordhuis, P., Wesolowski, L., Kyrola, A., Tulloch, A., Jia, Y., He, K. (2017). Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour. arXiv:1706.02677. Section 2.1, page 2: the linear scaling rule. Section 2.1, page 3: with `eta_hat = k eta` one large step approximates `k` small steps. Section 2.2, page 3: gradual warmup. Section 2.3, page 4, equation (7): the average of the gradients of the `k` workers.
- Micikevicius, P. et al. (2017). Mixed Precision Training. arXiv:1710.03740. Section 3.1, page 2, and pages 4-5, for the mixed step used with several devices.

### Choices stated explicitly

- **`jax.pmap` with `jax.lax.pmean`.** The step is a function compiled for every device, with the device axis named `devices`. The other way, sharding the arrays over a mesh, was not chosen. `pmap` runs in jax 0.11 without a deprecation warning and runs on virtual CPU devices, which the tests need.
- **The average is taken in float32.** With a compute dtype the gradients are stored in that dtype, cast to float32, averaged, divided by the loss scale and then clipped. The sum of two float16 gradients can exceed 65504 and give a false overflow, which the float32 average avoids.
- **One flag for all devices.** The overflow flag is reduced with `pmin`, so that a step skipped on one device is skipped on all of them and the replicated weights stay equal. The dynamic loss scale is a single value for all devices.
- **The loss is averaged too.** The loss reported by `fit` is the mean over the devices.
- **The weights stay on the device axis during an epoch,** and are written back to the model once per epoch, from the first device.
- **Neither paper prescribes the combination** of the average, the flag and the loss for the mixed step: Goyal covers the average over workers and not the mixed step, Micikevicius covers the mixed step and not several devices.
- **`devices=None` keeps the step as it was:** the losses of the default compiled trainer equal the values stored from the master branch.

### Measured cost

`benchmarks/data_parallel.py` runs a `transformer_encoder` of 6 layers, `d_model=512`, sequence 256, global batch 64, on synthetic data (seed 0), with SGD at learning rate `0.01` and one step per epoch. Each device count runs in a process of its own. Values in `benchmarks/data_parallel_gpu.json`: `Tesla T4`, jax 0.11.1, a Kaggle kernel with two GPUs; the speed is the median of 3 sessions of 20 steps.

| devices | steps/s, median (range) | loss at step 1 | step 10 | step 50 | step 200 | peak bytes on device 0 |
|---|---|---|---|---|---|---|
| 1 | 1.275 (1.266 to 1.279) | 2.806067 | 1.306522 | 0.980802 | 0.953850 | 4 523 575 040 |
| 2 | 2.286 (2.270 to 2.310) | 2.806067 | 1.306519 | 0.980803 | 0.953850 | 2 595 831 040 |

The claims were written before the first run:

| Claim | Result | Number |
|---|---|---|
| 1. the loss on 2 GPUs is within `1e-4` relative of 1 GPU at steps 1, 10, 50 and 200 | **holds** | largest relative difference `2.5e-6`, at step 10 |
| 2. 2 GPUs have at least 1.5 times the steps per second of 1 GPU | **holds** | ratio `1.792` |

The ratio is from one run of three sessions per device count. The sessions inside a run agree to a few percent, but two runs of the single-device case on kernels of the same kind gave speeds that differ by more than that range, so the ratio of another run can differ.

### Not measured

More than two GPUs, several machines, the share of the step spent in the average, models other than the transformer encoder, a comparison with a statistic (Wilcoxon, A12) because the claims compare one run per device count, and a comparison with another framework.

### Development diary

The first version of the parallel step passed the tests of its own file but failed in two ways that the tests did not cover: a second call of `fit` crashed because the weights written back after the first call lived on one device only, and in the mixed mode the optimizer state was rebuilt at every call. A list of device ids was accepted and ignored, and a last batch that the devices did not divide ended with an error from the reshape. The tests now cover each of them. A first benchmark showed that `fit` copied the weights to the host at every epoch, which with one step per epoch cost more than the step; the copy is now made only when a validation set is evaluated and at the end of `fit`.
