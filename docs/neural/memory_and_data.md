# Memory and data: how do I train when the data or the activations do not fit?

Training can run out of room in three places. The dataset may be larger than the memory of the machine. A batch that is good for the optimizer may be larger than the memory of the device. And a deep network keeps the activations of every layer until the backward pass needs them. This page shows one tool for each: a stream that serves the data in batches, gradient accumulation, and recomputation of the activations.

## 1. Stream the data

A `DataStream` serves batches of `(x, y)` without holding the whole dataset. The source is a pair of arrays, a pair of `.npy` files opened as memory maps (only the batch being served is read), or a function that returns an iterator of samples. Every sample is served once per epoch, and the last batch is smaller when the number of samples is not a multiple of the batch size.

The next block serves six samples in batches of four, then serves them again through a shuffle buffer of three samples. The buffer mixes the order while holding only three samples; at the end of the epoch it is emptied, so no sample is lost.

```python
import jax
import numpy as np
from dense_armor.utility.neural.stream import DataStream

X = np.arange(12, dtype=np.float32).reshape(6, 2)
y = np.arange(6)
a = DataStream((X, y), 4)
b = DataStream((X, y), 4, 3, jax.random.PRNGKey(0))
print([yb.tolist() for _, yb in a])
print(sorted(v for _, yb in b for v in yb.tolist()))
```

```
[[0, 1, 2, 3], [4, 5]]
[0, 1, 2, 3, 4, 5]
```

`Trainer.fit` takes the stream through its `stream` argument; `X` and `y` are then `None`, and `batch_size` is `None` or the batch size of the stream. The next block writes a dataset to two files and trains a small network from them.

```python
import tempfile
from pathlib import Path
import jax
import numpy as np
from dense_armor.utility.neural.models import mlp
from dense_armor.utility.neural.losses import MSE
from dense_armor.utility.neural.stream import DataStream
from dense_armor.utility.neural.trainer import Trainer
from dense_armor.utility.optim import SGD

rng = np.random.default_rng(0)
X = rng.normal(size=(64, 2)).astype(np.float32)
y = rng.normal(size=(64, 1)).astype(np.float32)
d = Path(tempfile.mkdtemp())
np.save(d / "x.npy", X)
np.save(d / "y.npy", y)
s = DataStream((d / "x.npy", d / "y.npy"), 8)
t = Trainer(mlp([2, 4, 1], jax.random.PRNGKey(0)), MSE(), SGD(0.05), compile=True)
t.fit(None, None, epochs=3, batch_size=None, key=jax.random.PRNGKey(0), stream=s)
print([round(v, 4) for v in t.history_["train_loss"]])
```

```
[1.9657, 1.1328, 1.0735]
```

## 2. Accumulate the gradient

A batch of 32 samples may not fit on the device while a batch of 8 does. With `accumulate=4` the trainer runs four batches of 8, adds their gradients, divides the sum by four and applies the optimizer once. The `batch_size` of `fit` is the size of the small batch. For small batches of the same size, the average of their mean gradients is the mean gradient of the large batch, so the update is the one of a batch of 32.

The next block trains the same network with a batch of 32 and with four accumulated batches of 8, for five epochs of 64 samples, ten updates in all, and compares the loss and the weights.

```python
import jax
import numpy as np
from dense_armor.utility.neural.models import mlp
from dense_armor.utility.neural.losses import MSE
from dense_armor.utility.neural.trainer import Trainer
from dense_armor.utility.optim import SGD

rng = np.random.default_rng(0)
X = rng.normal(size=(64, 2)).astype(np.float32)
y = rng.normal(size=(64, 1)).astype(np.float32)

def run(batch, k):
    m = mlp([2, 4, 1], jax.random.PRNGKey(0))
    t = Trainer(m, MSE(), SGD(0.05), compile=True, accumulate=k)
    t.fit(X, y, epochs=5, batch_size=batch, key=jax.random.PRNGKey(0), shuffle=False)
    return t.history_["train_loss"][-1], [np.asarray(p.value) for p in m.params()]

(l1, w1), (l4, w4) = run(32, 1), run(8, 4)
print(round(l1, 5), round(l4, 5), all(np.allclose(a, b, rtol=1e-5, atol=1e-7) for a, b in zip(w1, w4)))
```

```
1.08722 1.08722 True
```

The two losses are the same and the weights agree. The clipping of the gradient acts on the average, so it is the clipping of the large batch. If the number of small batches of an epoch is not a multiple of `accumulate`, the last update averages the small batches that are left.

## 3. Recompute the activations

A network keeps what each layer computed in the forward pass, because the backward pass needs it. For a deep transformer this is most of the memory. A block built with `remat=True` keeps only its input and runs its forward pass again when the backward pass reaches it (Chen et al. 2016, Section 4.1, page 4). The result is the same, and the cost is one more forward pass of the block per step.

The next block runs one block with and without `remat`, on the same weights, forward and backward. It prints whether the outputs and the gradients agree, and whether the first feed-forward layer still holds its input after the forward pass.

```python
import jax
import jax.numpy as jnp
from dense_armor.utility.neural.attention import TransformerBlock

k = jax.random.PRNGKey(0)
x = jax.random.normal(jax.random.PRNGKey(1), (5, 8))
g = jax.random.normal(jax.random.PRNGKey(2), (5, 8))
a = TransformerBlock(8, 2, 16, k)
b = TransformerBlock(8, 2, 16, k, remat=True)
ya, yb = a.forward(x), b.forward(x)
da, db = a.backward(g), b.backward(g)
print(bool(jnp.allclose(ya, yb, atol=1e-5)), bool(jnp.allclose(da, db, atol=1e-5)))
print(a.ffn1.x_ is None, b.ffn1.x_ is None)
```

```
True True
False True
```

`transformer_encoder(..., remat=True)` builds every block this way. `remat=True` needs `dropout=0`: the dropout mask would change when the forward pass is run again.

## Details

### References

- Chen, T., Xu, B., Zhang, C., Guestrin, C. (2016). Training Deep Nets with Sublinear Memory Cost. arXiv:1604.06174. Page 1: a network of `n` layers trained with `O(sqrt(n))` memory and the cost of one extra forward pass per mini-batch. Section 4.1, page 4: drop some intermediate results and recover them from an extra forward computation when needed. Section 4.3, page 5: the segments that give the `O(sqrt(n))` cost.

### Choices stated explicitly

- **The unit of `remat` is the block,** not a segment of several blocks as in the `O(sqrt(n))` scheme of Section 4.3: each block keeps its input and recomputes inside. It is done with `jax.checkpoint` on a function of the parameter values and of the input.
- **The accumulated gradient is a mean,** so the learning rate does not change with `accumulate`. With micro-batches of different sizes (the last batch of an epoch) the mean of the means weights the samples unequally.
- **`accumulate` is not combined** with `devices`, `compute_dtype` or `loss_scale`.
- **The schedule of a stream of unknown length** is evaluated at the integer epoch; for a stream of arrays or files the length is known and the schedule moves inside the epoch as before.
- **The shuffle buffer is bounded:** it mixes the samples only within the window of its size, so a buffer smaller than the dataset gives a partial shuffle.

### Measured cost

`benchmarks/memory_and_data.py` runs a `transformer_encoder` of 12 layers, `d_model=512`, sequence 256, batch 32, on synthetic data (seed 0), with SGD at learning rate `0.01` and one step per epoch. Each configuration runs in a process of its own, because the peak memory of a process only grows. Values in `benchmarks/memory_and_data_gpu.json`: `Tesla T4`, jax 0.11.1, a Kaggle GPU kernel; the speed is the median of 3 sessions of 20 steps.

| `remat` | peak bytes | steps/s, median (range) |
|---|---|---|
| `False` | 4 429 664 256 | 1.745 (1.727 to 1.773) |
| `True` | 991 240 704 | 1.236 (1.208 to 1.262) |

The stream is measured on 6400 synthetic samples of shape `(256, 64)` (64 KiB each) in a `.npy` file, in batches of 32:

| source | samples/s, median (range) |
|---|---|
| slices of the array in memory | 57 769 554 (52 757 833 to 58 711 826) |
| `DataStream` over the memory map, in order | 7 262 782 (7 237 789 to 7 436 110) |
| `DataStream` over the memory map, shuffle buffer of 512 | 55 451 (52 018 to 60 893) |

The slices of the array are views, so their rate measures only the loop. The shuffled stream handles one sample at a time in Python and is about 130 times slower than the in-order stream; the 12-layer model above consumes `1.745 * 32 = 56` samples per second, far below 55 451.

The claims were written before the first run:

| Claim | Result | Number |
|---|---|---|
| 1. the peak memory with `remat` is at most 0.7 times the peak without | **holds** | ratio `0.224` |
| 2. the time per step with `remat` is at most 1.5 times | **holds** | ratio `1.412` |
| 3. the accumulation of 4 batches of 8 equals one batch of 32 on the GPU too | **holds** | largest difference `9.3e-8` relative to the largest weight of each array, after 10 updates |

Each claim comes from one run of three sessions. The time ratio `1.412` is close to the bound of `1.5`, and the speed of a single T4 changes from run to run, so another run can put it on either side.

### Not measured

The memory of a stream on a dataset larger than the memory of the machine, the combination of `remat` with `accumulate`, a model of several blocks per segment as in Section 4.3, and a comparison with another framework.

### Development diary

The first version of the page cited Section 3 of Chen et al. for the idea of recomputing; the idea is in Section 4.1, page 4, and the docstring was corrected.
