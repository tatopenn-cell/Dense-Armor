# Choosing features and rare events

## 8. Keep only the useful features

```python
from dense_armor.utility.preprocessing.select import SelectKBest, VarianceThreshold

vt, sk = VarianceThreshold(threshold=0.1), SelectKBest(k=1)
for i in range(20):
    x = {"a": float(i), "b": float(i % 3), "c": 7.0}
    vt.learn_one(x)
    sk.learn_one(x, y=2.0 * i)
print(sorted(vt.transform_one(x)), sorted(sk.transform_one(x)))
```

```
['a', 'b'] ['a']
```

The variance threshold drops `c`, which never changes; k-best keeps the feature most correlated
with the target, `a`. The two tests look at different things: variance is about a feature alone,
correlation is about a feature and the label together. A constant feature cannot help any model,
and a feature uncorrelated with the target is just noise.

## 9. Rare events: queue resampling

When only 1 % of the samples are the event of interest (a fault), a model sees almost only
normal samples and learns to ignore the rare ones. Queue resampling keeps the last few examples
of each class and trains on both every time (Malialis et al. 2018, Algorithm 1).

```python
import random
from dense_armor.utility.evaluate import progressive_val_score
from dense_armor.utility.learn.online_classifiers import OnlineGaussianNB
from dense_armor.utility.metrics import Recall
from dense_armor.utility.preprocessing.imbalance import QueueResampler

rng = random.Random(0)
s = []
for _ in range(5000):
    y = 1 if rng.random() < 0.01 else 0
    m = 2.0 if y else -2.0
    s.append(({"x0": rng.gauss(m, 1.0), "x1": rng.gauss(m, 1.0)}, y))
a = progressive_val_score(s, OnlineGaussianNB(), Recall(positive=1))
b = progressive_val_score(s, QueueResampler(OnlineGaussianNB(), queue_size=25), Recall(positive=1))
print(round(a.get(), 4), round(b.get(), 4))
```

```
0.8955 0.9403
```

Recall of the rare class (how many of the real events are caught) without and with queue
resampling, on 5000 samples with 1 % events. The resampler keeps two queues of the same length,
one for the positive class and one for the negative, and at each step it feeds the classifier
the union of the two. The queue length bounds how many old examples are kept, so the training
set stays balanced even when the stream is not. The union also acts as a sliding window: very
old examples leave the queue, which lets the classifier follow a change in the stream.

## float32

`SelectKBest(dtype=numpy.float32)` keeps the running correlation
between each feature and the target in float32 and applies the
shifted-data algorithm with ``K = (x[0], y[0])`` (Healy 2022,
Algorithm 2) online: five TwoSum accumulators per feature and the
two-term formulas for ``m2_x``, ``m2_y`` and ``c_xy``. Relative error
of ``m2_x`` against the float64 reference on N(1e3, 0.01), 100,000
samples, seed 0: 1.7e-8. On a constant stream at 1e10 and 1e38
``m2_x``, ``m2_y`` and the correlation are exactly 0.0.

Cost, median of 7 runs of one update over 20,000 samples, one CPU thread: default 0.001245 ms (IQR 0.000022), float32 0.020472 ms (IQR 0.009036). The claim that the float32 path
is at most as slow as the default is not met: on a scalar loop the
numpy float32 arithmetic is slower than the Python float64 arithmetic.
The float32 speed of these classes is not measured; no class in this
batch has a `learn_many` on which the float32 blocks of the T3 kernel
could be timed.

