# Metrics on a stream: windows, predict then learn, model choice

## 6. Windows: only the recent past

A robot changes over time; the accuracy of last week says little about now. `Rolling` keeps a
metric over the last `window` samples, or the last `window_s` seconds.

```python
from dense_armor.utility.metrics import Accuracy, Rolling

m = Rolling(Accuracy(), window_s=0.05)
for i in range(20):
    m.update(1 if i < 17 else 0, 1, t=i * 0.01)
print(m.count, m.get())
```

```
5 0.4
```

At 100 Hz a window of 0.05 s holds 5 samples: the window keeps the samples with
$t_\text{now} - t < 0.05$. The last three predictions are wrong, so the accuracy over the
window is 2/5. The window uses the timestamps, not the sample count, so a dropped sample does
not stretch it. Any metric with `revert` can be rolled.

The **ROC-AUC** is the probability that a random positive gets a higher score than a random
negative. `RollingAUC` computes it exactly on the last `window` pairs.

```python
import numpy as np
from dense_armor.utility.metrics import RollingAUC

rng = np.random.default_rng(0)
m = RollingAUC(window=200)
for _ in range(1000):
    y = int(rng.random() < 0.3)
    m.update(y, y + rng.normal(0, 1.0))
print(round(m.get(), 3))
```

```
0.712
```

$$\mathrm{AUC} = \frac{1}{n_+ n_-}\sum_{i \in +}\sum_{j \in -}
\begin{cases} 1 & s_i > s_j \\ \tfrac12 & s_i = s_j \\ 0 & \text{otherwise} \end{cases}$$

Here positives score one unit higher than negatives with unit noise, so the long-run value is
$\Phi(1/\sqrt2) = 0.760$; a window of 200 pairs gives 0.712.

## 7. Predict, then learn

`progressive_val_score` runs the protocol for you: for each sample the model predicts, the
metric scores, then the model learns. Labels often arrive late (a technician confirms a fault at
the end of the shift); with `delay` the model learns each label only when it arrives.

```python
import numpy as np
from dense_armor.utility.evaluate import progressive_val_score
from dense_armor.utility.learn.online_classifiers import OnlineGaussianNB
from dense_armor.utility.metrics import Accuracy, Rolling

rng = np.random.default_rng(0)
qd = rng.normal(0, 1, 2000)
y = np.abs(qd) > np.where(np.arange(2000) < 1000, 0.5, 1.5)
s = [({"qd": a}, int(b)) for a, b in zip(qd, y)]
for d in (None, 300):
    m, tr = progressive_val_score(s, OnlineGaussianNB(), Rolling(Accuracy(), window=200), delay=d, every=200)
    print(d, [round(r, 2) for r in tr])
```

```
None [0.91, 0.94, 0.95, 0.95, 0.92, 0.54, 0.66, 0.76, 0.84, 0.84]
300 [0.0, 0.0, 0.0, 0.89, 0.95, 0.92, 0.71, 0.38, 0.59, 0.7]
```

The task is "is the joint moving fast?", and at sample 1000 the meaning of "fast" changes from
$|\dot q| > 0.5$ to $|\dot q| > 1.5$. With labels on time the rolling accuracy drops to 0.54 and
recovers to 0.84. With labels 300 samples late the drop comes later and goes deeper (0.38),
because the model keeps answering with the old rule until the new labels reach it. For `delay=300` the first
three values are 0.0 because no label has arrived yet and nothing has been scored. `every=200`
returns the trace of the metric.

For metrics that need a probability, `progressive_val_proba_score` uses `predict_proba_one`.

```python
import numpy as np
from dense_armor.utility.evaluate import progressive_val_proba_score
from dense_armor.utility.learn.online_classifiers import OnlineGaussianNB
from dense_armor.utility.metrics import LogLoss

rng = np.random.default_rng(0)
s = [({"qd": v}, int(abs(v) > 0.5)) for v in rng.normal(0, 1, 2000)]
m, tr = progressive_val_proba_score(s, OnlineGaussianNB(), LogLoss(), every=500)
print([round(v, 3) for v in tr])
```

```
[0.312, 0.283, 0.264, 0.253]
```

The log-loss falls as the model sees more samples.

## 8. Choosing a model on the stream

`best_of` runs several models on the same stream with the same protocol and tells which one is
ahead.

```python
import numpy as np
from dense_armor.utility.evaluate import best_of
from dense_armor.utility.learn.online_classifiers import OnlineGaussianNB, OnlineSoftmaxRegression
from dense_armor.utility.metrics import Accuracy

rng = np.random.default_rng(0)
s = [({"qd": v}, int(abs(v) > 0.5)) for v in rng.normal(0, 1, 2000)]
sel = best_of([OnlineGaussianNB(), OnlineSoftmaxRegression()], Accuracy, stream=s)
print([round(v, 3) for v in sel.scores()], sel.current_best())
```

```
[0.941, 0.621] 0
```

"Fast in either direction" is not a straight cut on $\dot q$, so a linear softmax model cannot
learn it (0.621), while the Gaussian model, which learns a different spread per class, can
(0.941). `current_best()` returns the index 0 and `best_model()` the model itself. The second
argument is the metric class: each model gets its own fresh metric.
