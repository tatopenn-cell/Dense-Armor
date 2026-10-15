# Getting started

Dense-Armor learns from data that arrives one sample at a time. This page starts with one model learning from a stream and ends with a command that cannot jump. Each step uses what the step before it showed.

## 1. Install

```bash
pip install dense-armor
```

Python 3.10 or newer. The computation runs on JAX, which is installed with it.

## 2. A model that learns one sample at a time

A normal model is trained once on a table and then used. An online model reads one sample, answers, and only then learns from it. Three methods describe every estimator in the library: `predict_one(x)` answers, `learn_one(x, y)` learns, `score_one(x)` says how strange `x` is.

```python
from dense_armor.utility.datasets import DriftStream
from dense_armor.utility.learn.online_classifiers import OnlineGaussianNB
from dense_armor.utility.metrics import Accuracy

ds = DriftStream(n_samples=2000, n_features=5, n_drifts=1, kind="sudden", seed=0)
nb = OnlineGaussianNB()
acc = Accuracy()
for x, y in ds.stream():
    p = nb.predict_one(x)
    if p is not None:
        acc.update(y, p)
    nb.learn_one(x, y)
print(acc)
```

```
Accuracy(n=1999, value=0.9989994997498749)
```

`DriftStream` makes 2,000 samples with 5 features and one sudden change of the rule that links them to the label. The model predicts first and learns second, so the accuracy is measured on samples it has never seen. The first sample has no prediction (`None`), so 1,999 are counted. The model made 2 mistakes, both in the first four samples.

## 3. Notice that the data changed

A model can fail quietly when the world changes. A drift detector watches one number and says when its average has moved. `PageHinkley` takes one value per sample and raises `drift_detected` when the mean shifts.

```python
import numpy as np
from dense_armor.utility.drift.page_hinkley import PageHinkley

rng = np.random.default_rng(0)
xs = np.concatenate([rng.normal(0, 1, 500), rng.normal(3, 1, 500)])
ph = PageHinkley()
hits = []
for i, v in enumerate(xs):
    ph.update(float(v))
    if ph.drift_detected:
        hits.append(i)
print(hits[0])
```

```
515
```

The mean of the stream moves from 0 to 3 at sample 500. The detector fires at sample 515: it needed 15 samples of the new level to be sure. In a real system the number to watch is often the error of the model from step 2. The [drift page](drift/index.md) lists the other detectors.

## 4. Score how strange a point is

Some data has no label. Anomaly detectors learn what normal looks like and give each new point a score: high means far from normal. They follow the same `learn_one` / `score_one` pair.

```python
from dense_armor.utility.anomaly.mahalanobis import OnlineRobustMahalanobis

rng = np.random.default_rng(0)
det = OnlineRobustMahalanobis()
for _ in range(1000):
    det.learn_one({"a": float(rng.normal()), "b": float(rng.normal())})
print(round(det.score_one({"a": 0.1, "b": -0.2}), 2))
print(round(det.score_one({"a": 5.0, "b": 5.0}), 2))
```

```
0.35
9.9
```

The detector learned from 1,000 points of a standard normal distribution on two channels. A point near the centre scores 0.35. The point (5, 5) scores 9.9; the default alarm threshold is 7.0. The score stays at 0.0 during the first samples while the detector builds its first estimate: with 50 samples the same two points both score 0.0, with 200 they score 0.40 and 8.83. Give a detector a few hundred samples before reading its scores. The [anomaly page](anomaly/index.md) compares the detectors.

## 5. Clean a series before the model sees it

A broken sensor sends `NaN` or a value of 9999 and the model learns from it. `Armatura` is a shield that finds such points and replaces them.

```python
from dense_armor import Armatura

serie = np.array([1.0, 1.1, 1.2, 9999.0, 1.15, np.nan, 1.2, 1.25])
pulito, K, anomalie = Armatura(livello_ia=0.0).analizza(serie)
print(pulito)
print(anomalie)
```

```
[1.   1.1  1.05 1.1  1.15 1.2  1.2  1.2 ]
[3, 5]
```

The names are Italian: `analizza` is "analyse", `livello_ia=0.0` means the shield filters actively (1.0 only marks, and the 9999 stays in the series). It returns three things: the cleaned series, `K`, and the list `anomalie` of replaced positions, here 3 (the 9999) and 5 (the `NaN`). Both are replaced and the series keeps its length. `K` is the flag of the internal trigger, one per sample, and it is not the same list: the `NaN` is cleaned before the trigger, so `K` is 0 at position 5, and the trigger also marks positions 2 and 7. The sample at position 2 is replaced by the local median (1.2 becomes 1.05) without being listed in `anomalie`. The [Armatura page](shields/armatura.md) explains the trigger. [Shields](shields/index.md) shows the others, including `Orca`, which wraps a whole model.

## 6. Keep a command from jumping

A controller that receives a spike can send it to a motor. `rate_limited_follower` follows a command but never changes faster than the limits you give.

```python
from dense_armor.utility.control.rate_limiter import rate_limited_follower

raw = np.sin(np.linspace(0, 6, 200))
raw[50] = 1e6
safe = rate_limited_follower(raw, max_vel=2.0, max_accel=1.0, dt=0.01)
print(raw.max(), round(np.abs(safe).max(), 2))
```

```
1000000.0 0.81
```

The command is a slow sine with one spike of 1,000,000 at sample 50. The applied command never goes above 0.81 and never changes faster than 0.9 per second, under the limit of 2.0. It uses only the previous applied value and the current target, so it works the same way on a live stream. The [control pages](control/index.md) add trajectories and a barrier filter for obstacles.

## Where to go next

- [Finance pipelines](finance/index.md): nine complete examples on markets, fraud, intrusion and credit scoring, each opening with the question it answers.
- [Online trees and forests](tree/index.md) and [forests](forest/index.md): models that grow as the data arrives.
- [Linear models](linear/glm.md) and [calibration](learn/calibration.md): the simplest learners and honest probabilities.
- [API reference](api/index.md): every class.

## Details

### What this page replaced

The first version was written before the learning, robot and vision modules existed. A second version added them as a list of tools. This one follows one thread, from learning to protection, and drops the examples that needed files from the repository (`panda.urdf`, vision patches, the toolkit). Those have their own pages: [dynamics](dynamics/index.md), [vision](vision/index.md), [toolkit](toolkit/toolkit.md).

### Numbers on this page

Every printed result was produced by running the code above with `numpy.random.default_rng(0)` or `DriftStream(seed=0)`. The earlier page stated that `OnlineRobustMahalanobis` gives a score of about 9.2 after 50 training points; the detector scores 0.0 at that size (step 4).
