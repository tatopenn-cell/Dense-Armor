# Online statistics and sketches

## What this is

A robot reads its own joints sixty times a second. A server watches the
latency of every request. A sensor publishes a temperature every
minute. In each case, the data arrives **one sample at a time**, and it
never stops.

If you want the average of that stream, the obvious move is to keep
every sample in a list and average it at the end. That is fine for a
hundred readings. It is not fine for a million, and it is impossible
for a stream that has no end: the list grows without bound, the answer
arrives too late, and if the machine restarts you lose everything.

This module answers statistical questions about a stream **while the
stream is still arriving**. Every estimator:

- reads a single sample with `learn_one(x)`;
- answers immediately, using only what has been seen so far;
- runs in **fixed memory** — `O(1)`, or `O(window)` when a rolling
  window is the right model;
- can be combined with another estimator of the same kind through
  `merge(other)` when the mathematics allows it.

## The first five lines

```python
from dense_armor.utility.stats.moments import RunningMoments

m = RunningMoments()
for v in [1.0, 2.0, 3.0, 4.0, 5.0]:
    m.learn_one({"x": v})
print(round(m.mean, 3), round(m.var, 3))
```

Five samples go in one by one, then the mean and the variance come out.
No list is kept. Memory stays the same from the first sample to the
millionth.

## Why "online" matters

The difference between batch statistics and online statistics is not a
detail of implementation. It is a difference in what the program can
do.

| Question | Batch (keep everything) | Online (keep a summary) |
|---|---|---|
| Memory for N samples | grows with N | constant |
| First answer available | after all samples | after the first sample |
| Handles an endless stream | no | yes |
| Survives a restart | lose everything | `state_dict` / `load_state_dict` |
| Cost per sample | one entry appended | a few arithmetic operations |

For a control loop that must decide *now*, the difference is between a
system that works and a system that stalls.

## Five families

Different streams need different summaries. The library groups the
statistics into five families, one file per family, each with its own
guide page.

| Family | The question it answers | Classes |
|---|---|---|
| Moments | What is the typical value, and how much does it wobble? | `RunningMoments`, `RunningMomentsVector`, `EWStats` |
| Robust | What is the typical value when a few outliers would ruin a mean? | `RollingMedian`, `RollingMAD`, `RollingIQR`, `RollingQuantile`, `RollingMedianVector` |
| Quantiles | Where is the 99th percentile of an unbounded stream? | `DDSketch`, `TDigest` |
| Dependence | Do two channels move together? Does a signal repeat itself? | `RunningCovariance`, `RunningCorrelation`, `RollingCovariance`, `RollingCorrelation`, `Autocorrelation` |
| Sketches | How many distinct items have I seen? Which items are frequent? | `CountMinSketch`, `HyperLogLog`, `BloomFilter`, `SpaceSaving` |

Each guide page follows the same structure: an idea in plain words, a
short runnable example, the mathematics behind the update, a small case
worked out by hand, a stream from a robot, and a Details section at the
bottom with references and design notes.

## The common shape

Every statistic inherits from `dense_armor.roles.Transformer` and reads
the same kind of input: a dict with one or more features.

```python
from dense_armor.utility.stats.moments import RunningMoments

m = RunningMoments()
for v in [1.0, 2.0, 3.0, 4.0, 5.0]:
    m.learn_one({"x": v})
print(round(m.mean, 3), round(m.var, 3))
```

The dict is one sample. `feature=None` (the default) reads the smallest
key, so the result never depends on the order in which the keys appear.
A specific key can be named at construction:

```python
from dense_armor.utility.stats.moments import RunningMoments

m = RunningMoments()
for v in [1.0, 2.0, 3.0, 4.0, 5.0]:
    m.learn_one({"x": v})
print(round(m.mean, 3), round(m.var, 3))
m = RunningMoments(feature="temperature")
m.learn_one({"temperature": 21.4, "humidity": 0.6})
```

`NaN` is not an error. When a statistic sees one, it skips that sample
and bumps an internal `n_missing` counter, so a sensor glitch never
poisons the state:

```python
from dense_armor.utility.stats.moments import RunningMoments

m = RunningMoments()
for v in [1.0, 2.0, 3.0, 4.0, 5.0]:
    m.learn_one({"x": v})
print(round(m.mean, 3), round(m.var, 3))
m = RunningMoments(feature="temperature")
m.learn_one({"temperature": 21.4, "humidity": 0.6})
m = RunningMoments()
for v in [1.0, float("nan"), 3.0]:
    m.learn_one({"x": v})
print(m.count, m.n_missing)
```

The output is `2 1` — two valid samples recorded, one skipped.

## Combining summaries

Two partial summaries of the same kind can be combined into one. This
is how a sharded system computes the same statistic across many
machines: each shard builds a summary of its own data, then the
summaries are merged.

```python
from dense_armor.utility.stats.moments import RunningMoments

left, right = RunningMoments(), RunningMoments()
for v in range(10):
    left.learn_one({"x": float(v)})
    right.learn_one({"x": float(v + 10)})
merged = left.merge(right)
print(merged.transform_one({}))
```

Whether the merge is exact depends on the mathematics. `RunningMoments`
and `RunningCovariance` merge exactly, using the pairwise formulas of
Chan, Golub and LeVeque (1979). `DDSketch` and `TDigest` merge by
combining their buckets or centroids. Every guide page states the
guarantee for its family.

## Time on a robot

The `t` argument of `learn_one` accepts the timestamp of the sample in
seconds. When timestamps are given, the base records the last 64
inter-sample intervals and exposes three derived quantities:

- `dt` — the median interval, robust to scheduler jitter;
- `rate` — `1 / dt`, the sample rate in Hz;
- `jitter` — `1.4826 · MAD(dt)`, the robust spread of the intervals.

A window given in seconds becomes a number of samples automatically:

```python
from dense_armor.utility.stats.robust import RollingMedian

import numpy as np

stream = [{"qd": float(v)} for v in np.random.default_rng(0).normal(0.5, 0.1, 100)]
m = RollingMedian(window_s=0.05)
for i, x in enumerate(stream):
    m.learn_one(x, t=i * 0.01)
```

Here `window_s=0.05` means "half a second", and at 100 Hz that becomes
five samples. If no `t` is given, `dt` stays `None` and windows are
interpreted in samples.

## A stream from a robot

A joint stream arrives as a sequence of standard joint-state readings:
positions `q`, velocities `qd`, torques `tau`, sampled at a fixed rate.
Each row is one sample, and each sample goes into the statistic of
interest.

```python
from dense_armor.utility.learn.online_dynamics import write_minimal_urdf
from dense_armor.dynamics.urdf_dynamics import RigidBodyModel
from dense_armor.utility.stats.moments import RunningMomentsVector

model = RigidBodyModel(write_minimal_urdf())
import numpy as np
rng = np.random.default_rng(0)
stream = [tuple(rng.uniform(-1, 1, (3, model.n))) for _ in range(200)]
stats = RunningMomentsVector()
for i, (q, qd, tau) in enumerate(stream):
    readings = {f"q{j}": float(q[j]) for j in range(model.n)}
    stats.learn_one(readings, t=i * 0.01)
```

The `RunningMomentsVector` keeps one running mean and variance per
joint, so all joints move through the summary in lockstep. After the
loop, `stats.transform_one({})` returns a dict with the summary of each
joint.

## How to read a family page

Each family guide is written so the first part can be followed by
someone who has never seen the library, and the last part gives the
detail a specialist expects. The structure is always the same.

1. **The idea in plain words.** What the statistic does, why it is
   needed, on which kind of stream it matters.
2. **The first estimator.** A short, runnable example. The code comes
   after the explanation, never before, and it is self-contained.
3. **The mathematics.** A formula, every symbol named, and a small
   case worked out by hand with the numbers the code actually produces.
4. **The merge.** How two partial summaries combine when the
   mathematics allows it.
5. **A robot example.** A stream of joint-state readings, at a fixed
   sample rate, processed with the same interface.
6. **Details.** References, design notes, edge cases, and the file path
   of the figure on the page.

If you have never used the library before, start from section 1 of any
family page and read down. If you already know what you need, jump to
the section title that matches it. The Details section is the one to
read last.

## Where each thing lives

```
dense_armor/utility/stats/__init__.py       the family index
dense_armor/utility/stats/moments.py        RunningMoments, RunningMomentsVector, EWStats
dense_armor/utility/stats/robust.py         RollingMedian, RollingMAD, RollingIQR, RollingQuantile, RollingMedianVector
dense_armor/utility/stats/quantiles.py      DDSketch, TDigest
dense_armor/utility/stats/dependence.py     RunningCovariance, RunningCorrelation, RollingCovariance, RollingCorrelation, Autocorrelation
dense_armor/utility/stats/sketches.py       CountMinSketch, HyperLogLog, BloomFilter, SpaceSaving
```

Every class inherits from `dense_armor.roles.Transformer`, passes
`dense_armor.checks.check_estimator`, and carries a runnable doctest in
its docstring.

## See also

- `docs/api/base.md` — the `Base` / `Transformer` interface that every
  statistic in this module shares.
- `docs/stats/moments.md`, `robust.md`, `quantiles.md`, `dependence.md`,
  `sketches.md` — the family guides.
