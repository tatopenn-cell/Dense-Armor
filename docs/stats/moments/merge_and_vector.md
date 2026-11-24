# Merging and many channels

[← Moments](index.md)

Two summaries can be joined into one, and one object can follow every joint of a robot at once.

## 1. Combining two halves

Two `RunningMoments` built on disjoint halves of a stream can be merged into the summary of the union.

```python
from dense_armor.core.signals.stats.moments import RunningMoments

a = RunningMoments()
b = RunningMoments()
for v in [1.0, 2.0, 3.0]:
    a.learn_one({"x": v})
for v in [10.0, 20.0]:
    b.learn_one({"x": v})
g = a.merge(b)
print(g.count, round(g.mean, 3), round(g.var, 3))
```

```
5 7.2 63.7
```

The merged count is the sum of the two, and the merged mean is $(1 + 2 + 3 + 10 + 20) / 5 = 7.2$, the mean of the union. The merged summary is **exactly** the one a single accumulator would have produced from all five samples: its variance, skewness and kurtosis are `63.7`, `0.916` and `-0.734` either way, not an approximation. The formulas are those of Chan, Golub and LeVeque (1979) for the variance, extended by Pébay (2008) to the third and fourth moments.

This is how a sharded system computes one global statistic without ever moving the samples. Each worker builds a summary of its own data, the workers send their summaries to a coordinator, and the coordinator merges them. The network cost is the size of one summary per worker, which is fixed regardless of how many samples each worker saw.

## 2. All joints at once

A robot has many joints. Feeding them one by one into separate `RunningMoments` objects works, but is repetitive to write and easy to get wrong. `RunningMomentsVector` keeps one `RunningMoments` per channel and updates all of them from a single dict.

```python
from dense_armor.core.signals.stats.moments import RunningMomentsVector

v = RunningMomentsVector()
for q in ([0.1, 0.2], [0.11, 0.19], [0.09, 0.21]):
    v.learn_one({"q0": q[0], "q1": q[1]})
s = v.transform_one({})
print(round(s["q0"]["mean"], 4), round(s["q1"]["mean"], 4))
```

```
0.1 0.2
```

`transform_one({})` returns one summary per channel, and each summary has the count, the mean, the variance, the standard deviation, the skewness, the kurtosis, the minimum, the maximum, the peak-to-peak and the number of missing readings. Both channels move together, which is what a synchronised joint stream looks like. The channel order is the sorted keys of the first dict seen, or the explicit list passed as `features=[...]`.

## 3. Merging many channels

`RunningMomentsVector.merge` combines two vector summaries channel by channel, with the same exact formulas of step 1.

```python
from dense_armor.core.signals.stats.moments import RunningMomentsVector

a = RunningMomentsVector()
b = RunningMomentsVector()
for q in ([0.1, 0.2], [0.11, 0.19], [0.09, 0.21]):
    a.learn_one({"q0": q[0], "q1": q[1]})
for q in ([0.2, 0.1], [0.21, 0.09]):
    b.learn_one({"q0": q[0], "q1": q[1]})
s = a.merge(b).transform_one({})
print(s["q0"]["count"], round(s["q0"]["mean"], 3), round(s["q1"]["mean"], 3))
```

```
5 0.142 0.158
```

Five samples per channel, and the mean of channel `q0` is $(0.1 + 0.11 + 0.09 + 0.2 + 0.21) / 5 = 0.142$. The two vectors must have the same channels and must both have seen data, otherwise `merge` raises an error.

![Left: the mean and standard deviation of four workers, each with a quarter of 2000 payments, the merged summary and a single accumulator; the merged and the single are on the same values. Right: the running mean and standard deviation of two noisy joint signals followed by RunningMomentsVector.](../../assets/moments/moments_merge.png)

On the left, 2000 payments are split among four workers; each has its own mean and spread, the merged summary has mean `11.9995` and variance `33.6227`, and a single accumulator that saw all the payments agrees to 13 digits. On the right, `RunningMomentsVector` follows two noisy joints at once, one line of mean and one of standard deviation per joint.

## 4. A stream from a robot

A joint stream arrives as readings of the joint positions at a fixed rate. Each reading feeds the moments of every joint in one call.

```python
import numpy as np
from dense_armor.core.online.learn.online_dynamics import write_minimal_urdf
from dense_armor.dynamics.urdf_dynamics import RigidBodyModel
from dense_armor.core.signals.stats.moments import RunningMomentsVector

n = RigidBodyModel(write_minimal_urdf()).n
rng = np.random.default_rng(0)
st = RunningMomentsVector()
for i in range(200):
    q = rng.uniform(-1, 1, n)
    st.learn_one({f"q{j}": float(q[j]) for j in range(n)}, t=i * 0.01)
s = st.transform_one({})["q0"]
print(s["count"], round(s["mean"], 3), round(s["std"], 3))
```

```
200 0.089 0.584
```

The minimal model has two joints, so the stream has the channels `q0` and `q1`. After the loop, `s` is the summary of joint 0 over the whole run: 200 readings, mean 0.089 and standard deviation 0.584, from an object that has been running in fixed memory for the entire stream. The same structure is available for joint 1.

Previous: [RunningMoments](running_moments.md). Next: [EWStats](ewstats.md).

## Details

### References

- Chan, T. F., Golub, G. H., LeVeque, R. J. (1979). *Updating formulae and a pairwise algorithm for computing sample variances.* In Compstat: the exact merge of two partial states.
- Pébay, P. (2008). *Formulas for robust, one-pass parallel computation of covariances and arbitrary-order statistical moments.* Sandia Report SAND2008-6212: equations 1.5 and 1.6 give the pairwise merge of $M_3$ and $M_4$.

### The classes

- `RunningMoments.merge(other)` returns a new `RunningMoments` with the state of the union.
- `RunningMomentsVector(features=None, dtype=None)` is one `RunningMoments` per feature, all updated from the same dict. It subclasses `dense_armor.roles.Transformer` and passes `dense_armor.checks.check_estimator`.

### Numeric results on this page

| Quantity | Value |
|---|---|
| merge of `[1,2,3]` and `[10,20]` | count `5`, mean `7.2`, variance `63.7`, skewness `0.916`, kurtosis `-0.734` |
| the same five samples in one accumulator | the same four numbers |
| two joints, 3 samples | means `0.1` and `0.2` |
| merge of two vectors, 3 + 2 samples | count `5`, means `0.142` and `0.158` |
| 200 readings of joint 0 | `200 0.089 0.584` |
| 2000 payments, four workers merged | mean `11.9995`, variance `33.6227`, same as one accumulator |

### The figure on this page

`docs/assets/moments/moments_merge.png` is generated by `docs/assets/moments/make_moments_pages.py`. The script splits 2000 gamma payments among four `RunningMoments`, merges them and compares them with one accumulator on the whole stream; then it feeds a 6 second, 100 Hz stream of two noisy joint signals into `RunningMomentsVector`.
