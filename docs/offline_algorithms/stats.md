# Classic and online stats: what is the difference?

A statistic such as a mean, a variance or a correlation can be computed in two ways. The classic way reads the whole table at once. The online way reads one row at a time and keeps a few numbers up to date. This page puts the two side by side for the statistics of the library, and for the sketches that answer a question about a stream (a quantile, a count, a membership) in a few kilobytes: the classic side of a sketch is the exact answer that the sketch approximates.

## 1. Classic: the algorithm of the paper on the whole dataset

Each class reads the whole dataset with `fit` and leaves its results in attributes with a trailing underscore.

### `ClassicRunningCovariance`

Sample covariance (``n - 1``) and Pearson correlation of two channels.

~~~python
from dense_armor.offline.stats import ClassicRunningCovariance
m = ClassicRunningCovariance().fit([1.0, 2.0, 3.0], [2.0, 4.0, 6.0])
print(round(m.cov_, 6), round(m.corr_, 6))
~~~

~~~
2.0 1.0
~~~

### `ClassicRunningCorrelation`

Pearson correlation of two channels over the whole dataset.

~~~python
from dense_armor.offline.stats import ClassicRunningCorrelation
print(ClassicRunningCorrelation().fit([1.0, 2.0, 3.0], [3.0, 2.0, 1.0]).corr_)
~~~

~~~
-1.0
~~~

### `ClassicRollingCovariance`

Sample covariance over the last ``window`` pairs and over every window.

~~~python
from dense_armor.offline.stats import ClassicRollingCovariance
m = ClassicRollingCovariance(window=3).fit([1., 2., 3., 4.], [1., 2., 3., 5.])
print(m.cov_series_.shape)
~~~

~~~
(2,)
~~~

### `ClassicRollingCorrelation`

Pearson correlation over the last ``window`` pairs and over every window.

~~~python
from dense_armor.offline.stats import ClassicRollingCorrelation
m = ClassicRollingCorrelation(window=3).fit([1., 2., 3.], [1., 2., 3.])
print(round(m.corr_, 6))
~~~

~~~
1.0
~~~

### `ClassicAutocorrelation`

Sample autocorrelation at a fixed lag: the standard estimator.

~~~python
from dense_armor.offline.stats import ClassicAutocorrelation
print(round(ClassicAutocorrelation(lag=1).fit([1., 2., 1., 2., 1.]).value_, 4))
~~~

~~~
-0.8
~~~

### `ClassicRunningMoments`

Mean, variance (``n - 1``), skewness, excess kurtosis, min, max, range.

~~~python
from dense_armor.offline.stats import ClassicRunningMoments
m = ClassicRunningMoments().fit([1.0, 2.0, 3.0, 4.0])
print(m.mean_, round(m.var_, 4), m.ptp_)
~~~

~~~
2.5 1.6667 3.0
~~~

### `ClassicRunningMomentsVector`

The moments of every column of a matrix.

~~~python
from dense_armor.offline.stats import ClassicRunningMomentsVector
m = ClassicRunningMomentsVector().fit([[1., 10.], [3., 20.]])
print(m.mean_.tolist())
~~~

~~~
[2.0, 15.0]
~~~

### `ClassicEWStats`

Exponentially weighted mean and variance of the whole series.

~~~python
from dense_armor.offline.stats import ClassicEWStats
print(round(ClassicEWStats(alpha=0.5).fit([1.0, 3.0]).mean_, 4))
~~~

~~~
2.0
~~~

### `ClassicDDSketch`

Exact quantiles: what DDSketch (Masson et al. 2019) approximates.

~~~python
from dense_armor.offline.stats import ClassicDDSketch
print(ClassicDDSketch().fit([1.0, 2.0, 3.0]).quantile(0.5))
~~~

~~~
2.0
~~~

### `ClassicTDigest`

Exact quantiles: what the t-digest (Dunning 2019) approximates.

~~~python
from dense_armor.offline.stats import ClassicTDigest
print(ClassicTDigest().fit([1.0, 2.0, 3.0, 4.0]).quantile(0.5))
~~~

~~~
2.0
~~~

### `ClassicRollingMedianVector`

Median over every window and over the last window, per column.

~~~python
from dense_armor.offline.stats import ClassicRollingMedianVector
m = ClassicRollingMedianVector(window=3).fit([1., 5., 2., 8.])
print(m.median_.tolist())
~~~

~~~
[5.0]
~~~

### `ClassicOnlineShrinkageCovariance`

Linear shrinkage toward ``(1/p) I`` on the sample covariance of all rows.

~~~python
from dense_armor.offline.stats import ClassicOnlineShrinkageCovariance
import numpy as np
X = np.random.default_rng(0).normal(size=(30, 3))
print(ClassicOnlineShrinkageCovariance(3).fit(X).shrunk_.shape)
~~~

~~~
(3, 3)
~~~

### `ClassicCountMinSketch`

Exact item counts: what the Count-Min sketch approximates.

~~~python
from dense_armor.offline.stats import ClassicCountMinSketch
print(ClassicCountMinSketch().fit(["a", "b", "a"]).estimate("a"))
~~~

~~~
2
~~~

### `ClassicHyperLogLog`

Exact number of distinct items: what HyperLogLog approximates.

~~~python
from dense_armor.offline.stats import ClassicHyperLogLog
print(ClassicHyperLogLog().fit(["a", "b", "a"]).count_distinct_)
~~~

~~~
2
~~~

### `ClassicBloomFilter`

Exact set membership: what the Bloom filter approximates.

~~~python
from dense_armor.offline.stats import ClassicBloomFilter
print(ClassicBloomFilter().fit(["a"]).check("b"))
~~~

~~~
False
~~~

### `ClassicSpaceSaving`

Exact most frequent items: what Space-Saving approximates.

~~~python
from dense_armor.offline.stats import ClassicSpaceSaving
print(ClassicSpaceSaving(k=1).fit(["a", "b", "a"]).top())
~~~

~~~
[('a', 2)]
~~~

## 2. Online: one sample at a time

The same quantity from `learn_one`; the class is the one of `dense_armor.utility`.

### `RunningCovariance`

Incremental covariance of two channels.

~~~python
from dense_armor.utility.stats.dependence import RunningCovariance
from dense_armor.utility.stats.dependence import RunningCovariance
c = RunningCovariance("a", "b")
for a, b in [(1.0, 2.0), (2.0, 4.0), (3.0, 6.0)]:
    _ = c.learn_one({"a": a, "b": b})
print(round(c.cov, 6), round(c.corr, 6))
~~~

~~~
2.0 1.0
~~~

### `RunningCorrelation`

Incremental Pearson correlation between two channels.

~~~python
from dense_armor.utility.stats.dependence import RunningCorrelation
from dense_armor.utility.stats.dependence import RunningCorrelation
r = RunningCorrelation("a", "b")
for a, b in [(1.0, -1.0), (2.0, -2.0), (3.0, -3.0)]:
    _ = r.learn_one({"a": a, "b": b})
print(round(r.corr, 6))
~~~

~~~
-1.0
~~~

### `RollingCovariance`

Sample covariance of two channels over a bounded window.

~~~python
from dense_armor.utility.stats.dependence import RollingCovariance
from dense_armor.utility.stats.dependence import RollingCovariance
c = RollingCovariance("a", "b", window=3)
for a, b in [(1.0, 1.0), (2.0, 2.0), (3.0, 3.0)]:
    _ = c.learn_one({"a": a, "b": b})
print(round(c.corr, 6))
~~~

~~~
1.0
~~~

### `RollingCorrelation`

Pearson correlation of two channels over a bounded window.

~~~python
from dense_armor.utility.stats.dependence import RollingCorrelation
from dense_armor.utility.stats.dependence import RollingCorrelation
r = RollingCorrelation("a", "b", window=3)
for a, b in [(1.0, -1.0), (2.0, -2.0), (3.0, -3.0)]:
    _ = r.learn_one({"a": a, "b": b})
print(round(r.corr, 6))
~~~

~~~
-1.0
~~~

### `Autocorrelation`

Running autocorrelation of a single channel at a fixed lag.

~~~python
from dense_armor.utility.stats.dependence import Autocorrelation
from dense_armor.utility.stats.dependence import Autocorrelation
import numpy as np
ac = Autocorrelation("x", lag=1)
rng = np.random.default_rng(0)
for v in rng.normal(0.0, 1.0, 2000):
    _ = ac.learn_one({"x": float(v)})
print(abs(ac.value) < 0.1)
~~~

~~~
True
~~~

### `RunningMoments`

Incremental mean, variance, skewness, kurtosis for one feature.

~~~python
from dense_armor.utility.stats.moments import RunningMoments
from dense_armor.utility.stats.moments import RunningMoments
m = RunningMoments()
for v in [1.0, 2.0, 3.0, 4.0, 5.0]:
    _ = m.learn_one({"x": v})
print(round(m.mean, 6), round(m.var, 6))
print(round(m.skewness, 6))
~~~

~~~
3.0 2.5
0.0
~~~

### `RunningMomentsVector`

One ``RunningMoments`` per channel, updated in a single call.

~~~python
from dense_armor.utility.stats.moments import RunningMomentsVector
from dense_armor.utility.stats.moments import RunningMomentsVector
v = RunningMomentsVector()
for q in ([0.1, 0.2], [0.11, 0.19], [0.09, 0.21]):
    _ = v.learn_one({"q0": q[0], "q1": q[1]})
print({k: round(s["mean"], 4) for k, s in v.transform_one({}).items()})
~~~

~~~
{'q0': 0.1, 'q1': 0.2}
~~~

### `EWStats`

Exponentially weighted mean and variance, one sample at a time.

~~~python
from dense_armor.utility.stats.moments import EWStats
from dense_armor.utility.stats.moments import EWStats
e = EWStats(alpha=0.5)
for v in [1.0, 2.0, 3.0, 4.0, 5.0]:
    _ = e.learn_one({"x": v})
print(round(e.mean, 6))
~~~

~~~
4.0625
~~~

### `DDSketch`

Relative-error quantile sketch (Masson, Rim, Lee 2019).

~~~python
from dense_armor.utility.stats.quantiles import DDSketch
from dense_armor.utility.stats.quantiles import DDSketch
import numpy as np
s = DDSketch(alpha=0.01)
rng = np.random.default_rng(0)
for v in rng.normal(100.0, 1.0, 2000):
    _ = s.learn_one({"x": float(v)})
print(99.0 < s.quantile(0.5) < 101.0)
~~~

~~~
True
~~~

### `TDigest`

Clustered sketch with small clusters near the tails (Dunning 2019).

~~~python
from dense_armor.utility.stats.quantiles import TDigest
from dense_armor.utility.stats.quantiles import TDigest
import numpy as np
t = TDigest(delta=100)
rng = np.random.default_rng(0)
for v in rng.normal(0.0, 1.0, 5000):
    _ = t.learn_one({"x": float(v)})
print(abs(t.quantile(0.5)) < 0.1)
~~~

~~~
True
~~~

### `RollingMedianVector`

One :class:`RollingMedian` per channel, one update call.

~~~python
from dense_armor.utility.stats.robust import RollingMedianVector
from dense_armor.utility.stats.robust import RollingMedianVector
v = RollingMedianVector(window=3)
for q in ([0.1, 0.2], [0.12, 0.18], [0.11, 0.21], [0.09, 0.19]):
    _ = v.learn_one({"q0": q[0], "q1": q[1]})
print({k: round(s["value"], 4) for k, s in v.transform_one({}).items()})
~~~

~~~
{'q0': 0.11, 'q1': 0.19}
~~~

### `OnlineShrinkageCovariance`

Online covariance with linear shrinkage toward ``(1/p) I``.

~~~python
from dense_armor.utility.stats.shrinkage import OnlineShrinkageCovariance
~~~

~~~

~~~

### `CountMinSketch`

Count-Min frequency sketch (Cormode & Muthukrishnan 2005).

~~~python
from dense_armor.utility.stats.sketches import CountMinSketch
from dense_armor.utility.stats.sketches import CountMinSketch
cm = CountMinSketch(epsilon=0.001, delta=0.001)
for w in ["a", "b", "a", "c", "a", "b", "a"]:
    _ = cm.learn_one({"w": w})
print(cm.estimate("a") >= 4)
~~~

~~~
True
~~~

### `HyperLogLog`

HyperLogLog distinct-count sketch (Flajolet et al. 2007).

~~~python
from dense_armor.utility.stats.sketches import HyperLogLog
from dense_armor.utility.stats.sketches import HyperLogLog
h = HyperLogLog(precision=10)
for i in range(1000):
    _ = h.learn_one({"w": f"item{i}"})
print(abs(h.count() - 1000) / 1000 < 0.1)
~~~

~~~
True
~~~

### `BloomFilter`

Bloom membership filter (Bloom 1970).

~~~python
from dense_armor.utility.stats.sketches import BloomFilter
from dense_armor.utility.stats.sketches import BloomFilter
b = BloomFilter(n_expected=1000, p_false=0.01)
for w in ["a", "b", "c"]:
    _ = b.learn_one({"w": w})
print(b.check("a"), b.check("b"), b.check("c"))
print(b.check("d") in (True, False))
~~~

~~~
True True True
True
~~~

### `SpaceSaving`

Space-Saving heavy hitters (Metwally et al. 2005).

~~~python
from dense_armor.utility.stats.sketches import SpaceSaving
from dense_armor.utility.stats.sketches import SpaceSaving
s = SpaceSaving(k=3)
for w in ["a"] * 10 + ["b"] * 5 + ["c"] * 2 + ["d"] * 1:
    _ = s.learn_one({"w": w})
print(s.top(1)[0][0])
~~~

~~~
a
~~~

## 3. The difference, on synthetic data

10 seeds. `abs` is the largest absolute difference between the two outputs, `rel` the largest relative difference. Median, 90th percentile and maximum over the seeds; the data are `synthetic`.

| class | kind | median | p90 | max | why |
|---|---|---|---|---|---|
| `RunningCovariance` | abs | 4.44e-16 | 8.99e-16 | 9.99e-16 |  |
| `RunningCorrelation` | abs | 3.05e-16 | 5.77e-16 | 7.77e-16 |  |
| `RollingCovariance` | abs | 0 | 2.22e-16 | 2.22e-16 |  |
| `RollingCorrelation` | abs | 5.55e-17 | 1.17e-16 | 1.67e-16 |  |
| `Autocorrelation` | abs | 0.0128 | 0.0251 | 0.0296 | the online estimator differs from the textbook sample autocorrelation |
| `RunningMoments` | abs | 8.88e-15 | 1.08e-14 | 1.24e-14 |  |
| `RunningMomentsVector` | abs | 9.99e-16 | 1.42e-15 | 2.22e-15 |  |
| `EWStats` | abs | 5e-16 | 1.13e-15 | 1.33e-15 |  |
| `DDSketch` | rel | 0.00811 | 0.00943 | 0.00962 | the sketch is within its relative error alpha of the exact quantile |
| `TDigest` | rel | 0.113 | 0.355 | 0.411 | the digest is within its rank error of the exact quantile |
| `RollingMedianVector` | abs | 0 | 0 | 0 |  |
| `OnlineShrinkageCovariance` | abs | 2.66e-15 | 7.46e-15 | 1.07e-14 |  |
| `CountMinSketch` | rel | 1 | 1.1 | 2 | the sketch never undercounts and overcounts within epsilon * n |
| `HyperLogLog` | rel | 0.00247 | 0.00578 | 0.00873 | the standard error of the sketch is 1.04 / sqrt(2^precision) |
| `BloomFilter` | abs | 0 | 0 | 0 | the filter has false positives at rate p_false and no false negatives |
| `SpaceSaving` | rel | 1 | 2.3 | 2.72 | the counts of the kept items are upper bounds of the exact counts |

Wilcoxon and A12: not computed: the classic and the online value are two outputs of the same formula on the same data, so the table gives their difference and not a ranking of the two.

## Details

### References

- `ClassicOnlineShrinkageCovariance`: Linear shrinkage toward ``(1/p) I`` on the sample covariance of all rows. Eq. (4.3) and (4.4), page 12 of Bodnar, Gupta and Parolya (2014): ``alpha = 1 - (tr(S)^2 / n) / (tr(S^2) - tr(S)^2 / p)`` and ``beta = (1 - alpha) tr(S)``; the shrunk matrix is ``alpha S + beta (1/p) I``. Args: n_features: num

Pearson correlation and the sample covariance with ``n - 1`` (textbook).
    West, D. H. D. (1979). Updating mean and variance estimates, CACM 22(9):
        the exponentially weighted recursion (not in the folder,
        bibliographic only).
    Bodnar, T., Gupta, A. K., Parolya, N. (2014). Optimal linear shrinkage
        estimator for large dimensional precision matrix, Eq. (4.3) and (4.4),
        page 12 (not in the folder, bibliographic only).
    Masson, C., Rim, J. E., Lee, H. K. (2019). DDSketch; Dunning, T. (2019).
        Computing extremely accurate quantiles using t-digests; Cormode, G.,
        Muthukrishnan, S. (2005); Flajolet, P. et al. (2007); Bloom, B. H.
        (1970); Metwally, A. et al. (2005): the sketches, whose exact
        counterpart is the classic class (not in the folder, bibliographic
        only).

### Choices stated explicitly

The classic class of an online sketch is the exact value the sketch approximates; the classic class of a statistic is the textbook formula on the whole array. Both are `our choice`, because the online classes name no batch algorithm.

### Development diary

The classic classes are new; the online classes are unchanged. The numbers of section 3 come from `benchmarks/offline/stats.py`.
