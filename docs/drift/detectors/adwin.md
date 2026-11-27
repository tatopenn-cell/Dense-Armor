# ADWIN: a window that cuts itself

[← Drift detectors](index.md)

`ADWIN` keeps the recent values of a stream in a window and, at every step, asks whether an older part of the window and a newer part have the same average. If they do not, it throws the older part away and says that the stream has changed. The window grows while the stream is stable and shrinks when it changes, so you do not have to choose its length.

## First example

```python
import numpy as np
from dense_armor.core.online.drift.adwin import ADWIN

rng = np.random.default_rng(0)
x = np.concatenate([rng.normal(0, 1, 300), rng.normal(4, 1, 300)])
det = ADWIN(delta=0.002, max_window=100)
print([i for i, v in enumerate(x) if det.update(v).drift_detected])
```

```
[327, 328]
```

In this example the first 300 values are normal noise, for instance the daily cost of an order after removing its usual level, and from the value 300 the mean moves up by 4 standard deviations. The detector flags the change at 327, twenty-seven values after it, and again at 328. `drift_detected` is `True` only on the values where a cut was made. One change can give several flags in a row: see "One flag or several" below.

## How it works

The window is not stored as a list of values. It is stored as a short list of buckets: a bucket is a run of consecutive values kept as one unit (its sum, its sum of squares, its length, its smallest and its largest value). Level $i$ holds buckets of $2^i$ values; each level keeps at most `max_buckets` buckets, and when a level has one more, its two oldest buckets are merged into one bucket of the level above. The newest values are in small buckets and the oldest in large ones. With `max_window=None`, the number of buckets that the window holds and its number of levels, measured while the stream goes on:

```python
import numpy as np
from dense_armor.core.online.drift.adwin import ADWIN

rng = np.random.default_rng(0)
det = ADWIN(max_window=None)
for n in (100, 1000, 10000, 100000):
    while det._n < n:
        det.update(rng.normal())
    print(n, sum(len(level) for level in det._buckets), len(det._buckets))
```

```
100 19 5
1000 35 8
10000 49 11
100000 63 15
```

The window holds 100,000 values in 63 buckets: the memory grows with the logarithm of the window, not with its length. The cuts that the detector tries are the borders between buckets, so it tries 63 cuts, not 100,000.

At every value, the detector does the following, in this order:

1. A value that is not a number (`nan`, `inf`) is skipped and counted in `n_missing`.
2. The value enters as a bucket of one value, and the buckets are merged where a level holds too many.
3. If the window is longer than `max_window`, the oldest bucket is dropped.
4. Every `clock` values, for each border between two buckets, the window is split into an older part $W_0$ of $n_0$ values and a newer part $W_1$ of $n_1$ values. If the cut passes the test below, the oldest bucket is dropped, the flag is raised and the test is run again on what is left. It stops when no cut passes.

The test is

$$|\hat\mu_0 - \hat\mu_1| > R\sqrt{\frac{\ln(4/\delta')}{2m}}, \qquad m = \frac{n_0\, n_1}{n_0 + n_1}, \qquad \delta' = \frac{\delta}{\ln n},$$

where $\hat\mu_0$ and $\hat\mu_1$ are the averages of the two parts, $n$ is the length of the window, $\delta$ is the confidence parameter and $R$ is the range of the window, its largest value minus its smallest (it is 1 if all the values are equal). The bound comes from Hoeffding's inequality for the difference of two averages; $m$ grows with the length of the shorter part, so the bar is high for short parts and low for long ones. $R$ carries the units of the data on both sides of the inequality, so the detector does not depend on the units:

```python
import numpy as np
from dense_armor.core.online.drift.adwin import ADWIN

rng = np.random.default_rng(0)
x = np.concatenate([rng.normal(0, 1, 300), rng.normal(4, 1, 300)])
for scale in (1.0, 1e-4, 1e6):
    det = ADWIN(max_window=100)
    print(scale, next(i for i, v in enumerate(x * scale) if det.update(v).drift_detected))
```

```
1.0 327
0.0001 327
1000000.0 327
```

## What it sees, and how fast

How fast the detector sees a step depends on its size and on `max_window`. 20 seeded streams of 300 values of $N(0, 1)$ followed by 300 values of $N(\text{shift}, 1)$; each row is the number of streams where the shift was found and the median delay, in values, after the change:

```python
import numpy as np
from dense_armor.core.online.drift.adwin import ADWIN

def delay(shift, s, **kw):
    r = np.random.default_rng(s)
    y = np.concatenate([r.normal(0, 1, 300), r.normal(shift, 1, 300)])
    det = ADWIN(**kw)
    return next((i - 300 for i, v in enumerate(y) if det.update(v).drift_detected and i >= 300), None)
for mw in (100, 200, None):
    for sh in (2.0, 3.0, 4.0):
        d = [v for v in (delay(sh, s, max_window=mw) for s in range(20)) if v is not None]
        print(mw, sh, len(d), np.median(d) if d else None)
```

```
100 2.0 0 None
100 3.0 0 None
100 4.0 11 27.0
200 2.0 2 100.0
200 3.0 20 47.0
200 4.0 20 24.5
None 2.0 20 83.0
None 3.0 20 42.5
None 4.0 20 24.0
```

With a window of 100 values, a step of 3 standard deviations goes unnoticed (0 streams out of 20) and a step of 4 is found in 11 streams. The default window is 200 (`max_window=200`): a step of 3 is found in all the streams, after 47 values, and a step of 2 in two. With no limit (`max_window=None`) the three steps are found in all the streams, a step of 2 after 83 values. The cause is the shape of the test: the bar for a cut is high when the shorter part is short, so a small step needs a long window to be seen, and `max_window` is the longest window that the detector keeps.

The cases that it does not see, the effect of `delta` and of `clock`, and the bursts of flags are in the Details below.

## The parameters

| Parameter | Default | What it does |
|---|---|---|
| `delta` | 0.002 | confidence, in $(0, 1)$; a larger one is faster and a smaller one is more cautious (0.002 is the value that the KSWIN paper uses for ADWIN) |
| `max_window` | 200 | the longest window that is kept; `None` means no limit, which is the algorithm of the paper |
| `min_sub` | 2 | the shortest part, on each side of a cut, that is considered |
| `max_buckets` | 5 | buckets per level before the two oldest are merged |
| `clock` | 1 | the test runs once every `clock` values; at most `clock - 1` values of delay |

`min_sub` and `max_buckets` change little: with a step of 3 and no limit, `min_sub` = 2 and 10 both give 20 streams found with a median delay of 42.5, and `max_buckets` = 2, 5 and 20 give 43.5, 42.5 and 38.0. A `delta` outside $(0, 1)$, a `clock` below 1 or a `max_window` below `4 * min_sub` raises a `ValueError`: `delta must be in (0, 1), got 2`.

## API reference

::: dense_armor.core.online.drift.adwin

## Details

### Other cases

The confidence parameter `delta` moves the bar. A step of 3 standard deviations, with no limit to the window, 20 streams:

```python
import numpy as np
from dense_armor.core.online.drift.adwin import ADWIN

def delay(shift, s, **kw):
    r = np.random.default_rng(s)
    y = np.concatenate([r.normal(0, 1, 300), r.normal(shift, 1, 300)])
    det = ADWIN(**kw)
    return next((i - 300 for i, v in enumerate(y) if det.update(v).drift_detected and i >= 300), None)
for dl in (0.1, 0.002, 1e-6):
    d = [v for v in (delay(3.0, s, delta=dl, max_window=None) for s in range(20)) if v is not None]
    print(dl, len(d), np.median(d) if d else None)
```

```
0.1 20 18.0
0.002 20 42.5
1e-06 20 88.5
```

A larger `delta` is faster; a smaller one is more cautious. On 50 stationary streams of 2000 values of $N(0, 1)$, with no limit to the window, the detector gave no false alarm with `delta` = 0.002 and none with `delta` = 0.1 (`benchmarks/adwin_threshold.py` prints the same count).

Three cases that it does not see, or sees late:

```python
import numpy as np
from dense_armor.core.online.drift.adwin import ADWIN

r = np.random.default_rng(0)
y = np.concatenate([r.normal(0, 1, 300), r.normal(3, 1, 300)])
for spike in (None, 20.0):
    z = y.copy()
    if spike:
        z[290] = spike
    det = ADWIN(max_window=None)
    print(spike, [i for i, v in enumerate(z) if det.update(v).drift_detected][:3])
```

```
None [336, 337, 338]
20.0 []
```

A single value of 20, ten values before a step of 3 standard deviations, is enough to hide the step: no flag at all in the 300 values after it. The reason is $R$: while the value of 20 is in the window, the range is about 20 and the bar for a cut is about six times higher than with the range of the noise alone. The same happened in the 20 streams (0 found; see the table of the comparison below).

A change of the spread, with the same mean, is not seen:

```python
import numpy as np
from dense_armor.core.online.drift.adwin import ADWIN

r = np.random.default_rng(0)
x = np.concatenate([r.normal(0, 1, 300), r.normal(0, 3, 300)])
det = ADWIN(max_window=None)
print([i for i, v in enumerate(x) if det.update(v).drift_detected])
```

```
[]
```

`ADWIN` compares averages, so a change of the variance leaves it silent (0 streams out of 20). For a slow change, a mean that rises by 4 standard deviations over 300 values:

```python
import numpy as np
from dense_armor.core.online.drift.adwin import ADWIN

r = np.random.default_rng(0)
x = np.concatenate([r.normal(0, 1, 300), r.normal(0, 1, 300) + np.linspace(0, 4, 300)])
det = ADWIN(max_window=None)
print([i for i, v in enumerate(x) if det.update(v).drift_detected][:3])
```

```
[500, 504, 510]
```

The first flag is at 500, two hundred values into the ramp, when the mean has already risen by about 2.7 standard deviations. If all the values of the window are equal, $R$ is taken as 1: a stream of 200 values of 0 followed by values of 1 is flagged at 204.

#### One flag or several

The detector keeps cutting while some cut passes the test, and a change that has only just started leaves part of the old values in the window, so the test passes again a few values later. In the first example the flags are 327 and 328. With `clock` the test runs only every `clock` values:

```python
import numpy as np
from dense_armor.core.online.drift.adwin import ADWIN

r = np.random.default_rng(0)
x = np.concatenate([r.normal(0, 1, 300), r.normal(4, 1, 300)])
for clock in (1, 10):
    det = ADWIN(max_window=100, clock=clock)
    print(clock, [i for i, v in enumerate(x) if det.update(v).drift_detected])
```

```
1 [327, 328]
10 [329]
```

With `clock` = 10 the flag comes two values later and it is a single one. In general `clock` delays the flag by at most `clock - 1` values. Act on the first flag of a burst, or start a new detector after it, to get one event per change; `test/core/online/drift/test_adwin_alarms.py` checks that the burst after a step of 3 standard deviations is short.

### Where it differs from the paper, and whether it is better

The paper (Bifet and Gavaldà, *Learning from time-changing data with adaptive windowing*) defines the rigorous threshold with $\delta' = \delta/n$ and values in $[0, 1]$, and then says that, in practice, it is too conservative: all its implementations use a threshold with the variance of the window (equation 3.1),

$$\varepsilon_\text{cut} = \sqrt{\frac{2}{m}\,\sigma_W^2 \ln\frac{2}{\delta'}} + \frac{2}{3m}\ln\frac{2}{\delta'}, \qquad \delta' = \frac{\delta}{\ln n},$$

where $\sigma_W^2$ is the observed variance of the window; the paper says that $\delta' = \delta/\ln n$ is enough because only $O(\log n)$ cuts are tried. The paper allows any known interval $[a, b]$ by rescaling; `ADWIN` uses the Hoeffding form of the first threshold, with the range $R$ observed in the window in place of a known interval, and with the $\delta' = \delta/\ln n$ of the second. So it has the property of the paper that the test does not use the variance, and the memory and the cuts of the second algorithm of the paper (the buckets and their borders, with `max_buckets` = 5, the value that the paper uses in its experiments).

`benchmarks/adwin_threshold.py` compares the two thresholds on the same buckets, the same cuts and the same streams (20 seeded streams of 300 + 300 values; each cell is the number of streams where the shift was found and the median delay). The "paper" column is the threshold of equation 3.1 with the second term multiplied by $R$, because the streams are not in $[0, 1]$; it is not part of the library.

| `max_window` | shift | library | paper, equation 3.1 |
|---|---|---|---|
| 100 | 2 | 0 / — | 0 / — |
| 100 | 3 | 0 / — | 0 / — |
| 100 | 4 | 11 / 27.0 | 3 / 37.0 |
| 200 | 2 | 2 / 100.0 | 20 / 53.5 |
| 200 | 3 | 20 / 47.0 | 20 / 30.0 |
| 200 | 4 | 20 / 24.5 | 20 / 22.0 |
| none | 2 | 20 / 83.0 | 20 / 41.0 |
| none | 3 | 20 / 42.5 | 20 / 25.5 |
| none | 4 | 20 / 24.0 | 20 / 20.0 |

|  | library | paper, equation 3.1 |
|---|---|---|
| step of 3 after a value of 20 | 0 found | 20 found, delay 78.0 |
| spread from 1 to 3, same mean | 0 found | 0 found |
| false alarms, normal, 50 streams of 2000 | 0 | 0 |
| false alarms, heavy tails (t, 3 degrees), 50 streams of 2000 | 0 | 0 |

On these streams the threshold of the paper is faster with the default window of 200 and with no limit: a step of 2 is seen in all the streams, and a single large value does not hide a step. The threshold of the library is more cautious: it is slower, and a single large value of the stream raises the bar through $R$. Neither gave a false alarm on the stationary streams that were tried, so the extra speed of the paper's threshold did not cost false alarms there. With a window of 100 the library is better: it finds a step of 4 in 11 streams out of 20, the other in 3. The comparison covers these streams and these settings only.

- **Source.** Bifet, A., Gavaldà, R. (2007), *Learning from time-changing data with adaptive windowing*, Proceedings of the 2007 SIAM International Conference on Data Mining, 443-448. The preprint in the library of papers is dated October 2006. The rigorous threshold is in Section 3, Theorem 3.1 states the false positive and false negative bounds, and the buckets (ADWIN2) are in Theorem 3.2: $O(M \log(W/M))$ memory and $O(1)$ amortised time per element.
- **Memory.** The docstring gives $O(\text{max\_buckets} \cdot \log W)$ for the memory, the number of buckets above is measured. An older version of this page said that the whole window was kept and that the cost of an update was $O(W)$: it is no longer the case.
- **Docstring.** The docstring of the module states the differences above, with the command that measures them.
