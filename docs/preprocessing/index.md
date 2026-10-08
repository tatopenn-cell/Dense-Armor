# Preprocessing

Raw sensor numbers come in different units and sizes — a joint position in radians, a torque in
newton-metres, a word in a sentence. Before a model can learn from them they are put on a common
scale, turned into the quantities that matter (speed, power, word weights), filtered, and
balanced. Everything here works one sample at a time: the estimator sees a sample, updates its
own state, and is ready for the next one, without keeping the whole history in memory.

Nine steps follow. The first four are about numbers, the next three about text, the last two
about which features to keep and how to handle rare events.

## 1. Same scale for every feature

A standard scaler subtracts the running mean and divides by the running standard deviation, so
every feature lives around 0 with spread 1.

```python
from dense_armor.utility.preprocessing.scale import StandardScaler

sc = StandardScaler()
for v in [1.0, 2.0, 3.0, 4.0, 5.0]:
    sc.learn_one({"x": v})
print(round(sc.transform_one({"x": 5.0})["x"], 4))
```

```
1.2649
```

The mean of 1, 2, 3, 4, 5 is 3, the standard deviation is $\sqrt{2.5} = 1.581$, so
$(5 - 3) / 1.581 = 1.2649$. In general the scaler computes

$$z = \frac{x - \bar x}{s}$$

where $\bar x$ and $s$ are the running mean and standard deviation seen so far. Both are updated
one sample at a time, so the scaler never needs to store the past values.

## 2. One statistic per joint

A robot arm sends its joints as one vector; each joint gets its own mean and spread.

```python
from dense_armor.utility.preprocessing.scale import StandardScaler

sc = StandardScaler()
for q in ([0.0, 10.0], [1.0, 20.0], [2.0, 30.0]):
    sc.learn_one({"q": q})
print([round(v, 3) for v in sc.transform_one({"q": [2.0, 30.0]})["q"]])
```

```
[1.0, 1.0]
```

Both joints are at the top of their own range, so both get the same standardised value, even
though the second joint moves ten times more. The scaler keeps one mean and one standard
deviation per joint, not one for the whole vector.

## 3. Robust to spikes

The robust scaler uses the rolling median and the median absolute deviation (MAD, times 1.4826
so it matches the standard deviation on Gaussian data); one spike barely moves it.

```python
from dense_armor.utility.preprocessing.scale import RobustScaler, StandardScaler

rs, ss = RobustScaler(window=20), StandardScaler()
for v in [1.0, 2.0, 3.0, 2.0, 1.0, 2.0, 3.0, 2.0, 100.0]:
    rs.learn_one({"x": v})
    ss.learn_one({"x": v})
print(round(rs.transform_one({"x": 3.0})["x"], 3), round(ss.transform_one({"x": 3.0})["x"], 3))
```

```
0.674 -0.303
```

The same value 3.0 after a spike of 100: the robust scaler sees it as 0.674 spreads above the
median (median 2, MAD-based spread 1.483); the standard scaler, pulled by the spike, sees it as below average. The
median is the middle value of the sorted window, so a single extreme sample does not move it;
the MAD is the median of the distances from the median, so it measures the typical spread
without ever squaring the spike. The factor 1.4826 turns the MAD into the same scale as a
standard deviation when the data is roughly Gaussian.

## 4. Speed, acceleration and jerk from positions

The robot reports positions; velocity, acceleration and jerk come from differences between
samples divided by the real time between them, even when the samples are not evenly spaced.

```python
from dense_armor.utility.preprocessing.joints import JointDerivatives

jd = JointDerivatives(order=3)
for t in [0.0, 0.3, 1.1]:
    jd.learn_one({"q": [t ** 3]}, t=t)
out = jd.transform_one({"q": [2.0 ** 3]}, t=2.0)
print([round(out[k][0], 6) for k in ("qd", "qdd", "jerk")])
```

```
[12.0, 12.0, 6.0]
```

Positions follow $q = t^3$ at uneven times 0, 0.3, 1.1, 2.0; at $t = 2$ the exact values are
$3t^2 = 12$, $6t = 12$, $6$: the method is exact on a cubic. It uses the Newton divided
differences of the last four samples; the current sample is included, so the value belongs to
the present moment, not to the previous sample. Divided differences are the natural generalisation
of "difference divided by time" to several unevenly spaced points: they are built recursively so
that the resulting polynomial passes through all the samples, and differentiating it at the last
point gives the derivative there.

## 5. Power

```python
from dense_armor.utility.preprocessing.joints import JointPower

out = JointPower().transform_one({"tau": [1.0, 2.0], "qd": [3.0, 4.0]})
print(out["power"], out["power_total"])
```

```
[3.0, 8.0] 11.0
```

Mechanical power of a joint is torque times angular velocity,

$$P = \tau\,\dot q$$

so the first joint contributes $1 \times 3 = 3$, the second $2 \times 4 = 8$, and the total is
$3 + 8 = 11$.

## 6. Text: words, counts and weights

A sentence becomes a list of words, then counts; TF-IDF weighs each word by how often it
appears in this document and how rare it is across all documents seen.

```python
from dense_armor.utility.preprocessing.text import TFIDF, BagOfWords, Tokenizer

tk, bw, tf = Tokenizer(), BagOfWords(), TFIDF()
docs = ["joint three overheats", "joint three ok", "gripper ok"]
for d in docs:
    tf.learn_one(bw.transform_one(tk.transform_one(d)))
out = tf.transform_one(bw.transform_one(tk.transform_one("joint three overheats")))
print({k: round(v, 3) for k, v in out["scores"].items()})
```

```
{'joint': 0.135, 'three': 0.135, 'overheats': 0.366}
```

The formula is the one of Silajev (2026), section 1:

$$\text{TF-IDF}(t, i) = \frac{d_i(t)}{|d_i|}\,\ln\frac{N}{\text{DF}(t)}$$

$d_i(t)$ is how many times the word appears in the document, $|d_i|$ the document length, $N$
the number of documents seen (3), $\text{DF}(t)$ how many contain the word: "overheats"
appears in one document, so it weighs most; "joint" and "three" appear in two.

The word "overheats" is rare and informative; "ok" appears in two documents out of three, so
its weight is small. The ratio $d_i(t)/|d_i|$ is the frequency of the word inside the
document, and the logarithm of $N/\text{DF}(t)$ rewards words that few documents share. Silajev
(2026) shows that the same formula is exactly a Kullback-Leibler divergence between two
probability models.

`TFIDF()` follows the paper exactly. The option `smooth_idf=True` replaces the logarithm with
$\ln(1 + N/\text{DF})$, a common practical variant that keeps every word with at least a small
positive score; it is off by default because the paper's formula does not include it.

## 7. Text in a fixed size: hashing

```python
from dense_armor.utility.preprocessing.text import FeatureHasher

fh = FeatureHasher(n_features=8, seed=0)
print(fh.transform_one({"counts": {"joint": 1.0, "three": 1.0, "overheats": 1.0}})["hashed"])
```

```
{7: -1.0, 2: 1.0, 5: 1.0}
```

Every word is sent by a hash function to one of 8 slots, with a random sign; the vocabulary
can grow forever, the vector stays the same size. Weinberger et al. (2009), Theorem 3
(eq. 4): for a unit vector $x$ with
$\|x\|_\infty \le \epsilon/(18\sqrt{\log(1/\delta)\log(m/\delta)})$ and
$m \ge 72\log(1/\delta)/\epsilon^2$, the hashed squared length is within $1 \pm \epsilon$ with
probability at least $1 - 2\delta$.

The map used here is the *signed* hash of the paper: each word gets an index and a random
sign, and words that share a slot add their weights with signs that cancel on average. Without
the sign, two words in the same slot would always add up, biasing the sum upward. The bound
above says how many slots are enough: the error decreases with more slots, the confidence
increases with fewer collisions, and the largest single coordinate of the input must not
dominate the vector. In practice eight slots already separate the three words of the example,
each landing in its own bucket.

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

## API reference

::: dense_armor.utility.preprocessing.scale
::: dense_armor.utility.preprocessing.joints
::: dense_armor.utility.preprocessing.text
::: dense_armor.utility.preprocessing.select
::: dense_armor.utility.preprocessing.imbalance

---

## Details

- Scalers keep one statistic per feature and per joint; a feature never seen, or with zero
  spread, passes unchanged.
- `TFIDF(smooth_idf=True)` uses $\ln(1 + N/\text{DF})$; this smoothing is not in Silajev
  (2026) and is off by default.
- `JointDerivatives` keeps the last `order + 1` valid samples; missing samples (NaN) are
  skipped and counted in `n_missing`.
- Sources: Silajev, I. (2026), "TF-IDF and BM25 are exact KL divergences", arXiv:2609.14016.
  Weinberger, K. et al. (2009), "Feature hashing for large scale multitask learning", ICML,
  arXiv:0902.2206. Malialis, K., Panayiotou, C., Polycarpou, M. (2018), "Queue-based
  resampling for online class imbalance learning", ICANN, arXiv:1809.10388.
````

