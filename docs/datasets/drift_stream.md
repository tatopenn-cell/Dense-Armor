# DriftStream: a stream whose concept changes

`DriftStream` makes a binary stream whose rule, the concept, is replaced at known places. The overview of the section is on [Datasets](index.md).

To test a drift detector you need a stream whose rule changes at a known place. This binary
stream changes its concept once, half way, gradually: near the change each sample comes from the
old or the new concept with a probability that follows a sigmoid.

```python
from dense_armor.core.signals.datasets import DriftStream

ds = DriftStream(n_samples=20000, n_features=2, n_drifts=1, kind="gradual", spacing=20.0, seed=0)
ds.data()
c = ds.concepts_
print([round(float(c[a:a + 1000].mean()), 2) for a in range(7000, 13000, 1000)])
```

```
[0.07, 0.22, 0.42, 0.61, 0.82, 0.92]
```

A *concept* is the rule that ties the input features to the class. In this generator each
concept is two Gaussian clusters, one per class, in a random direction; the paper's own
`StreamGenerator` builds them from `make_classification` and the Madelon rules (section 3,
page 7). The stream visits one concept, then the next; the change sits at sample 10000. Each
number in the output is the share of samples from the new concept in a window of 1000, from
sample 7000 to 13000; it rises smoothly through the change (0.42 just before, 0.61 just
after).

The weight of the new concept at position $p \in [0, 1]$ of the stream is the sigmoid that
Ksieniewicz and Zyblewski (2020) introduce for all three drift types in section 3.2.1
(page 8). The paper does not give its closed form; the generator uses the standard logistic
form,

$$s(p) = \frac{1}{1 + e^{-\text{spacing}\,(p - d)}}$$

with $p$ the position of the sample as a fraction of the stream, $d = 0.5$ the position of the
drift (half way) and `spacing` how sharp the change is. At $p = d$ the sigmoid is exactly $0.5$;
with `spacing` $= 20$ it goes from $0.12$ at $p = 0.4$ to $0.88$ at $p = 0.6$. The same sigmoid
drives the three drift types of the paper (section 3.2, pages 8–9): `"sudden"` is the limit for
a large `spacing` and swaps the concept at once, `"gradual"` draws the old or the new concept at
each sample with probability `s(p)`, `"incremental"` moves the clusters continuously with the
same weight.

## API reference

::: dense_armor.core.signals.datasets.drift_stream

---

## Details

- `DriftStream`: each concept is two Gaussian clusters, one per class, placed along a random
  direction; the class of each cluster alternates between concepts, and each concept draws its own random
  direction, so a model trained on one concept is wrong on the next only when the new direction lies
  on the same side as the old one: on 20 seeds, with one drift and 2000 samples, a linear rule fitted
  before the change had an accuracy below 0.5 after it on 9 seeds with 2 features and on 7 with 4
  features, and above 0.9 on 10 and on 11. Each sample uses the drift nearest to its position, so every
  transition is continuous on both sides. `"sudden"` is the limit of the sigmoid for a large
  `spacing` and does not use it; in `"incremental"` the label follows the concept with the larger
  weight. Implemented from the description in Ksieniewicz and Zyblewski (2020), sections 3.2.1
  to 3.2.3 (pages 8–9).
- Source: Ksieniewicz, P., Zyblewski, P. (2020), "stream-learn — open-source Python library for
  difficult data stream batch analysis", arXiv:2001.11077.
