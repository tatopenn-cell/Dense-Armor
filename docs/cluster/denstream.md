# DenStream: groups of any shape, with outliers

`DenStream` keeps small micro-clusters, a centre and a weight, lets their weight fade with time, and gives the final groups on demand with `macro_clusters()`. The overview of the section is on [Clustering](index.md).

## Groups of any shape, with outliers

DenStream keeps small "micro-clusters" (a centre and a weight), lets their weight fade with
time, and only groups that keep receiving points become real clusters; an isolated point stays
an outlier.

```python
import numpy as np
from dense_armor.core.online.cluster.denstream import DenStream

rng = np.random.default_rng(0)
ds = DenStream(eps=1.0, beta=0.4, mu=3.0, decay=0.01)
centres = np.array([[0.0, 0.0], [8.0, 0.0], [0.0, 8.0]])
t = 0.0
for i in range(600):
    x = centres[i % 3] + rng.normal(0, 0.3, 2) if i % 20 else rng.uniform(-4, 12, 2)
    ds.learn_one({"x": x.tolist()}, t=t)
    t += 1.0
clusters, _ = ds.macro_clusters()
print(len(clusters), [[round(v, 1) for v in c["center"]] for c in clusters])
```

```
3 [[8.0, -0.0], [-0.0, 8.0], [-0.1, 0.0]]
```

Six hundred points around three centres, one in twenty uniform noise; DenStream finds the three
groups. The clusters come back in the order they were discovered, not in the order of the input
centres, so the first one in the list is the group around (8, 0). The weight of a micro-cluster
fades with the damped window of the review of Zubaroğlu and Atalay (2020, section 2.3.1):

$$f(t) = 2^{-\lambda t}$$

$\lambda$ is the decay rate; a micro-cluster heavier than $\beta\mu$ is a real (potential) one,
lighter ones are candidates (outliers) until more points arrive. An isolated point that lands
far from every micro-cluster only creates a candidate, which will be pruned later if no other
point comes near it. The offline step `macro_clusters()` merges reachable real micro-clusters on
demand, and the centre of each resulting cluster is the weight-weighted mean of its micro-
clusters.

## The parameters

| Parameter | Meaning | Default |
|---|---|---|
| `eps` | neighbourhood radius | 0.5 |
| `beta` | multiplier that separates potential micro-clusters from outlier ones, in `(0, 1]` | 0.2 |
| `mu` | minimum weight of a potential micro-cluster | 1.0 |
| `decay` | decay rate `λ` of the damped window | 0.1 |

## API reference

::: dense_armor.core.online.cluster.denstream

---

## Details

- `DenStream`: an outlier micro-cluster is removed when its weight falls below $\beta\mu/4$
  (the review gives no number); two micro-clusters are reachable when their centres are within
  `eps` (the review, p. 9, uses the sum of the radii; each radius is at most `eps`).
- Sources: Zubaroğlu, A., Atalay, V. (2020), "Data stream clustering: a review", Artificial Intelligence Review, arXiv:2007.10781.
