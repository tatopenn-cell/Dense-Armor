# Clustering

A robot meets situations nobody labelled: idle, reaching, carrying, a joint behaving oddly.
Clustering groups similar samples without labels, one sample at a time, so the robot can
discover its own states and notice a new one. The sections below build up the picture: a
simple online k-means, the same method following groups that move, a denser method that keeps
outliers out and shapes of any kind, and a use of clustering on image patches, a visual
vocabulary.

## 1. Online k-means: k groups, one point at a time

Each new point joins the nearest centre, and that centre moves a little towards it.

```python
from dense_armor.utility.cluster.kmeans import OnlineKMeans

km = OnlineKMeans(k=2, warmup=4)
for v in [0.0, 0.1, 10.0, 10.1, 0.2, 9.9]:
    km.learn_one({"x": [v]})
print(km.predict_one({"x": [0.05]}), km.predict_one({"x": [10.05]}))
print([round(float(c[0]), 3) for c in km.centers_])
```

```
0 1
[0.1, 10.0]
```

Two groups, around 0 and around 10; each centre is the mean of the points that joined it. The
first `warmup` points (here 4, default 10 per centre) are kept aside; the centres are chosen
among them one at a time, each the point farthest from those already chosen (the minimax rule of
Roth et al. 2021, eq. 5); then learning goes on one point at a time. Starting from the first
points instead could put two centres in the same group.

With the points above, the four points 0.0, 0.1, 10.0, 10.1 wait in the buffer. Farthest-first
starts from the buffered point closest to their mean, 0.1, then picks the point farthest from it,
10.1. The other two buffered points join their nearest centre: 0.0 moves the first centre to 0.05,
10.0 moves the second to 10.05. Then 0.2 moves the first centre to 0.1 and 9.9 moves the second
to 10.0, the means of the points in each group. The update rule moves the assigned centre by a
fraction of the distance, one over the number of points that centre already has:
$$c_j \leftarrow c_j + \frac{1}{n_j}\,(x - c_j)$$

where $c_j$ is the centre of group $j$, $x$ is the new point, and $n_j$ is how many points have
already joined group $j$.

## 2. Three groups, compared with the batch method

The online method, which sees each point once, agrees with the true groups as well as the batch
method, which sees all the points at once.

```python
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from dense_armor.utility.cluster.kmeans import OnlineKMeans

rng = np.random.default_rng(0)
centres = np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0]])
y = rng.integers(0, 3, 600)
X = centres[y] + rng.normal(0, 0.5, (600, 2))
km = OnlineKMeans(k=3)
for x in X:
    km.learn_one({"x": x.tolist()})
pred = [km.predict_one({"x": x.tolist()}) for x in X]
batch = KMeans(n_clusters=3, n_init=10, random_state=0).fit_predict(X)
print(round(adjusted_rand_score(y, pred), 4), round(adjusted_rand_score(y, batch), 4))
```

```
1.0 1.0
```

The adjusted Rand index measures how well two groupings agree: 1 is identical, 0 is chance, and
negative is worse than chance. Six hundred points drawn around three well-separated centres,
σ = 0.5, distance 10. The online version sees each point exactly once, in the order it comes;
the batch version sees the whole matrix and runs ten restarts. Both reach 1.0, so the online
method has not lost anything by streaming.

## 3. Following groups that move

With a half-life the step stays constant, $1 - 2^{-1/h}$, so old points fade and the centres
follow a drifting stream.

```python
import numpy as np
from dense_armor.utility.cluster.kmeans import OnlineKMeans

rng = np.random.default_rng(1)
km = OnlineKMeans(k=2, halflife=50)
for i in range(3000):
    shift = i / 3000 * 5.0
    c = [shift, 0.0] if i % 2 == 0 else [shift, 10.0]
    km.learn_one({"x": (np.array(c) + rng.normal(0, 0.3, 2)).tolist()})
print(sorted([round(float(c[0]), 2), round(float(c[1]), 2)] for c in km.centers_))
```

```
[[4.78, 9.99], [4.81, -0.01]]
```

Both groups slid from $x = 0$ to $x = 5$ over 3000 steps, and each point of the stream belongs
to a group whose position keeps changing. With a `halflife` the step is a constant
$1 - 2^{-1/h}$ instead of $1 / n_j$, so the influence of a point is halved after `halflife`
samples and the centre follows the group rather than averaging over its whole history. The
final centres end near $x = 5$, where the groups are now.

## 4. Groups of any shape, with outliers: DenStream

DenStream keeps small "micro-clusters" (a centre and a weight), lets their weight fade with
time, and only groups that keep receiving points become real clusters; an isolated point stays
an outlier.

```python
import numpy as np
from dense_armor.utility.cluster.denstream import DenStream

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

## 5. A visual vocabulary

An image is cut into small patches; each patch is described by the directions of its edges;
online k-means groups the patch descriptions into "visual words"; a frame becomes a histogram
of the words it contains.

```python
import numpy as np
from dense_armor.utility.vision.vocabulary import BagOfVisualWords

vert = np.zeros((32, 32), dtype=np.float32)
vert[:, ::4] = 1.0
horz = vert.T.copy()
bow = BagOfVisualWords(n_words=4, patch_size=8, n_bins=4)
for _ in range(10):
    bow.learn_one({"frame": vert})
    bow.learn_one({"frame": horz})
print(bow.transform_one({"frame": vert})["hist"], bow.transform_one({"frame": horz})["hist"])
```

```
[0.0, 0.75, 0.0, 0.25] [0.25, 0.0, 0.75, 0.0]
```

Vertical stripes and horizontal stripes use different words, so their histograms do not overlap:
two textures, two signatures. The vocabulary is learned online with `OnlineKMeans` on the
descriptors of the 8×8 patches, so the same estimator can be fed patches from any number of
frames without ever storing them; a frame becomes a fixed-length histogram, a signature ready
for a classifier.

## API reference

::: dense_armor.utility.cluster.kmeans
::: dense_armor.utility.cluster.denstream
::: dense_armor.utility.vision.vocabulary

---

## Details

- `OnlineKMeans` buffers the first `warmup` points (default `10 * k`) and picks the centres by
  greedy farthest-first selection (Roth et al. 2021, eq. 5). Bhattacharjee et al. (2021) compare
  the online loss $\sum_t d(x_t, S_{t-1})^2$ with the best `k` fixed centres in hindsight; their
  lower bound (Theorem 2, p. 4) applies to any online method, their upper bound (Theorem 1, p. 3)
  needs a randomised algorithm with more than `k` centres and does not apply to this simple
  scheme.
- `DenStream`: an outlier micro-cluster is removed when its weight falls below $\beta\mu/4$
  (the review gives no number); two micro-clusters are reachable when their centres are within
  `eps` (the review, p. 9, uses the sum of the radii; each radius is at most `eps`).
- Sources: Zubaroğlu, A., Atalay, V. (2020), "Data stream clustering: a review", Artificial
  Intelligence Review, arXiv:2007.10781. Bhattacharjee, R., Dasgupta, S., Imola, J. J.,
  Moshkovitz, M. (2021), "Online k-means clustering on arbitrary data streams",
  arXiv:2102.09101. Huang, C., Huang, J. (2017), "A fast HOG descriptor using lookup table and
  integral image", arXiv:1703.06256. Roth, K. et al. (2021), "Towards total recall in
  industrial anomaly detection", arXiv:2106.08265.
````
