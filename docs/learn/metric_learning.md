# Metric learning (online)

A robot arm picks objects. It has seen thousands of grasps: some worked, some did not.
When a new situation arrives, the first thing any retrieval or imitation step needs to
answer is **"which past situations look like this one?"**

"Look like" is a distance. The default answer is Euclidean distance on the raw feature
vector. On real robot features it is often the wrong answer: joint angles wrap around,
velocities matter more than positions for some tasks and less for others, and two
situations can have very different coordinates and still be the *same* grasp. The
distance has to be **learned from data**, not assumed.

`MetricKNNClassifier` learns that distance online, one triplet or pair at a time, while
the robot works. It uses one of three algorithms ported from their papers:

- **OASIS** (Chechik et al. 2009) — bilinear similarity from triplets `(anchor, similar, different)`.
- **LEGO** (Jain et al. 2008) — LogDet-regularized Mahalanobis matrix from pairs with a target distance.
- **POLA** (Shalev-Shwartz et al. 2004) — pseudo-metric from similar / dissimilar pairs, kept positive semi-definite.

## The data

A stream of grasp features, one per attempt. Each sample is a 2-D vector: two joint
torques at the moment of contact. Label: which grasp type succeeded.

```python
import numpy as np
rng = np.random.default_rng(42)
grasps_a = rng.normal([1.0, 0.0], 0.1, (200, 2))
grasps_b = rng.normal([0.0, 1.0], 0.1, (200, 2))
X = np.vstack([grasps_a, grasps_b])
y = np.array(["A"] * 200 + ["B"] * 200)
```

Euclidean distance on this data works, because the clusters are far apart. On a real
robot they are not.

## 1. Learn a distance from triplets: OASIS

OASIS takes three samples at a time: an anchor `xa`, a similar sample `xp`, and a
different sample `xn`. It learns a matrix `W` such that the bilinear similarity

```
s(x, x') = xᵀ W x'
```

is **higher for `(xa, xp)` than for `(xa, xn)` by at least a margin of 1**.

### The formula, symbol by symbol

For an anchor `xa`, a similar `xp`, a different `xn`, define the loss

```
ℓ = max(0, 1 − s(xa, xp) + s(xa, xn))
```

If `ℓ > 0` (the constraint is violated), the update is a single rank-one step:

```
V = xa (xp − xn)ᵀ
τ = min(C, ℓ / ‖V‖²)
W ← W + τ V
```

`C` is the aggressiveness: how large a single step is allowed to be. `V` is the
rank-one direction that increases similarity to `xp` and decreases similarity to `xn`
simultaneously. The step size `τ` is the loss divided by the squared gradient magnitude,
capped at `C`.

### Hand case

Start `W = I`, `C = 0.1`, `xa = (1, 0)`, `xp = (1, 0.2)`, `xn = (0.8, 0)`.

```
s(xa, xp) = xaᵀ W xp = 1
s(xa, xn) = xaᵀ W xn = 0.8
ℓ = max(0, 1 − 1 + 0.8) = 0.8
xp − xn = (0.2, 0.2)
V = [[0.2, 0.2], [0, 0]]
‖V‖² = 0.08
τ = min(0.1, 0.8 / 0.08) = min(0.1, 10) = 0.1
W ← I + 0.1 · [[0.2, 0.2], [0, 0]] = [[1.02, 0.02], [0, 1]]
```

The `(0, 0)` entry of `W` and the off-diagonal entry increased. The next time the same
triplet arrives, `s(xa, xp)` is `1.02`, `s(xa, xn)` is `0.816`: the margin grew.

```python
from dense_armor.utility.learn.metric_learning import OASIS

learner = OASIS(C=0.1)
learner.learn_triplet({"x0": 1.0, "x1": 0.0},
                      {"x0": 1.0, "x1": 0.2},
                      {"x0": 0.8, "x1": 0.0})
```

OASIS's learned `W` is not required to be symmetric. The distance used downstream is
built from `M = WᵀW`, which is always positive semi-definite.

## 2. Learn from pairs with a target distance: LEGO

LEGO takes two samples and a target squared distance `y`. It updates a PSD matrix `A`
toward that target using a LogDet-regularized gradient step.

### The formula, symbol by symbol

For a pair `(u, v)` with squared-distance target `y` and vector `z = u − v`:

```
ŷ = zᵀ A z                    (current squared distance)
a = η y ŷ − 1
ȳ = (a + √(a² + 4η ŷ²)) / (2η ŷ)
A ← A − η (ȳ − y) (A z)(A z)ᵀ / (1 + η (ȳ − y) ŷ)
```

`η` is the learning rate; `ȳ` is the *closed-form* optimal post-update distance, derived
in the paper (Section 2.1). Because `A` is updated through a LogDet-divergence step, it
stays positive definite by construction: no separate projection needed.

### Hand case

Start `A = I`, `z = (1, 0)`, target `y = 0.5`, learning rate `η = 0.1`.

```
ŷ = 1
a = 0.1 · 0.5 · 1 − 1 = −0.95
ȳ = (−0.95 + √(0.9025 + 0.4)) / 0.2 = 0.9564
A z = (1, 0)
(A z)(A z)ᵀ = [[1, 0], [0, 0]]
A ← I − 0.1 · (0.9564 − 0.5) · [[1, 0], [0, 0]] / (1 + 0.1 · (0.9564 − 0.5) · 1)
  = I − 0.04364 · [[1, 0], [0, 0]]
  = [[0.9564, 0], [0, 1]]
```

`A₀₀` moved from 1 to 0.9564 — toward the target 0.5, but only partway, because the
LogDet regularizer resists large single steps. This is what makes LEGO's parameter
updates stable without any tuning of a projection radius.

```python
from dense_armor.utility.learn.metric_learning import LEGO

learner = LEGO(eta=0.1)
learner.learn_pair({"x0": 1.0, "x1": 0.0},
                   {"x0": 0.0, "x1": 0.0},
                   0.5)
```

LEGO needs a target squared distance per pair. `LEGO.percentile_targets` computes the
5th/95th percentiles of same-class / different-class distances, as in Section 4 of the
paper.

## 3. Learn a pseudo-metric from similar/dissimilar pairs: POLA

POLA learns a pseudo-metric from a stream of pairs labelled `y = +1` (similar) or
`y = −1` (different). It keeps a matrix `A` and a threshold `b`; a pair is predicted
similar when the squared Mahalanobis distance `d = zᵀ A z` is below `b`.

### The formula, symbol by symbol

For a pair with `z = u − v` and label `y ∈ {+1, −1}`:

```
d = zᵀ A z
ℓ = max(0, y (d − b) + 1)
α = ℓ / (1 + ‖z‖⁴)
A ← A − y α z zᵀ
b ← b + y α
```

Then `A` is projected back onto the PSD cone. If `A` still has a negative eigenvalue,
it is removed (the paper shows only a single eigenvalue can go negative, so removing one
rank-one direction is enough).

### Hand case

Start `A = I`, `b = 1`, `z = (1, 1)`, label `y = +1`.

```
d = [1, 1] · I · [1, 1]ᵀ = 2
ℓ = max(0, 1 · (2 − 1) + 1) = 2
‖z‖⁴ = (√2)⁴ = 4
α = 2 / (1 + 4) = 0.4
A ← I − 1 · 0.4 · [[1, 1], [1, 1]] = [[0.6, −0.4], [−0.4, 0.6]]
b ← 1 + 0.4 = 1.4
```

The eigenvalues of the new `A` are 0.2 and 1.0 — both positive, no projection needed.
The pair was "different" (label +1 means similar, but the distance was above the
threshold, so the pair was violating), and the update shrank `A` along the direction of
`z` to make that direction's distance smaller next time.

```python
from dense_armor.utility.learn.metric_learning import POLA

learner = POLA()
learner.learn_pair({"x0": 1.0, "x1": 0.0},
                   {"x0": 0.0, "x1": 0.0},
                   1)
```

## 4. The k-NN classifier that learns its metric

A metric learner does not fit river's `Classifier` / `Regressor` / `Transformer` types:
it learns from two or three samples at a time and outputs a distance. `MetricLearner` is
the base class; `MetricKNNClassifier` is a river k-NN classifier that uses the learner
as its distance and — with `learn_metric=True`, the default — trains it online inside
`learn_one`. Each new sample is paired with random samples from the window: a triplet
for OASIS, a similar / dissimilar pair for POLA, a pair with a target distance for LEGO.

```python
from dense_armor.utility.learn.metric_learning import MetricKNNClassifier, OASIS

knn = MetricKNNClassifier(OASIS(C=0.1), n_neighbors=1, window_size=1000)
for i in range(len(X)):
    knn.learn_one({"x0": float(X[i, 0]), "x1": float(X[i, 1])}, y[i])
knn.predict_one({"x0": 0.9, "x1": 0.1})
```

On the running data:

```python
correct = sum(knn.predict_one({"x0": float(X[i, 0]), "x1": float(X[i, 1])}) == y[i]
              for i in range(len(X)))
```

## 5. Results on river streams

`progressive_val_score` (predict, then learn), `StandardScaler` in front, k = 5,
`window_size=1000`, metric learned online with the default parameters:

| Stream | river `KNNClassifier` (Euclidean) | OASIS | LEGO | POLA |
|---|---|---|---|---|
| Phishing (1,250 samples) | 0.8823 | **0.8911 (+0.88)** | 0.8815 (−0.08) | 0.8871 (+0.48) |
| ImageSegments (2,310 samples) | 0.8705 | 0.9082 (+3.77) | 0.9112 (+4.07) | **0.9138 (+4.33)** |

On ImageSegments (7 classes, 18 features) all three learned metrics beat the Euclidean
k-NN by about 4 points; on Phishing the gains are under one point and LEGO is level with
the baseline.

## API reference

::: dense_armor.utility.learn.metric_learning

---

## Details

**OASIS** (Chechik, Sharma, Shalit, Bengio, 2009). W is not symmetric; the distance
uses `M = WᵀW`. The proof of the update rule uses the Sherman–Morrison formula; the
resulting step is exact for the passive-aggressive objective with hinge loss and a
Frobenius regularizer on the parameter matrix.

**POLA** (Shalev-Shwartz, Singer, Ng, 2004). The `1 + ‖z‖⁴` in the denominator is not a
typo: the squared gradient of the hinge loss with respect to `A`, in Frobenius norm,
is `‖z zᵀ‖_F² = ‖z‖⁴`. The `+ 1` is the projection-onto-PSD contribution.

**LEGO** (Jain, Kulis, Dhillon, Grauman, 2008). The `ȳ` formula is equation (2.3) of
the paper, solved by setting `dA/dȳ = 0` on the LogDet objective with a squared loss.
The LogDet step keeps `A` positive definite without projection: this is the difference
between LEGO and POLA in the same online setting, and the reason LEGO has no separate
"enforce PSD" step.

The three algorithms compared as kNN metrics on Iris, Wine, Digits and 20 newsgroups
(including `WᵀW` against the symmetrized `W`) are in Dense-Evolution-Discovery,
[Online Metric Learning: OASIS, LEGO, POLA](https://tatopenn-cell.github.io/Dense-Evolution-Discovery/river_metric_learning/).

**See also**: [Calibration](calibration.md) — if the k-NN above is used with
`predict_proba_one`, calibration turns its raw class frequencies into honest
probabilities.
