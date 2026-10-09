# Trees that grow while the robot works

A decision tree answers with a chain of questions ("is the joint torque above 3.5 N·m?"), so a
person can read why it decided. Trained in a batch, a tree needs all the data at once. The trees
on this page grow one sample at a time, while the robot works: they split a leaf only when
enough samples show that the split is real, they can change their mind when the robot's
behaviour changes, and one family also says how sure it is.

## 1. A tree that learns a robot state

Two states of an arm: idle (0) and carrying a load (1). The load raises the joint torque
`tau`; the velocity `qd` says nothing about it.

```python
import numpy as np
from dense_armor.utility.tree import HoeffdingTreeClassifier

rng = np.random.default_rng(0)
ht = HoeffdingTreeClassifier(grace_period=100)
for _ in range(2000):
    y = int(rng.random() < 0.5)
    x = {"tau": rng.normal(2.0 + 3.0 * y, 0.5), "qd": rng.normal(0.0, 1.0)}
    ht.learn_one(x, y)
print(ht.predict_one({"tau": 1.8, "qd": 0.3}), ht.predict_one({"tau": 5.2, "qd": -0.4}))
print(ht.explain_one({"tau": 5.2, "qd": -0.4}))
```

```
0 1
[{'feature': 'tau', 'threshold': 3.5024219539142933, 'side': 'right'}]
```

The tree split on `tau` at 3.50, halfway between the two states (2.0 and 5.0), and ignored
`qd`. `explain_one` gives the path of questions for one sample. When to split: every
`grace_period` samples a leaf compares the two best splits; it splits when the gain of the best
exceeds the second by more than the Hoeffding bound (Manapragada et al. 2018, eq. 1):

$$\epsilon = \sqrt{\frac{R^2 \ln(1/\delta)}{2n}}$$

$R$ is the range of the gain ($\log_2$ of the number of classes), $n$ the samples at the leaf,
$\delta$ the probability of choosing the wrong split. With two classes ($R = 1$),
$\delta = 10^{-7}$ and $n = 200$, $\epsilon = 0.20$: the best split must win by 0.20 bits.

## 2. Changing a split that is no longer the best

At first variable `a` tells the state; after 3000 samples `a` stops mattering and `b` does. The
Hoeffding tree never revisits a split; the Hoeffding Anytime Tree re-checks every internal node
every `grace_period` samples and replaces its split when another one wins by the bound.

```python
import numpy as np
from dense_armor.utility.tree import HoeffdingAnytimeTreeClassifier, HoeffdingTreeClassifier

rng = np.random.default_rng(2)
ht = HoeffdingTreeClassifier(grace_period=100)
ef = HoeffdingAnytimeTreeClassifier(grace_period=100)
for i in range(12000):
    y = int(rng.random() < 0.5)
    a = rng.normal(3.0 * y if i < 3000 else 0.0, 0.5)
    b = rng.normal(3.0 * y if i >= 3000 else 0.0, 0.5)
    ht.learn_one({"a": a, "b": b}, y)
    ef.learn_one({"a": a, "b": b}, y)
print(ht.root_.split_feature_, ef.root_.split_feature_, ef.n_replaced_)
```

```
a b 2
```

The Hoeffding tree keeps its first question on `a` forever; the Anytime tree moved its root to
`b` (2 splits replaced in total). It changes its mind only once the statistics of all the
samples it has seen favour the new split, so it is slow after a sudden change: the next step is
faster.

## 3. Following a sudden change: the adaptive tree

After 3000 samples the meaning of the torque flips (the load is now the low-torque state). The
same split on `tau` still separates the two states, but the answers are swapped. The adaptive
tree watches its own error with a drift detector (`ADWIN`) on each node; when the error jumps,
it grows a second subtree in the background and swaps it in when that subtree is clearly more
accurate.

```python
import numpy as np
from dense_armor.utility.drift.adwin import ADWIN
from dense_armor.utility.metrics import Accuracy
from dense_armor.utility.tree import HoeffdingAdaptiveTreeClassifier, HoeffdingTreeClassifier

rng = np.random.default_rng(1)
ht = HoeffdingTreeClassifier(grace_period=100)
hat = HoeffdingAdaptiveTreeClassifier(grace_period=100, drift_detector=ADWIN())
acc_ht, acc_hat = Accuracy(), Accuracy()
for i in range(6000):
    y = int(rng.random() < 0.5)
    c = 3.0 * y if i < 3000 else 3.0 * (1 - y)
    x = {"tau": rng.normal(2.0 + c, 0.5), "qd": rng.normal(0.0, 1.0)}
    if i >= 3000:
        acc_ht.update(y, ht.predict_one(x))
        acc_hat.update(y, hat.predict_one(x))
    ht.learn_one(x, y)
    hat.learn_one(x, y)
print(round(acc_ht.get(), 3), round(acc_hat.get(), 3), hat.n_drifts_, hat.n_swaps_)
```

```
0.056 0.95 3 1
```

Over the 3000 samples after the change, the plain tree is right 5.6 % of the time (it keeps the
old answers); the adaptive tree detected the change (3 drift signals), swapped in its background
subtree once, and is right 95 % of the time. The swap test compares the errors of the two
subtrees counted on the same samples since the background one started, and swaps when the
difference exceeds $\sqrt{\ln(2/\delta)/n}$ after at least `kappa_alt` samples.

## 4. Predicting a number: the torque of a two-link arm

A regression tree that splits on the gradient of the loss (stochastic gradient tree): it
predicts the gravity torque of the first joint of a two-link arm from the two joint angles.

```python
import numpy as np
from dense_armor.utility.metrics import MeanAbsoluteError
from dense_armor.utility.tree import SGTRegressor

rng = np.random.default_rng(3)
sgt = SGTRegressor()
mae_sgt, mae_mean, mean, n = MeanAbsoluteError(), MeanAbsoluteError(), 0.0, 0
for _ in range(20000):
    q1, q2 = rng.uniform(-np.pi, np.pi, 2)
    tau = 9.81 * (2.0 * np.cos(q1) + np.cos(q1 + q2))
    x = {"q1": q1, "q2": q2}
    mae_sgt.update(tau, sgt.predict_one(x))
    mae_mean.update(tau, mean)
    sgt.learn_one(x, tau)
    n += 1
    mean += (tau - mean) / n
print(round(mae_sgt.get(), 3), round(mae_mean.get(), 3), sgt.n_nodes_)
```

```
7.715 13.241 91
```

Tested before learning each sample, the tree's mean absolute error is 7.7 N·m against
13.2 N·m for the running mean, with 91 nodes. A split is made only when a t-test says the loss
reduction is real (Gouk et al. 2019, eq. 16):

$$t = \frac{\bar L}{s / \sqrt{n}}$$

$\bar L$ is the mean change of the loss over the $n$ samples of the leaf, $s$ its standard
deviation; the split happens when the one-sided p-value is below `delta` (0.01). The variance of
each sample's change uses the gradient $G$, the Hessian $H$ and the value $v$ of the new leaf
(eq. 18):

$$\operatorname{Var}(L_i) = v^2 \operatorname{Var}(G) + \tfrac14 v^4 \operatorname{Var}(H) + v^3 \operatorname{Cov}(G, H)$$

## 5. A prediction with its uncertainty: Mondrian forests

A forest of 10 random trees that also says how sure it is; trained on joint positions between
0 and 1 only, it is asked inside and outside that range.

```python
import numpy as np
from dense_armor.utility.tree import MondrianForestRegressor

rng = np.random.default_rng(4)
mf = MondrianForestRegressor(n_trees=10, seed=0)
for _ in range(1000):
    q = rng.uniform(0.0, 1.0)
    mf.learn_one({"q": q}, np.sin(6.0 * q) + rng.normal(0.0, 0.05))
for q in (0.5, 1.5, 3.0):
    e = mf.predict_one({"q": q}, return_estimate=True)
    print(q, round(e.mean, 3), round(e.std, 3))
```

```
0.5 0.131 0.054
1.5 -0.219 0.7
3.0 -0.055 0.723
```

At `q = 0.5` (inside the data) the prediction 0.131 is close to the true `sin(3) = 0.141`, with
standard deviation 0.054, about the noise of the data. Outside the data the standard deviation
jumps to 0.70: the forest says "I do not know", which a controller can use to slow down or ask
for help. Where it comes from: walking down a tree, a point outside a node's box "branches off"
before that node with probability (Lakshminarayanan et al. 2016, Algorithm 5)

$$p_j(x) = 1 - e^{-\Delta_j \eta_j(x)}, \qquad \Delta_j = \tau_j - \tau_{\mathrm{parent}(j)}$$

$\eta_j(x)$ is how far $x$ lies outside the box, $\tau$ the split times of the Mondrian process;
the prediction is the mixture of the node statistics weighted by these probabilities (eq. 3), so
the farther the point, the wider the mixture.

## 6. Which tree for which job

The same streams for every model, 4000 samples each; "change" = the meaning of the signal
changes at sample 2000 (classes swap; for the torque, gravity grows by 50 %). Tested before
learning each sample.

| model | task | stationary | with a change | ms per sample |
|---|---|---|---|---|
| `HoeffdingTreeClassifier` | accuracy | 0.986 | 0.542 | 0.015 |
| `HoeffdingAnytimeTreeClassifier` | accuracy | 0.986 | 0.654 | 0.013 |
| `HoeffdingAdaptiveTreeClassifier` (`ADWIN`) | accuracy | 0.986 | 0.756 | 0.451 |
| `SGTClassifier` | accuracy | 0.844 | 0.774 | 0.023 |
| `MondrianForestClassifier` | accuracy | 0.997 | 0.591 | 0.959 |
| `SGTRegressor` | mean abs. error | 10.146 | 12.344 | 0.022 |
| `MondrianForestRegressor` | mean abs. error | 1.113 | 3.265 | 0.944 |

On a stable stream the three Hoeffding trees and the Mondrian forest classify almost perfectly;
after a change, the adaptive tree recovers best among the trees, at about 30 times the cost per
sample of the plain tree; the Mondrian forest is the most accurate regressor here and the only
model with an uncertainty, at about 1 ms per sample; the stochastic gradient tree is the
cheapest regressor. Accuracy "with a change" counts the whole stream, before and after the
change.

## API reference

::: dense_armor.utility.tree.hoeffding
::: dense_armor.utility.tree.efdt
::: dense_armor.utility.tree.adaptive
::: dense_armor.utility.tree.sgt
::: dense_armor.utility.tree.mondrian

---

## Details

- Hoeffding tree: each leaf keeps a Gaussian per class and feature; candidate thresholds are
  midpoints between consecutive class means; leaves predict with naive Bayes (`leaf="nb"`) or
  the majority class; `tau` (0.05) is the tie threshold. Domingos and Hulten (2000) introduced
  the tree; the bound and the Anytime variant are from Manapragada et al. (2018), Algorithm 3.1,
  Functions 3.2 and 3.3.
- Adaptive tree: a detector on each internal node (default `ADWIN`, cloned per node); swap and
  discard are decided only after `kappa_alt` (100) samples of the background subtree, on errors
  counted on the same samples. As in Esteban et al. (2024), the detectors sit on the internal
  nodes.
- Stochastic gradient trees: features are cut into 64 equal-width bins, their range estimated on
  the first 1000 samples (section 3.2); per-bin moments of gradient and Hessian are merged with
  the pairwise update before eq. 18; the t distribution is computed with the regularized
  incomplete beta function, no external library at runtime; `SGTClassifier` is binary.
- Mondrian forests: split times follow Algorithm 2 (leaves have time equal to the lifetime) and
  new points extend the tree with Algorithm 4 of the 2014 paper, so online and batch trees have
  the same distribution (eq. 1 there); every node keeps the statistics of its block. With the
  default lifetime (infinite) a tree keeps growing until `max_nodes`. Departure from the paper:
  classification uses normalised class histograms in the mixture instead of the hierarchical
  smoothing of the paper, and blocks are not paused when their labels are all identical.
- Every class caps its size with `max_nodes` (default 10000), so memory is bounded; `budget_s`
  is about ten times the 99th percentile of `learn_one` measured on a desktop CPU.
- Sources: Manapragada, C., Webb, G. I., Salehi, M. (2018), "Extremely fast decision tree",
  arXiv:1802.08780. Esteban, A., Cano, A., Zafra, A., Ventura, S. (2024), "Hoeffding adaptive
  trees for multi-label classification on data streams", arXiv:2410.20242. Gouk, H.,
  Pfahringer, B., Frank, E. (2019), "Stochastic gradient trees", arXiv:1901.07777.
  Lakshminarayanan, B., Roy, D. M., Teh, Y. W. (2014), "Mondrian forests: efficient online
  random forests", arXiv:1406.2673; (2016) "Mondrian forests for large-scale regression when
  uncertainty matters", arXiv:1506.03805.
