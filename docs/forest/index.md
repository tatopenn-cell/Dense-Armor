# Streaming forests: adaptive ensembles and online structural aggregation

A single decision tree relies on a single rule, which breaks the moment a robot's conditions change. A forest combines multiple trees to outvote single-tree errors, using background adaptive detectors to replace failing branches or weighting different tree depths based on their recent prediction accuracy to handle noise and sudden shifts.

## The map of forests

Click a page to open it, or the job on the left (or the button at the top) to go back to the [map of the library](../index.md#the-library-folder-by-folder).

<object type="image/svg+xml" data="../assets/maps/forest.svg" style="width:100%"></object>

## Gamestory

New Year's Eve in London. A mysterious thief is emptying the banks of the city.

So far on his trail there was only one detective. The problem is that if the thief changes trick all of a sudden, the old trail becomes useless: a detective alone has to notice the mistake, forget everything and start over, with no one to lean on.

For this reason an entire team (a Forest) steps in that splits the risk. Every detective keeps an eye on his own mistakes: when one starts losing ground, he immediately prepares a replacement behind the scenes, while the team keeps chasing the thief with the detectives that still work.

## A first example: payments whose rule changes

Half of the payments are fraud. At first a fraud is a large payment (around 400) and a normal one is small (around 60); after 750 payments they swap. A forest of 5 trees predicts each payment before it learns it, and the accuracy is counted before the change (payments 500 to 750), over the 100 payments after it, and late in the stream (the last 200).

```python
import numpy as np
from dense_armor.core.online.forest import AdaptiveRandomForestClassifier

rng = np.random.default_rng(0)
forest = AdaptiveRandomForestClassifier(n_models=5, seed=1)
hits = []
for i in range(1500):
    fraud = int(rng.random() < 0.5)
    big = fraud if i < 750 else 1 - fraud
    amount = float(rng.normal(400 if big else 60, 40 if big else 15))
    hits.append(forest.predict_one({"amount": amount}) == fraud)
    forest.learn_one({"amount": amount}, fraud)
h = np.array(hits, float)
print(h[500:750].mean().round(3), h[750:850].mean().round(3), h[1300:].mean().round(3))
```

```
1.0 0.92 1.0
```

The three numbers are the accuracy before the change, over the 100 payments after it, and late in the stream. Every tree keeps an eye on its own mistakes; the trees that start to fail are replaced in the background while the others keep voting, which is what the page of the [adaptive random forest](adaptive_random_forest.md) describes.

## The pages of the section

| Page | Classes | What it is for |
|---|---|---|
| [AdaptiveRandomForestClassifier](adaptive_random_forest.md) | `AdaptiveRandomForestClassifier`, `SubspaceHoeffdingTree` | a forest of Hoeffding trees that replaces the trees that go wrong |
| [AggregatedMondrianForest](aggregated_mondrian_forest.md) | `AggregatedMondrianForestClassifier`, `AggregatedMondrianForestRegressor` | the average of every depth of a tree, each depth weighted by its past loss |
| [OnlineExtraTreesRegressor](online_extra_trees.md) | `OnlineExtraTreesRegressor` | a regression forest that draws the split at random instead of searching for it |

## The numbers

`DriftStream`, one sudden change at half, 2000 samples, four features, test-then-train, median of five seeds. The seeds are the first five (2, 3, 6, 10 and 12) for which a tree that learns until the change and then stops is wrong after it, with an accuracy below 0.5: that tree is the last row of the table, and it shows that the change makes an old rule wrong. Accuracy (mean squared error of the label for the regressors) on the 300 samples before the change, on the first 100 after it, and on 300 samples starting 200 after it.

| learner | before | first 100 | after + 200 | ms per sample |
|---|---|---|---|---|
| AdaptiveRandomForestClassifier | 1.000 | 0.940 | 1.000 | 0.97 |
| HoeffdingAdaptiveTreeClassifier | 1.000 | 0.740 | 1.000 | 0.11 |
| AggregatedMondrianForestClassifier | 1.000 | 0.930 | 1.000 | 3.87 |
| MondrianForestClassifier | 1.000 | 0.940 | 1.000 | 1.77 |
| AggregatedMondrianForestRegressor (MSE) | 0.001 | 0.157 | 0.015 | 3.25 |
| MondrianForestRegressor (MSE) | 0.000 | 0.046 | 0.003 | 2.06 |
| OnlineExtraTreesRegressor (MSE) | 0.002 | 0.049 | 0.006 | 1.22 |
| AMRulesRegressor (MSE) | 0.039 | 0.249 | 0.047 | 0.11 |
| a tree that stops learning at the change | 1.000 | 0.240 | 0.183 | |

The rows of the two regressors OXT and AMRules come from `benchmarks/oxt_streaming.py`, which also runs the Mondrian forest regressor (0.000, 0.046, 0.003 and 1.87 ms per sample); the others come from `benchmarks/forest_streaming.py`.

On the first 100 samples after the change the adaptive random forest is right 0.94 of the time and the single adaptive tree 0.74; on five seeds the forest is better on four and equal on one (paired one-sided Wilcoxon p = 0.062). The forest has ten trees against one, which is part of the difference and of the cost. The aggregated Mondrian classifier scores 0.93 against 0.94 for the Mondrian forest already in the library, and the aggregated regressor has a mean squared error of 0.157 against 0.046 for the Mondrian regressor; both aggregated forests cost more per sample, and the aggregation needs a box around the inputs, taken here from the range of the whole stream.

OXT has a mean squared error of 0.049 in the first 100 samples after the change against 0.046 for the Mondrian forest, at 1.22 ms against 1.87 ms per sample (the OXT forest has twenty trees, the Mondrian one ten). Seed by seed OXT is lower than the Mondrian forest on three seeds of five (one-sided paired Wilcoxon with OXT lower, p = 0.406). Against AMRules OXT is lower on three seeds of five (p = 0.156), and it is slower (1.22 ms against 0.11 ms); AMRules is far off on one seed (2.592 against 0.060). Neither difference is significant at 0.05, and with five seeds no p-value can fall below 0.031 (one-sided). The random split is not the winner on this benchmark; its purpose is to remove the split search.

### Go or no-go for AMF and OXT

Whether `AggregatedMondrianForestRegressor` (AMF) and `OnlineExtraTreesRegressor` (OXT) earn a place next to `MondrianForestRegressor` was decided on five regimes, 20 seeds each, 1000 samples, mean squared error over the second half of the stream, prequential. The rule, fixed before the run: GO: lower error than the Mondrian forest with two-sided paired Wilcoxon p < 0.05 and the 95% bootstrap interval of A12 above 0.5, or the same error (interval contains 0.5) at no more than half the ms per sample. The script is `benchmarks/z_go_nogo.py` and the numbers are in `benchmarks/z_go_nogo.json` (run on Intel(R) Xeon(R) CPU @ 2.20GHz, 4 cores).

| regime | model | median error | Mondrian forest | A12 [95% interval] | seeds better | p (two-sided) | ms per sample, model / Mondrian | verdict |
|---|---|---|---|---|---|---|---|---|
| 20 features, 2 relevant | AMF | 0.3584 | 0.2657 | 0.04 [0.00, 0.09] | 0/20 | 1.9e-06 | 2.92 / 3.91 | NO-GO |
| 20 features, 2 relevant | OXT | 0.1839 | 0.2657 | 0.97 [0.94, 1.00] | 20/20 | 1.9e-06 | 4.15 / 3.91 | GO |
| 100 features, 2 relevant | AMF | 0.4122 | 0.3912 | 0.34 [0.21, 0.46] | 4/20 | 0.0083 | 2.93 / 16.85 | NO-GO |
| 100 features, 2 relevant | OXT | 0.3512 | 0.3912 | 0.66 [0.50, 0.82] | 14/20 | 0.28 | 4.72 / 16.85 | NO-GO |
| piecewise function of one feature | AMF | 0.0593 | 0.0313 | 0.00 [0.00, 0.00] | 0/20 | 1.9e-06 | 2.86 / 1.32 | NO-GO |
| piecewise function of one feature | OXT | 0.0323 | 0.0313 | 0.36 [0.23, 0.49] | 5/20 | 0.07 | 3.92 / 1.32 | NO-GO |
| 5% outliers | AMF | 1.4647 | 1.4662 | 0.45 [0.36, 0.48] | 3/20 | 0.0017 | 2.81 / 1.32 | NO-GO |
| 5% outliers | OXT | 1.4526 | 1.4662 | 0.52 [0.46, 0.58] | 14/20 | 0.07 | 3.89 / 1.32 | NO-GO |
| limit of 1 ms per sample (3 trees, 5 for the Mondrian forest) | AMF | 0.1139 | 0.0433 | 0.00 [0.00, 0.00] | 0/20 | 1.9e-06 | 0.94 / 0.69 | NO-GO |
| limit of 1 ms per sample (3 trees, 5 for the Mondrian forest) | OXT | 0.0802 | 0.0433 | 0.00 [0.00, 0.01] | 0/20 | 1.9e-06 | 1.26 / 0.69 | NO-GO |

AMF is worse than the Mondrian forest in all five regimes, with a paired p below 0.01 in each; in the regime with 5% outliers the medians differ only in the fourth decimal (1.4647 against 1.4662) and AMF is better on 3 seeds out of 20. OXT is better in one regime only, 20 features with 2 relevant, where it is better on 20 seeds out of 20 (error 0.184 against 0.266) at about the same cost; with 100 features the difference is not significant (p = 0.28), on a piecewise function it ties, and under a 1 ms limit it loses. The decision: keep OXT as an option for streams with many irrelevant features, and do not recommend AMF over the Mondrian forest.

## Details

### References

- Gomes, H. M., Bifet, A., Read, J., Barddal, J. P., Enembreck, F., Pfahringer, B., Holmes, G., Abdessalem, T. (2017). Adaptive random forests for evolving data stream classification. Machine Learning 106, 1469-1495. Algorithms 1 and 2 (pages 7 and 8 of the PDF), the detector settings on page 17.
- Mourtada, J., Gaiffas, S., Scornet, E. AMF: Aggregated Mondrian forests for online learning. arXiv:1906.10529v2, the manuscript version of the PDF used. Equations 1, 5, 6 and 7, Algorithms 2 and 3, Lemma 1, Corollaries 1 and 2.
- Lakshminarayanan, B., Roy, D. M., Teh, Y. W. (2014). Mondrian forests: efficient online random forests. NIPS.

### Choices the papers do not fix

- **Detector thresholds.** The paper compares a slow setting (`delta_w = 1e-4`, `delta_d = 1e-5`) and a fast one (`delta_w = 0.01`, `delta_d = 0.001`). The default here is the fast one, which reacts inside streams of a few thousand samples.
- **Feature subset.** The paper draws the subset when a node is created; here it is drawn at the first split attempt of the node, from the features seen there, and kept for the later attempts. The size is `int(sqrt(M)) + 1` unless `max_features` is given.
- **Drift without a warning.** The tree is replaced by a new empty tree, since there is no background tree to promote.
- **The vote.** It adds the class probabilities of each tree, weighted by `c / n`, and normalises. When every weight is zero the trees count equally.
- **Parameter names of the aggregated forest.** `step` is the learning rate `eta` of the paper and `dirichlet` is the parameter `a` of the Dirichlet prior of a node forecaster, `(n_y + a) / (n + K a)`; `a = 1/2` is the Krichevsky-Trofimov forecaster. The paper uses neither name.
- **Equation 5.** The paper prints `t - 1` in the denominator of the forecaster. The version that sums to one over the labels of a node uses the count of that node, which is what is implemented.
- **Labels and inputs.** The set of labels grows with the stream unless `classes` fixes it. Inputs are clipped to the box `[low, high]`.
- **Random predictions.** As in the paper, a prediction draws the splits that would separate the query from the sample in its cell and discards them, so `predict_one` is random and leaves the trees unchanged.

### What is implemented

The aggregated forests implement the version of Section 2 of the paper, on an unrestricted Mondrian partition (Algorithm 2). The restricted partition of Section 4, which the paper uses in its experiments and for which it states no guarantee, is not implemented. The regret bounds quoted above are those of the implemented version, for a tree below its `max_nodes` (default 10000 per tree); once a tree has that many nodes it stops splitting, and the bounds are not claimed for what it predicts after that.

### Checks

- The collapsed weights equal the exponentially weighted sum of equation 6 taken over every pruning of a small tree, with a difference below 1e-9 for `step` from 0.125 to 200 (a leaf of the finished tree stands for all its refinements, which have the same loss, so its prior mass is 1).
- Lemma 1, Corollary 1 and Corollary 2 hold on the prequential streams of `test/core/online/forest/test_amf.py`.
- On a noiseless step function the aggregated prediction with `step = 1e4` equals the leaf prediction. On a noisy curve the two differ: with one tree, 300 training samples, queries at 100 of them, and a large `step`, the two differ by at most 0.19 (median 0.04). A large `step` selects the best pruning in hindsight, which is the deepest tree only when the data is clean.
- `check_estimator` passes for the three classes with no check skipped. They have no pure `init_state` / `step` form, since the state is a set of trees of changing shape.
- After `load_state_dict`, a pickle round trip and `clone(include_attributes=True)`, an aggregated classifier still accepts a label it has not seen and returns probabilities that sum to one.

### Cost and memory

p99 of one `learn_one` plus one `predict_one` over 3000 samples with four features, on a Kaggle CPU: 9.4 ms for the adaptive random forest of ten trees, 4.3 ms for the aggregated classifier and 4.0 ms for the aggregated regressor; `budget_s` is 0.1 s for the adaptive forest and 0.04 s for the two aggregated forests, about ten times the measured p99.

The trees of the adaptive random forest have no node cap by default: the statistics of each tree are bounded by `max_active_leaves` (the least promising leaves are deactivated), so they are constant in the length of the stream, and `max_nodes` or `max_depth` can still be set. The aggregated trees are bounded by `max_nodes`, so their state is constant in the length of the stream. The aggregated trees keep two weights per node and, on uniform inputs with 1, 2 and 4 features, between 3.6 and 4.0 nodes per sample until they reach the limit. The depth of a tree on two features was 21, 30 and 43 after 100, 1000 and 4000 samples.

### Development diary

The first test of the collapsed weights against the sum over prunings differed by 0.77. The cause was in the test: the sum of the paper runs over the subtrees of the infinite Mondrian tree, so a leaf of the finished tree has to carry the weight of all its refinements. With that correction the two agree to about 4e-15.

The first test of the adaptive forest expected a drift after a plain label swap. None came: the trees relearn from the weighted samples within about 50 samples, before the drift detector at `delta = 0.001` collects enough errors. The warnings came, and the accuracy recovered.
