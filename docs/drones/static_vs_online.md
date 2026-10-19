# Static vs online: a CART tree and an online Hoeffding tree on the same stream

## The scene

Two thousand samples of synthetic flight, sixteen features per sample (gyro x, y, z; acc x, y, z; gps velocity north, east, down; motor 1-4; battery voltage; current; payload). One actuator loses power at 60% of the stream; the label is the fault flag. One batch tree, one online tree, ten seeds. The question is: how much accuracy does the batch tree buy, and at what cost in time and memory?

## What is in the box

| Side | Class | Protocol |
|---|---|---|
| Batch | `BatchDecisionTreeClassifier` (`dense_armor/utility/tree/batch.py`) | `fit(X, y)` once on the whole training half, then `predict_one` on the test half |
| Online | `HoeffdingTreeClassifier` (`dense_armor/utility/tree/hoeffding.py`, `grace_period=50`) | `predict_one` then `learn_one` on each test sample, in order |

The batch tree is a CART tree: gini impurity `I(t) = 1 - sum_k p_k^2`, recursive binary split (Breiman et al. 1984; also Bourel et al. 2026, arXiv:2606.13984, page 8 and page 14). The online tree is a Hoeffding tree (Hoeffding 1963). Both are classifiers of the library; the pipeline only splits, scores, and compares.

## 1. The two protocols, side by side

The batch tree sees the whole training half in one call to `fit`, then never changes. The online tree sees one test sample at a time: first `predict_one`, then `learn_one`, in the shuffled order of the test half. On the same toy set of four samples the two agree on the split:

```python
from dense_armor.utility.tree.batch import BatchDecisionTreeClassifier

X = [{"x": 0.0}, {"x": 1.0}, {"x": 2.0}, {"x": 3.0}]
y = [0, 0, 1, 1]
clf = BatchDecisionTreeClassifier(max_depth=2).fit(X, y)
print(clf.predict_one({"x": 0.5}), clf.predict_one({"x": 2.5}))
```

```
0 1
```


## 2. The data, and the split

Training and test are a 50/50 shuffled split of the stream. The fault starts at 60% of the stream, so the post-fault part is 40% of the samples; both halves contain both classes. The batch fits on the training half; the online starts empty and walks the test half. Both are scored on the same test half, same samples, same order. Ten seeds, `max_depth=3` for the CART tree.

## 3. The result, on 10 seeds

The batch wins on every seed. The gap is small in absolute terms because the fault is a strong signal, but it is consistent: 1.000 against 0.992 median.

```python
from pipelines.drones.static_vs_online import run

r = run(n_seeds=10, n_samples=2000, max_depth=3)
print(f"batch={r.median_accuracy_batch:.4f} online={r.median_accuracy_online:.4f}")
print(f"p={r.p_two_sided:.4f} A12={r.a12:.3f} CI=({r.a12_ci95[0]:.3f},{r.a12_ci95[1]:.3f})")
print(f"seeds batch_better={r.seeds_batch_better} online_better={r.seeds_online_better}")
```

```
batch=1.0000 online=0.9920
p=0.0020 A12=1.000 CI=(1.000,1.000)
seeds batch_better=10 online_better=0
```

Paired two-sided Wilcoxon on the 10 per-seed accuracies: **p = 0.0020**. A12 effect size: **1.000**, bootstrap 95% interval **(1.000, 1.000)**. With 10 seeds the smallest p Wilcoxon can report is 0.002, and it reports it: the batch wins on every seed, so the two-sided test is at its floor.

## 4. The cost

The batch tree scans the whole training set once; the online tree touches one sample at a time. On the full run the batch costs **5.09 ms per sample** (fit on the training half plus predict on the test half) and the online costs **0.018 ms per sample** (predict plus learn). The ratio is about 280x. `run(...)` also returns the change of the process RSS around each model (`memory_batch_kb`, `memory_online_kb`). The value is a delta, not a peak, and it can be negative: the process can release memory between the two reads. On 2000 samples it is under the page resolution of `psutil`, so this page does not use it as a comparison.

The cost per sample is not reproduced in a block because it depends on the load of the host: the same 1-seed run measured 5.87 ms and 2.43 ms in two consecutive executions. The numbers above are the medians over the 10 seeds of the full run, and they are the ones the comparison uses.


## Details

### References

- Breiman, L., Friedman, J., Olshen, R., Stone, C. (1984). *Classification and Regression Trees.* Wadsworth. Chapter 2: gini impurity and recursive binary splitting. The book is not on arXiv and was not read for this page; cited bibliographically.
- Bourel, M. et al. (2026). *A Bregman Perspective on Classification and Regression Trees.* arXiv:2606.13984. Page 8 and page 14: the Gini index I(t) = 1 - sum_k p_k^2, which is the criterion used by `BatchDecisionTreeClassifier._best_split`.
- Hoeffding, W. (1963). *Probability inequalities for sums of bounded random variables.* Journal of the American Statistical Association 58(301):13-30. Pre-arXiv; cited bibliographically.
- Oza, N. C., Russell, S. (2001). *Online Bagging and Boosting.* In AISTATS 2001, 105-112. Section 2, page 2: the batch bagging, which is what `OnlineBaggingClassifier.fit_batch` implements (kept in the library, not used on this page).
- Wilcoxon, F. (1945). *Individual comparisons by ranking methods.* Biometrics Bulletin 1(6):80-83. Pre-arXiv; cited bibliographically.
- Vargha, A., Delaney, H. D. (2000). *A critique and improvement of the CL common language effect size statistics of McGraw and Wong.* Journal of Educational and Behavioral Statistics 25(2):101-132. Pre-arXiv; cited bibliographically.

### Choices stated explicitly

- 10 seeds, not 20: a single seed takes about 11 s on this host (mostly the online Hoeffding loop), so 20 seeds would take about 4 minutes. With 10 seeds the smallest p Wilcoxon can report is 0.002, and the test reaches it.
- 2000 samples per stream, 50/50 shuffled split, 40% of the samples after the fault. Both halves contain both classes.
- `max_depth=3` for the CART tree: the fault is a single strong feature, depth 3 is enough to separate the two classes, and the CART fit drops from 0.08 s (depth 6) to 0.02 s (depth 3).
- `grace_period=50` for the Hoeffding tree: the library default, kept for both the online side of this page and the online side of the comparison.
- The memory is the change of the process RSS around each model, not a peak: `psutil` reads the RSS before and after, and the value can be negative. On 2000 samples it is under the page resolution, so the page does not compare the two models on it.

### Measured cost

On 10 seeds and 2000 samples, the full `run(...)` takes about 113 s on this host. Per sample: 5.09 ms for the batch model, 0.018 ms for the online model. The block in section 4 reproduces the ratio on 1000 samples and 1 seed in about 6 s.

### Development diary

The first version of the pipeline compared `OnlineBaggingClassifier.fit_batch` against `OnlineBaggingClassifier.learn_one`. That is the same online tree fed in another order, not a static model. The second version replaced the batch side with `BatchDecisionTreeClassifier`, a CART tree fitted on the whole training set in one call. The online side stayed on `HoeffdingTreeClassifier` with `grace_period=50`. The comparison now has a static model, a statistic (paired Wilcoxon, A12 with bootstrap CI), and the cost per sample.

### Reproduce

```bash
python -m pytest test/drones/test_static_vs_online.py -q
```
