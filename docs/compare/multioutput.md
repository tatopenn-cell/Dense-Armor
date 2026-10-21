# Multioutput: machine learning against online learning

The machine learning classes read the whole table and the online classes read one row at a time. They are two different parts of the library; this page runs both on the same synthetic data and reports how far apart their outputs are.

Some problems have several targets at once, for example the position and the speed of a joint. These classes build one model per target, or a chain in which each model also sees the targets of the earlier links, or a chain that finds the most probable set of labels as a whole.

## The difference, on synthetic data

10 seeds. `abs` is the largest absolute difference between the two outputs, `rel` the largest relative difference; median, 90th percentile and maximum over the seeds.

| class | kind | median | p90 | max | why |
|---|---|---|---|---|---|
| `PerOutputRegressor` | abs | 0.0248 | 0.0406 | 0.0628 | the online regressors take one stochastic step per row; the batch ones solve exactly |
| `RegressorChain` | abs | 0.032 | 0.0522 | 0.0628 | the online regressors take one stochastic step per row; the batch ones solve exactly |
| `PerOutputClassifier` | abs | 0.015 | 0.0255 | 0.03 | difference between the test accuracies; the online models see each row once |
| `ClassifierChain` | abs | 0.0125 | 0.0215 | 0.035 | difference between the test accuracies; the online models see each row once |
| `ProbabilisticClassifierChain` | abs | 0.015 | 0.026 | 0.035 | difference between the test accuracies; the online models see each row once |

Wilcoxon and A12: not computed: the classic and the online value are two outputs of the same formula on the same data, so the table gives their difference and not a ranking of the two.

## Run it

~~~bash
python benchmarks/compare/multioutput.py
~~~

The command writes `benchmarks/compare/multioutput.json`, the table above.
