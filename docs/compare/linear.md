# Linear: machine learning against online learning

The machine learning classes read the whole table and the online classes read one row at a time. They are two different parts of the library; this page runs both on the same synthetic data and reports how far apart their outputs are.

A linear model predicts a number, or the probability of a class, as a weighted sum of the features. Here the weights are found on the whole table: by one linear solve for the regression, by Newton steps for the logistic model, and as a posterior distribution for the Bayesian regression.

## The difference, on synthetic data

10 seeds. `abs` is the largest absolute difference between the two outputs, `rel` the largest relative difference; median, 90th percentile and maximum over the seeds.

| class | kind | median | p90 | max | why |
|---|---|---|---|---|---|
| `LinearRegression` | abs | 0.0334 | 0.0558 | 0.0651 | the online model takes one stochastic gradient step per row, the batch model solves the problem exactly |
| `LogisticRegression` | abs | 0.452 | 0.48 | 0.519 | the online model takes one stochastic gradient step per row, the batch model runs Newton steps to the optimum |
| `BayesianLinearRegression` | abs | 3.55e-15 | 4.49e-15 | 4.88e-15 | the same conjugate posterior: the two differ only by rounding |

Wilcoxon and A12: not computed: the classic and the online value are two outputs of the same formula on the same data, so the table gives their difference and not a ranking of the two.

## Run it

~~~bash
python benchmarks/compare/linear.py
~~~

The command writes `benchmarks/compare/linear.json`, the table above.
