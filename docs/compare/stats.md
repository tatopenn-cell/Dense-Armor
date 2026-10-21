# Stats: machine learning against online learning

The machine learning classes read the whole table and the online classes read one row at a time. They are two different parts of the library; this page runs both on the same synthetic data and reports how far apart their outputs are.

A statistic such as a mean, a variance or a correlation is computed on a whole table at once. This page lists the statistics of the machine learning part of the library: covariance and correlation, moments, the exponentially weighted mean and variance, rolling windows, the rolling median and the shrunk covariance.

## The difference, on synthetic data

10 seeds. `abs` is the largest absolute difference between the two outputs, `rel` the largest relative difference; median, 90th percentile and maximum over the seeds.

| class | kind | median | p90 | max | why |
|---|---|---|---|---|---|
| `RunningCovariance` | abs | 4.44e-16 | 8.99e-16 | 9.99e-16 |  |
| `RunningCorrelation` | abs | 3.05e-16 | 5.77e-16 | 7.77e-16 |  |
| `RollingCovariance` | abs | 0 | 2.22e-16 | 2.22e-16 |  |
| `RollingCorrelation` | abs | 5.55e-17 | 1.17e-16 | 1.67e-16 |  |
| `Autocorrelation` | abs | 0.0128 | 0.0251 | 0.0296 | the online estimator differs from the textbook sample autocorrelation |
| `RunningMoments` | abs | 8.88e-15 | 1.08e-14 | 1.24e-14 |  |
| `RunningMomentsVector` | abs | 9.99e-16 | 1.42e-15 | 2.22e-15 |  |
| `EWStats` | abs | 5e-16 | 1.13e-15 | 1.33e-15 |  |
| `RollingMedianVector` | abs | 0 | 0 | 0 |  |
| `OnlineShrinkageCovariance` | abs | 2.66e-15 | 7.46e-15 | 1.07e-14 |  |

Wilcoxon and A12: not computed: the machine learning and the online value are two outputs of the same formula on the same data, so the table gives their difference and not a ranking of the two.

## Run it

~~~bash
python benchmarks/compare/stats.py
~~~

The command writes `benchmarks/compare/stats.json`, the table above.
