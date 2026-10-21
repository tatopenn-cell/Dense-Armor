# Proba: machine learning against online learning

The machine learning classes read the whole table and the online classes read one row at a time. They are two different parts of the library; this page runs both on the same synthetic data and reports how far apart their outputs are.

A distribution summarises a sample in a few numbers. These classes keep the posterior of the Beta, Gaussian and Multinomial families: the prior counts plus the counts observed in the whole sample.

## The difference, on synthetic data

10 seeds. `abs` is the largest absolute difference between the two outputs, `rel` the largest relative difference; median, 90th percentile and maximum over the seeds.

| class | kind | median | p90 | max | why |
|---|---|---|---|---|---|
| `Beta` | abs | 3.9e-13 | 1.2e-12 | 2.02e-12 | the same conjugate update, counted one row at a time |
| `Gaussian` | abs | 3.11e-15 | 9.06e-15 | 1.07e-14 | the same conjugate update, counted one row at a time |
| `Multinomial` | abs | 0 | 0 | 0 | the same conjugate update, counted one row at a time |

Wilcoxon and A12: not computed: the classic and the online value are two outputs of the same formula on the same data, so the table gives their difference and not a ranking of the two.

## Run it

~~~bash
python benchmarks/compare/proba.py
~~~

The command writes `benchmarks/compare/proba.json`, the table above.
