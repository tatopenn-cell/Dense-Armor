# Bayes: machine learning against online learning

The machine learning classes read the whole table and the online classes read one row at a time. They are two different parts of the library; this page runs both on the same synthetic data and reports how far apart their outputs are.

A naive Bayes classifier counts how often each feature value goes with each class and turns the counts into probabilities. Here the counts are taken over the whole table, with a small pseudo-count so that no probability is zero.

## The difference, on synthetic data

10 seeds. `abs` is the largest absolute difference between the two outputs, `rel` the largest relative difference; median, 90th percentile and maximum over the seeds.

| class | kind | median | p90 | max | why |
|---|---|---|---|---|---|
| `MultinomialNB` | abs | 2.22e-16 | 6.44e-16 | 9.44e-16 | the same counts and the same smoothing |
| `BernoulliNB` | abs | 3.05e-16 | 3.33e-16 | 3.33e-16 | the same counts and the same smoothing |
| `CategoricalNB` | abs | 5.55e-16 | 6.91e-16 | 9.16e-16 | the same counts and the same smoothing |
| `ComplementNB` | abs | 1.67e-16 | 1.72e-16 | 2.22e-16 | the same counts and the same smoothing |

Wilcoxon and A12: not computed: the machine learning and the online value are two outputs of the same formula on the same data, so the table gives their difference and not a ranking of the two.

## Run it

~~~bash
python benchmarks/compare/bayes.py
~~~

The command writes `benchmarks/compare/bayes.json`, the table above.
