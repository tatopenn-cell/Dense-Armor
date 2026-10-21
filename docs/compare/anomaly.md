# Anomaly: machine learning against online learning

The machine learning classes read the whole table and the online classes read one row at a time. They are two different parts of the library; this page runs both on the same synthetic data and reports how far apart their outputs are.

An anomaly detector marks the values that do not look like their neighbours. Four rules (Hampel, Tukey, Chauvenet, sigma clipping) judge each value of a series against its centred window; the Local Outlier Factor compares the density around a row with the density around its neighbours.

## The difference, on synthetic data

10 seeds. `abs` is the largest absolute difference between the two outputs, `rel` the largest relative difference; median, 90th percentile and maximum over the seeds.

| class | kind | median | p90 | max | why |
|---|---|---|---|---|---|
| `HampelFilter` | abs | 0 | 0 | 0 | the same rule on the same centred window |
| `TukeyScorer` | abs | 0 | 0 | 0 | the same rule on the same centred window |
| `ChauvenetScorer` | abs | 0 | 0 | 0 | the same rule on the same centred window |
| `SigmaClipScorer` | abs | 0 | 0 | 0 | the same rule on the same centred window |
| `LocalOutlierFactor` | abs | 0 | 0 | 0 | the same definitions, with the whole table as the reference set |

Wilcoxon and A12: not computed: the machine learning and the online value are two outputs of the same formula on the same data, so the table gives their difference and not a ranking of the two.

## Run it

~~~bash
python benchmarks/compare/anomaly.py
~~~

The command writes `benchmarks/compare/anomaly.json`, the table above.
