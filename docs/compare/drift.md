# Drift: machine learning against online learning

The machine learning classes read the whole table and the online classes read one row at a time. They are two different parts of the library; this page runs both on the same synthetic data and reports how far apart their outputs are.

A change detector reads a series of values and says where its level shifts. CUSUMDriftDetector fixes the reference level from the first values and adds up the deviations that follow until the sum crosses a threshold.

## The difference, on synthetic data

10 seeds. `abs` is the largest absolute difference between the two outputs, `rel` the largest relative difference; median, 90th percentile and maximum over the seeds.

| class | kind | median | p90 | max | why |
|---|---|---|---|---|---|
| `CUSUMDriftDetector` | abs | 0 | 0 | 0 | the same fixed-reference recursion of Page, read over the whole series |

Wilcoxon and A12: not computed: the machine learning and the online value are two outputs of the same formula on the same data, so the table gives their difference and not a ranking of the two.

## Run it

~~~bash
python benchmarks/compare/drift.py
~~~

The command writes `benchmarks/compare/drift.json`, the table above.
